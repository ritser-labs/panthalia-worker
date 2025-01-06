# file: spl/tests/test_holds_flow.py

import pytest
import json
from datetime import datetime, timezone
from spl.db.server.app import original_app
from spl.models.enums import TaskStatus, OrderType
from sqlalchemy import select

@pytest.mark.asyncio
async def test_holds_flow_end_to_end(db_adapter_server_fixture):
    """
    This test ensures the Holds system works correctly from initial deposit,
    through bidding & asking, final settlement, and checks leftover amounts
    and Freed/Charged logic in Holds are correct.

    Steps:
     1) Admin deposit for buyer => verify leftover in Credits hold.
     2) Admin deposit for solver => verify leftover in Credits hold.
     3) Create plugin, subnet, job => basic.
     4) Create a new Task => status=SelectingSolver.
     5) Buyer places a Bid => verify buyer's leftover (Credits hold) goes down.
     6) Solver places an Ask => stake=stake_multiplier * price => verify solver leftover drops.
     7) Solver submits => buyer finalizes => correct =>
        - Buyer hold is charged
        - Solver hold is freed
        - Solver gets new Earnings hold
     8) Validate final leftover in buyer's Credits hold, solver's Freed stake, solver's new Earnings hold.
     9) Check final ledger invariants are still correct.
    """

    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture

        # 1) Admin deposit to buyer
        buyer_id = "buyer_holds_test"
        deposit_buyer = 500.0
        await server.admin_deposit_account(user_id=buyer_id, amount=deposit_buyer)

        # Switch to buyer
        server._user_id_getter = lambda: buyer_id
        balance_buyer_initial = await server.get_balance_details_for_user()
        assert abs(balance_buyer_initial["credits_balance"] - deposit_buyer) == 0, (
            f"Buyer credits_balance expected={deposit_buyer}, got={balance_buyer_initial['credits_balance']}"
        )

        # 2) Admin deposit to solver
        solver_id = "solver_holds_test"
        deposit_solver = 300.0
        await server.admin_deposit_account(user_id=solver_id, amount=deposit_solver)

        # Check invariant so far
        inv_1 = await server.check_invariant()
        assert inv_1["invariant_holds"], f"Invariant broken after initial deposits: {inv_1}"

        # 3) Create plugin, subnet, job
        plugin_id = await server.create_plugin(name="HoldsFlowPlugin", code="print('test holds flow')")
        subnet_id = await server.create_subnet(dispute_period=300, solve_period=600, stake_multiplier=2.0)
        job_id = await server.create_job(
            name="HoldsFlowJob",
            plugin_id=plugin_id,
            subnet_id=subnet_id,
            sot_url="http://dummy_sot",
            iteration=0
        )

        # 4) Create a new Task => status=SelectingSolver
        task_id = await server.create_task(
            job_id=job_id,
            job_iteration=1,
            status=TaskStatus.SelectingSolver.name,
            params=json.dumps({"foo": "bar"})
        )

        # 5) Buyer places a bid => price=100
        bid_price = 100.0
        bid_order_id = await server.create_order(
            task_id=task_id,
            subnet_id=subnet_id,
            order_type=OrderType.Bid,
            price=bid_price,
            hold_id=None
        )

        bal_buyer_after_bid = await server.get_balance_details_for_user()
        leftover_buyer_bid = deposit_buyer - bid_price
        assert abs(bal_buyer_after_bid["credits_balance"] - leftover_buyer_bid) == 0, (
            f"Buyer leftover after placing bid expected={leftover_buyer_bid}, "
            f"got={bal_buyer_after_bid['credits_balance']}"
        )

        # 6) Solver places an ask => user_id=solver
        server._user_id_getter = lambda: solver_id
        ask_price = 100.0
        ask_order_id = await server.create_order(
            task_id=None,
            subnet_id=subnet_id,
            order_type=OrderType.Ask,
            price=ask_price,
            hold_id=None
        )

        # Because stake_multiplier=2.0 => solver locks up 200
        leftover_solver_after_ask = deposit_solver - 2.0 * ask_price
        bal_solver_after_ask = await server.get_balance_details_for_user()
        assert abs(bal_solver_after_ask["credits_balance"] - leftover_solver_after_ask) == 0, (
            f"Solver leftover after ask expected={leftover_solver_after_ask}, "
            f"got={bal_solver_after_ask['credits_balance']}"
        )

        # Check invariant again
        inv_2 = await server.check_invariant()
        assert inv_2["invariant_holds"], f"Invariant broken after placing ask: {inv_2}"

        # 7) Solver => submit => buyer => finalize => correct
        solver_result_data = {"final": "correct_output"}
        await server.submit_task_result(task_id, result=json.dumps(solver_result_data))

        # Switch back to buyer to finalize
        server._user_id_getter = lambda: buyer_id
        await server.finalize_sanity_check(task_id, is_valid=True)

        # 8) Validate final leftover:
        #    Because solver is correct, the solver's stake is freed back into Credits,
        #    so solver reverts to the original deposit => 300. 
        #    Meanwhile, buyer's leftover is 500 - 100 => 400,
        #    and solver also receives new Earnings => ~90 if fee=10%.

        bal_buyer_final = await server.get_balance_details_for_user()
        final_buyer_leftover = deposit_buyer - bid_price
        assert abs(bal_buyer_final["credits_balance"] - final_buyer_leftover) == 0, (
            f"Buyer leftover final expected={final_buyer_leftover}, got={bal_buyer_final['credits_balance']}"
        )

        # Switch to solver to check final leftover
        server._user_id_getter = lambda: solver_id
        bal_solver_final = await server.get_balance_details_for_user()

        # Freed stake => leftover => 300.0 in credits_balance
        expected_credits_final = deposit_solver
        assert abs(bal_solver_final["credits_balance"] - expected_credits_final) == 0, (
            f"Solver final leftover credits expected={expected_credits_final}, "
            f"got={bal_solver_final['credits_balance']}"
        )

        # With a default 10% platform fee => solver_earnings= (1 - 0.1) * 100= 90
        expected_earnings = 0.9 * bid_price
        assert abs(bal_solver_final["earnings_balance"] - expected_earnings) == 0, (
            f"Solver final leftover earnings expected={expected_earnings}, "
            f"got={bal_solver_final['earnings_balance']}"
        )

        # 9) Final invariant
        final_inv = await server.check_invariant()
        assert final_inv["invariant_holds"], f"Final invariant broken: {final_inv}"

        print("test_holds_flow_end_to_end => PASS. All hold logic checks out.")
