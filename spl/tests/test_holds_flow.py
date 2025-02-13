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
     5) Buyer places a Bid => verify buyer leftover goes down.
     6) Solver places an Ask => stake=stake_multiplier * price => leftover drops.
     7) Solver submits => buyer finalizes => correct => leftover checks
     8) Final leftover checks
     9) Final invariants
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
        subnet_id = await server.create_subnet(dispute_period=300, solve_period=600, stake_multiplier=2.0)
        plugin_id = await server.create_plugin(name="HoldsFlowPlugin", code="print('test holds flow')", subnet_id=subnet_id)
        job_id = await server.create_job(
            name="HoldsFlowJob",
            plugin_id=plugin_id,
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

        # 5) Buyer => bid => price=100
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

        # 6) Solver => ask => price=100 => stake=200
        server._user_id_getter = lambda: solver_id
        ask_price = 100.0
        ask_order_id = await server.create_order(
            task_id=None,
            subnet_id=subnet_id,
            order_type=OrderType.Ask,
            price=ask_price,
            hold_id=None
        )

        leftover_solver_after_ask = deposit_solver - 2.0 * ask_price
        bal_solver_after_ask = await server.get_balance_details_for_user()
        assert abs(bal_solver_after_ask["credits_balance"] - leftover_solver_after_ask) == 0, (
            f"Solver leftover after ask expected={leftover_solver_after_ask}, "
            f"got={bal_solver_after_ask['credits_balance']}"
        )

        # match & commit
        async with server.get_async_session() as sess:
            await server.match_bid_ask_orders(sess, subnet_id)
            await sess.commit()

        # Check invariant again
        inv_2 = await server.check_invariant()
        assert inv_2["invariant_holds"], f"Invariant broken after placing ask: {inv_2}"

        # 7) Solver => submit => buyer => finalize => correct
        solver_result_data = {"final": "correct_output"}
        await server.submit_partial_result(task_id, json.dumps(solver_result_data), final=True)

        # Switch buyer => finalize
        server._user_id_getter = lambda: buyer_id
        await server.finalize_sanity_check(task_id, is_valid=True)

        # 8) Final leftover checks
        bal_buyer_final = await server.get_balance_details_for_user()
        final_buyer_leftover = deposit_buyer - bid_price  # e.g. 500 - 100=400
        assert abs(bal_buyer_final["credits_balance"] - final_buyer_leftover) == 0, (
            f"Buyer leftover final expected={final_buyer_leftover}, "
            f"got={bal_buyer_final['credits_balance']}"
        )

        # Switch solver => check final leftover
        server._user_id_getter = lambda: solver_id
        bal_solver_final = await server.get_balance_details_for_user()

        # Freed stake => leftover => original deposit=300
        assert abs(bal_solver_final["credits_balance"] - deposit_solver) == 0, (
            f"Solver final leftover credits expected={deposit_solver}, "
            f"got={bal_solver_final['credits_balance']}"
        )

        # With fee=10% => solver gets new Earnings=90
        # (If your platform fee is stored differently, adjust as needed.)
        expected_earnings = 0.9 * bid_price
        assert abs(bal_solver_final["earnings_balance"] - expected_earnings) == 0, (
            f"Solver final leftover earnings expected={expected_earnings}, "
            f"got={bal_solver_final['earnings_balance']}"
        )

        # 9) Final invariants
        inv_3 = await server.check_invariant()
        assert inv_3["invariant_holds"], f"Final invariant broken: {inv_3}"

        print("test_holds_flow_end_to_end => PASS. All hold logic checks out.")
