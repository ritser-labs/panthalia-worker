# file: spl/tests/test_deposit_task_balance_consistency.py

import pytest
import json
from spl.db.server.app import original_app
from spl.models.enums import TaskStatus, OrderType
from sqlalchemy import select

@pytest.mark.asyncio
async def test_deposit_task_balance_consistency(db_adapter_server_fixture):
    """
    Checks deposit tasks & balances via repeated invariant checks:
     1) Buyer deposit
     2) Solver deposit
     3) Create plugin, subnet, job
     4) Create a new Task
     5) Buyer places a Bid => locks 50
     6) Solver places an Ask => locks 100 (since stake_multiplier=2 => 2*50)
     7) Solver => submit => buyer => finalize => correct => check & cleanup
     8) Final check that the overall ledger is balanced.
    """
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture

        buyer_user_id = "test_buyer"
        solver_user_id = "test_solver"

        deposit_amount_buyer = 100.0
        deposit_amount_solver = 200.0

        #
        # 1) Admin deposit for buyer => 100
        #
        await server.admin_deposit_account(user_id=buyer_user_id, amount=deposit_amount_buyer)

        # Switch user ID => 'test_buyer'
        server._user_id_getter = lambda: buyer_user_id

        # Confirm buyer's new credits_balance
        balance_info_buyer_1 = await server.get_balance_details_for_user()
        assert balance_info_buyer_1["credits_balance"] == deposit_amount_buyer, (
            f"Expected buyer's credits_balance={deposit_amount_buyer}, "
            f"got {balance_info_buyer_1['credits_balance']}"
        )

        # Invariant check after buyer deposit
        invariant_1 = await server.check_invariant()
        assert invariant_1["invariant_holds"], (
            f"Invariant broken after buyer deposit: {invariant_1}"
        )

        #
        # 2) Admin deposit for solver => 200
        #
        await server.admin_deposit_account(user_id=solver_user_id, amount=deposit_amount_solver)

        # Switch user ID => 'test_solver'
        server._user_id_getter = lambda: solver_user_id

        # Confirm solver's new credits_balance
        balance_info_solver_1 = await server.get_balance_details_for_user()
        assert balance_info_solver_1["credits_balance"] == deposit_amount_solver, (
            f"Expected solver's credits_balance={deposit_amount_solver}, "
            f"got {balance_info_solver_1['credits_balance']}"
        )

        # Invariant check after solver deposit
        invariant_2 = await server.check_invariant()
        assert invariant_2["invariant_holds"], (
            f"Invariant broken after solver deposit: {invariant_2}"
        )

        #
        # 3) Switch to buyer => create plugin, subnet, job
        #
        server._user_id_getter = lambda: buyer_user_id

        plugin_id = await server.create_plugin(
            name="TestPluginBalanceCheck",
            code="print('test balance check')"
        )
        subnet_id = await server.create_subnet(
            dispute_period=3600,
            solve_period=1800,
            stake_multiplier=2.0
        )
        job_id = await server.create_job(
            name="BalanceCheckJob",
            plugin_id=plugin_id,
            subnet_id=subnet_id,
            sot_url="http://example.com",
            iteration=0
        )

        # Invariant check after creating job
        invariant_3 = await server.check_invariant()
        assert invariant_3["invariant_holds"], (
            f"Invariant broken after creating job: {invariant_3}"
        )

        #
        # 4) Create a new Task => status=SelectingSolver
        #
        task_id = await server.create_task(
            job_id=job_id,
            job_iteration=1,
            status=TaskStatus.SelectingSolver.name,
            params=json.dumps({"test": "params"})
        )

        # Invariant check after creating the task
        invariant_4 = await server.check_invariant()
        assert invariant_4["invariant_holds"], (
            f"Invariant broken after creating task: {invariant_4}"
        )

        #
        # 5) Buyer places a Bid => price=50
        #
        bid_price = 50.0
        bid_order_id = await server.create_order(
            task_id=task_id,
            subnet_id=subnet_id,
            order_type=OrderType.Bid,
            price=bid_price,
            hold_id=None
        )

        # Confirm buyer leftover => 100 - 50 => 50
        balance_info_buyer_after_bid = await server.get_balance_details_for_user()
        leftover_after_bid = balance_info_buyer_after_bid["credits_balance"]
        assert abs(leftover_after_bid - (deposit_amount_buyer - bid_price)) < 1e-9, (
            f"Buyer leftover should now be {deposit_amount_buyer - bid_price}, got {leftover_after_bid}"
        )

        # Invariant check after buyer places bid
        invariant_5 = await server.check_invariant()
        assert invariant_5["invariant_holds"], (
            f"Invariant broken after buyer places bid: {invariant_5}"
        )

        #
        # 6) Solver places an Ask => price=50 => stake=2*50=100
        #
        server._user_id_getter = lambda: solver_user_id
        ask_price = 50.0
        ask_order_id = await server.create_order(
            task_id=None,
            subnet_id=subnet_id,
            order_type=OrderType.Ask,
            price=ask_price,
            hold_id=None
        )

        # Confirm solver leftover => 200 - (2*50) => 100
        balance_info_solver_after_ask = await server.get_balance_details_for_user()
        leftover_solver_after_ask = balance_info_solver_after_ask["credits_balance"]
        assert abs(leftover_solver_after_ask - (deposit_amount_solver - 2 * ask_price)) < 1e-9, (
            f"Solver leftover must be {deposit_amount_solver - 2 * ask_price}, got {leftover_solver_after_ask}"
        )

        # Invariant check after solver places ask
        invariant_6 = await server.check_invariant()
        assert invariant_6["invariant_holds"], (
            f"Invariant broken after solver places ask: {invariant_6}"
        )

        #
        # 7) Solver => submit => buyer => finalize => correct => cleanup
        #
        solver_result_data = {"test_result": "ok", "loss": 0.123}
        await server.submit_task_result(task_id, result=json.dumps(solver_result_data))

        # Switch to buyer to finalize
        server._user_id_getter = lambda: buyer_user_id
        await server.finalize_sanity_check(task_id, is_valid=True)
        await server.check_and_cleanup_holds()

        # Invariant check after finalizing sanity check
        invariant_7 = await server.check_invariant()
        assert invariant_7["invariant_holds"], (
            f"Invariant broken after finalize sanity check: {invariant_7}"
        )

        #
        # 8) Final check: rely purely on invariant
        #
        final_invariant = await server.check_invariant()
        assert final_invariant["invariant_holds"], (
            f"Final invariant broken: {final_invariant}"
        )

        # Done! We no longer do partial arithmetic because check_invariant handles all sums.
