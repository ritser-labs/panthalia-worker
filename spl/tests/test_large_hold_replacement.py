import pytest
import json
from datetime import datetime, timedelta
from spl.db.server.app import original_app
from spl.models.enums import OrderType, TaskStatus

@pytest.mark.asyncio
async def test_multiple_tasks_partial_usage(db_adapter_server_fixture):
    """
    Scenario:
      1) The buyer has a *huge credits hold* (e.g. 999999955).
      2) We create multiple tasks (say 3). For each task:
         - The buyer places a small bid (e.g. 10).
         - The solver places an ask (some small number).
         - We finalize with correct resolution => the old hold is charged for 10,
           and a new leftover hold is created for leftover=(previous_leftover - 10).
      3) We confirm that after each finalization, there's exactly one uncharged hold
         with the leftover total. Summation across tasks => 30 used in total, etc.
      4) Also, we check the money-in/money-out "invariant" after each key step.
    """

    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture

        buyer_user_id = "multi_task_user"
        solver_user_id = "multi_task_solver"

        # (A) Admin deposit => e.g. 999999955 for the buyer
        big_deposit_amount = 999999955.0
        await server.admin_deposit_account(user_id=buyer_user_id, amount=big_deposit_amount)

        # Invariant check after buyer deposit
        inv_1 = await server.check_invariant()
        assert inv_1["invariant_holds"], f"Invariant broken after buyer deposit: {inv_1}"

        # Switch user => buyer, confirm new credits_balance
        server._user_id_getter = lambda: buyer_user_id
        balance_info_1 = await server.get_balance_details_for_user()
        assert abs(balance_info_1["credits_balance"] - big_deposit_amount) < 1e-6, (
            f"Expected buyer's credits_balance={big_deposit_amount}, got {balance_info_1['credits_balance']}"
        )

        # Also deposit some solver credit if needed (100.0)
        await server.admin_deposit_account(user_id=solver_user_id, amount=100.0)

        # Invariant check after solver deposit
        inv_2 = await server.check_invariant()
        assert inv_2["invariant_holds"], f"Invariant broken after solver deposit: {inv_2}"

        # (B) Create plugin, subnet, job
        plugin_id = await server.create_plugin("MultiPartialTest", "print('multiple tasks test')")
        subnet_id = await server.create_subnet(dispute_period=3600, solve_period=1800, stake_multiplier=1.0)
        job_id = await server.create_job(
            name="MultiPartialUsageJob",
            plugin_id=plugin_id,
            subnet_id=subnet_id,
            sot_url="http://example.com",
            iteration=0
        )

        # Invariant check after job creation
        inv_3 = await server.check_invariant()
        assert inv_3["invariant_holds"], f"Invariant broken after job creation: {inv_3}"

        # We'll do 3 tasks, each using 10 from the buyer
        number_of_tasks = 3
        usage_per_task = 10.0

        total_used = 0.0  # track how many total credits have been used so far

        for i in range(number_of_tasks):
            # (C) Create a new Task
            task_id = await server.create_task(
                job_id=job_id,
                job_iteration=i+1,
                status=TaskStatus.SelectingSolver.name,
                params="{}"
            )

            # Invariant check after creating each task
            inv_task = await server.check_invariant()
            assert inv_task["invariant_holds"], f"Invariant broken after creating Task {i+1}: {inv_task}"

            # Buyer places a small bid => usage_per_task
            # Switch to buyer
            server._user_id_getter = lambda: buyer_user_id
            bid_order_id = await server.create_order(
                task_id=task_id,
                subnet_id=subnet_id,
                order_type=OrderType.Bid,
                price=usage_per_task,
                hold_id=None
            )

            # Invariant check after buyer places bid
            inv_bid = await server.check_invariant()
            assert inv_bid["invariant_holds"], f"Invariant broken after buyer places bid for Task {i+1}: {inv_bid}"

            # Switch to solver => place an Ask
            server._user_id_getter = lambda: solver_user_id
            ask_price = 5.0  # any small number
            await server.create_order(
                task_id=None,
                subnet_id=subnet_id,
                order_type=OrderType.Ask,
                price=ask_price,
                hold_id=None
            )

            # Invariant check after solver places ask
            inv_ask = await server.check_invariant()
            assert inv_ask["invariant_holds"], f"Invariant broken after solver places ask for Task {i+1}: {inv_ask}"

            # Solver => submit => buyer => finalize => correct
            solver_result_data = {"output": "multi-test", "iteration": i}
            await server.submit_task_result(task_id, json.dumps(solver_result_data))

            server._user_id_getter = lambda: buyer_user_id
            await server.finalize_sanity_check(task_id, is_valid=True)

            # Invariant check after finalizing
            inv_finalize = await server.check_invariant()
            assert inv_finalize["invariant_holds"], f"Invariant broken after finalizing Task {i+1}: {inv_finalize}"

            # The old hold is fully charged => usage_per_task
            total_used += usage_per_task

            # (D) Check leftover hold
            final_balance_info = await server.get_balance_details_for_user()
            final_holds = final_balance_info["detailed_holds"]

            leftover_amount_expected = big_deposit_amount - total_used

            # Make sure there's exactly 1 uncharged credits hold with total= leftover_amount_expected
            uncharged_credits_holds = [
                h for h in final_holds
                if (not h["charged"]) and (h["hold_type"] == "credits")
            ]
            matching_holds = [
                h for h in uncharged_credits_holds
                if abs(h["total_amount"] - leftover_amount_expected) < 1e-6
            ]
            assert len(matching_holds) == 1, (
                f"After finalizing task {i+1}, expected exactly 1 leftover hold total={leftover_amount_expected}, "
                f"found {len(matching_holds)} among: {uncharged_credits_holds}"
            )

            # final credits_balance => leftover_amount_expected
            final_credits_balance = final_balance_info["credits_balance"]
            assert abs(final_credits_balance - leftover_amount_expected) < 1e-6, (
                f"After finalizing task {i+1}, credits_balance={final_credits_balance} != leftover={leftover_amount_expected}"
            )

        print(f"[test_multiple_tasks_partial_usage] after {number_of_tasks} tasks, leftover={leftover_amount_expected}")

        # Final check after the loop completes
        final_inv = await server.check_invariant()
        assert final_inv["invariant_holds"], f"Final invariant broken after all tasks: {final_inv}"
