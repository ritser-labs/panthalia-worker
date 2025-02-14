import pytest
import json
from spl.db.server.app import original_app
from spl.models import Task, Order, TaskStatus, OrderType
from sqlalchemy import select
from sqlalchemy.orm import selectinload

@pytest.mark.asyncio
async def test_ask_order_reserves_full_stake(db_adapter_server_fixture):
    """
    1) Buyer user => deposit 100 => places a Bid=50 => leftover=50
    2) Solver user => deposit 500 => places an Ask=50 => stake=100 => leftover=400
    3) Manually match => task.bid & task.ask => set task.status=SolverSelected
    4) Solver => submit => finalize => is_valid=False => leftover_amount=bid.price => freed to buyer
       => solver stake is fully charged => no leftover => solver's final leftover=400
       => buyer reverts to leftover=100
    """
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture

        # 1) Create plugin & subnet => stake_multiplier=2.0
        #    We'll do it under the buyer context so the buyer owns the job & can create tasks:
        buyer_user = "ask_test_buyer"
        server._user_id_getter = lambda: buyer_user

        subnet_id = await server.create_subnet(dispute_period=300, solve_period=300, stake_multiplier=2.0)
        plugin_id = await server.create_plugin(name="AskTest Plugin", code="print('ask test')", subnet_id=subnet_id)
        job_id = await server.create_job(
            name="AskTestJob",
            plugin_id=plugin_id,
            sot_url="http://fake-sot",
            iteration=0,
            limit_price=1
        )

        # Buyer deposit => 100 => leftover=100
        await server.admin_deposit_account(buyer_user, amount=100.0)

        # Create a new Task => status=SelectingSolver
        task_id = await server.create_task(
            job_id=job_id,
            job_iteration=1,
            status=TaskStatus.SelectingSolver.name,
            params=json.dumps({"some": "params"})
        )

        # Buyer places a Bid => price=50 => leftover => 50
        bid_price = 50.0
        bid_order_id = await server.create_order(
            task_id=task_id,
            subnet_id=subnet_id,
            order_type=OrderType.Bid,
            price=bid_price,
        )
        bal_buyer_1 = await server.get_balance_details_for_user()
        assert abs(bal_buyer_1["credits_balance"] - 50.0) < 1e-9, (
            f"Expected buyer leftover=50, got {bal_buyer_1['credits_balance']}"
        )

        # 2) Switch user => solver => deposit=500 => leftover=500
        solver_user = "ask_solver_user"
        server._user_id_getter = lambda: solver_user
        await server.admin_deposit_account(solver_user, amount=500.0)

        # Place an Ask => price=50 => stake=2*50=100 => leftover=400
        ask_price = 50.0
        ask_order_id = await server.create_order(
            task_id=None,
            subnet_id=subnet_id,
            order_type=OrderType.Ask,
            price=ask_price
        )
        bal_solver_1 = await server.get_balance_details_for_user()
        assert abs(bal_solver_1["credits_balance"] - 400.0) < 1e-9, (
            f"Expected solver leftover=400, got {bal_solver_1['credits_balance']}"
        )

        # 3) "Manual match" => set task.bid & task.ask => status=SolverSelected
        async with server.get_async_session() as session:
            t_obj = await session.get(Task, task_id)
            b_order = await session.get(Order, bid_order_id)
            a_order = await session.get(Order, ask_order_id)

            t_obj.bid = b_order
            t_obj.ask = a_order
            t_obj.status = TaskStatus.SolverSelected

            await session.commit()

        # 4) Solver => submit => finalize => incorrect => leftover_amount=bid.price => freed to buyer, solver stake is charged
        await server.submit_partial_result(task_id, json.dumps({"outcome": "incorrect"}), final=True)
        await server.finalize_sanity_check(task_id, is_valid=False)

        # Final check:
        #   Buyer leftover => originally 50 after the bid, +50 refund => 100
        #   Solver leftover => 400 => stake is lost => remains 400
        server._user_id_getter = lambda: buyer_user
        final_buyer_bal = await server.get_balance_details_for_user()
        assert abs(final_buyer_bal["credits_balance"] - 100.0) < 1e-9, (
            f"Buyer leftover expected=100, got {final_buyer_bal['credits_balance']}"
        )

        server._user_id_getter = lambda: solver_user
        final_solver_bal = await server.get_balance_details_for_user()
        assert abs(final_solver_bal["credits_balance"] - 400.0) < 1e-9, (
            f"Solver leftover expected=400, got {final_solver_bal['credits_balance']}"
        )

        # Invariant
        inv = await server.check_invariant()
        assert inv["invariant_holds"], f"Invariant broken => {inv}"

        print("[test_ask_order_reserves_full_stake] => PASS")
