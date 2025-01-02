import pytest
import json
from datetime import datetime, timedelta
from sqlalchemy import select
from spl.db.server.app import original_app
from spl.models import AsyncSessionLocal, Hold, HoldType, TaskStatus
from spl.models.enums import OrderType
from spl.db.server.adapter import DBAdapterServer
import logging


@pytest.mark.asyncio
async def test_cc_hold_incorrect_leftover_to_credits(db_adapter_server_fixture):
    """
    Ensures that when a solver creates an ASK order using a Credit Card hold,
    and the task is resolved incorrectly, the leftover from that hold is converted
    into deposit-based credits with a 1-year expiry.
    """
    async with original_app.test_request_context('/'):
        server: DBAdapterServer = db_adapter_server_fixture

        # 1) Create plugin, subnet, job
        plugin_id = await server.create_plugin(name="CreditCardTestPlugin", code="print('cc-test')")
        subnet_id = await server.create_subnet(dispute_period=3600, solve_period=1800, stake_multiplier=2.0)
        job_id = await server.create_job(
            name="CC_Ask_Job",
            plugin_id=plugin_id,
            subnet_id=subnet_id,
            sot_url="http://example.com/sot",
            iteration=0
        )

        # 2) Buyer => deposit 500
        await server.admin_deposit_account(user_id="testuser", amount=500.0)

        # 3) Solver => "solveruser" => has a CC hold
        solver_server = DBAdapterServer(user_id_getter=lambda: "solveruser")
        async with AsyncSessionLocal() as session:
            solver_account = await solver_server.get_or_create_account("solveruser", session=session)
            solver_cc_hold = Hold(
                account_id=solver_account.id,
                user_id="solveruser",
                hold_type=HoldType.CreditCard,
                total_amount=300.0,
                used_amount=0.0,
                expiry=datetime.utcnow() + timedelta(days=5),
                charged=False,
                charged_amount=0.0
            )
            session.add(solver_cc_hold)
            await session.commit()
            await session.refresh(solver_cc_hold)

        # 4) Create a task => job iteration=1 => status=SelectingSolver
        task_id = await server.create_task(
            job_id=job_id,
            job_iteration=1,
            status=TaskStatus.SelectingSolver,
            params="{}"
        )

        # 5) Buyer => place a BID => price=100
        async with AsyncSessionLocal() as session:
            buyer_account = await server.get_or_create_account("testuser", session=session)
            buyer_cc_hold = Hold(
                account_id=buyer_account.id,
                user_id="testuser",
                hold_type=HoldType.CreditCard,
                total_amount=100.0,
                used_amount=0.0,
                expiry=datetime.utcnow() + timedelta(days=5),
                charged=False,
                charged_amount=0.0
            )
            session.add(buyer_cc_hold)
            await session.commit()
            await session.refresh(buyer_cc_hold)

        bid_order_id = await server.create_order(
            task_id=task_id,
            subnet_id=subnet_id,
            order_type=OrderType.Bid,
            price=100.0,
            hold_id=buyer_cc_hold.id
        )

        # 6) Solver => place an ASK => price=100 => stake=2*100=200 => leftover=100 in that 300 hold
        ask_order_id = await solver_server.create_order(
            task_id=None,
            subnet_id=subnet_id,
            order_type=OrderType.Ask,
            price=100.0,
            hold_id=solver_cc_hold.id
        )

        # 7) Solver => submit => scPending => finalize => is_valid=False => incorrect
        await solver_server.submit_task_result(task_id, result=json.dumps({"output": "fail"}))
        await server.finalize_sanity_check(task_id, is_valid=False)

        # 8) Re-fetch the solver's leftover hold => turned into deposit-based credits
        async with AsyncSessionLocal() as session:
            updated_solver_cc_hold = await session.get(Hold, solver_cc_hold.id)
            assert updated_solver_cc_hold.charged is True
            assert updated_solver_cc_hold.used_amount == updated_solver_cc_hold.total_amount

            leftover_stmt = (
                select(Hold).where(
                    Hold.parent_hold_id == solver_cc_hold.id,
                    Hold.user_id == "solveruser"
                )
            )
            leftover_res = await session.execute(leftover_stmt)
            leftover_hold = leftover_res.scalar_one_or_none()

            assert leftover_hold is not None
            assert leftover_hold.total_amount == 100.0
            assert leftover_hold.used_amount == 0.0
            assert leftover_hold.hold_type == HoldType.Credits

        logging.info("test_cc_hold_incorrect_leftover_to_credits passed successfully!")
