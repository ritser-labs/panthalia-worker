import pytest
from datetime import datetime, timedelta
from spl.db.server.app import original_app
from spl.models import Hold, HoldType, Task
from spl.models.enums import OrderType
from spl.common import TaskStatus
from sqlalchemy import select

from spl.db.server.adapter import DBAdapterServer

@pytest.mark.asyncio
async def test_create_cc_hold_and_use_for_bid(db_adapter_server_fixture):
    """
    1) Create a job
    2) Create a hold (credit card) for testuser
    3) Use that hold to place a bid
    4) Confirm used_amount increments
    """
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture

        plugin_id = await server.create_plugin(name="BidTest Plugin", code="print('bid')")
        subnet_id = await server.create_subnet(dispute_period=3600, solve_period=1800, stake_multiplier=2.0)
        job_id = await server.create_job(
            name="Bid Job",
            plugin_id=plugin_id,
            subnet_id=subnet_id,
            sot_url="http://sot_url",
            iteration=0
        )

        async with DBAdapterServer().get_async_session() as session:
            account = await server.get_or_create_account("testuser", session=session)
            cc_hold = Hold(
                account_id=account.id,
                user_id="testuser",
                hold_type=HoldType.CreditCard,
                total_amount=200.0,
                used_amount=0.0,
                expiry=datetime.utcnow() + timedelta(days=5),
                charged=False,
                charged_amount=0.0
            )
            session.add(cc_hold)
            await session.commit()
            await session.refresh(cc_hold)

        # Create a task
        task_id = await server.create_task(
            job_id=job_id,
            job_iteration=1,
            status=TaskStatus.SelectingSolver,
            params="{}"
        )

        # Place a BID order using that hold
        bid_order_id = await server.create_order(
            task_id=task_id,
            subnet_id=subnet_id,
            order_type=OrderType.Bid,
            price=50.0,
            hold_id=cc_hold.id
        )

        async with DBAdapterServer().get_async_session() as session:
            hold_obj = await session.get(Hold, cc_hold.id)
            assert hold_obj.used_amount == 50.0


@pytest.mark.asyncio
async def test_create_ask_order_with_cc_hold_and_match(db_adapter_server_fixture):
    """
    Full scenario:
     - testuser => buyer with deposit
     - solveruser => solver with CC hold
     - matching => finalize => solver's deposit not charged => gets new earnings hold, etc.
    """
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture

        plugin_id = await server.create_plugin(name="FullScenario Plugin", code="print('full')")
        subnet_id = await server.create_subnet(dispute_period=3600, solve_period=1800, stake_multiplier=1.5)
        job_id = await server.create_job(
            name="Full Job",
            plugin_id=plugin_id,
            subnet_id=subnet_id,
            sot_url="http://example.com",
            iteration=0
        )

        # Buyer => deposit 500
        await server.admin_deposit_account(user_id="testuser", amount=500.0)

        # solver => "solveruser"
        solver_server = DBAdapterServer(user_id_getter=lambda: "solveruser")

        async with DBAdapterServer().get_async_session() as session:
            solver_account = await solver_server.get_or_create_account("solveruser", session=session)
            solver_hold = Hold(
                account_id=solver_account.id,
                user_id="solveruser",
                hold_type=HoldType.CreditCard,
                total_amount=300.0,
                used_amount=0.0,
                expiry=datetime.utcnow() + timedelta(days=5),
                charged=False,
                charged_amount=0.0
            )
            session.add(solver_hold)
            await session.commit()

        # create a task => testuser's job
        task_id = await server.create_task(
            job_id=job_id,
            job_iteration=1,
            status=TaskStatus.SelectingSolver,
            params="{}"
        )

        # Buyer => place BID=100 with new CC hold
        async with DBAdapterServer().get_async_session() as session:
            buyer_account = await server.get_or_create_account("testuser", session=session)
            buyer_cc_hold = Hold(
                account_id=buyer_account.id,
                user_id="testuser",
                hold_type=HoldType.CreditCard,
                total_amount=100.0,
                used_amount=0.0,
                expiry=datetime.utcnow() + timedelta(days=2),
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

        # solver => ask => same price=100 => using solver_hold
        ask_order_id = await solver_server.create_order(
            task_id=None,
            subnet_id=subnet_id,
            order_type=OrderType.Ask,
            price=100.0,
            hold_id=solver_hold.id
        )

        # solver => scPending => finalize correct => new earnings hold
        await solver_server.submit_task_result(task_id, result='{"output": "success"}')
        await server.finalize_sanity_check(task_id, True)

        # confirm
        async with DBAdapterServer().get_async_session() as session:
            solver_acc = await solver_server.get_or_create_account("solveruser", session=session)
            updated_solver_hold = await session.execute(
                select(Hold).where(Hold.account_id == solver_acc.id, Hold.total_amount == 300.0)
            )
            updated_solver_hold = updated_solver_hold.scalars().first()
            assert updated_solver_hold is not None
            assert not updated_solver_hold.charged
            assert updated_solver_hold.used_amount == 0.0

            earn_hold = await session.execute(
                select(Hold).where(Hold.user_id=="solveruser", Hold.total_amount==90.0)
            )
            earn_hold = earn_hold.scalars().first()
            assert earn_hold is not None
            assert earn_hold.charged is False
            assert earn_hold.used_amount == 0.0
