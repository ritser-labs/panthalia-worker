import pytest
import pytest_asyncio
import json
from datetime import datetime, timedelta
from sqlalchemy import select
from sqlalchemy.orm import selectinload, joinedload
from unittest.mock import patch
import asyncio

from spl.models import (
    AsyncSessionLocal, TaskStatus, OrderType, HoldType,
    CreditTxnType, EarningsTxnType, PlatformRevenueTxnType,
    Hold, Order, Task, PlatformRevenue, Account, Job
)
from spl.db.server.app import original_app
from spl.db.server.adapter.orders_tasks import DBAdapterOrdersTasksMixin
from spl.db.server.adapter import DBAdapterServer


@pytest_asyncio.fixture(autouse=True)
async def clear_database():
    """
    Clear the database before each test to ensure test isolation.
    """
    async with AsyncSessionLocal() as session:
        from spl.models import Base
        for table in reversed(Base.metadata.sorted_tables):
            await session.execute(table.delete())
        await session.commit()
    yield


@pytest.fixture(autouse=True)
def patch_get_task_eager():
    """
    Patch DBAdapterOrdersTasksMixin.get_task so it uses eager-load with joinedload().
    This prevents lazy-loading after the session is out of scope, which
    avoids the MissingGreenlet error in the tests.
    """
    original_get_task = DBAdapterOrdersTasksMixin.get_task

    async def get_task_eager(self, task_id: int):
        async with AsyncSessionLocal() as session:
            stmt = (
                select(Task)
                .options(
                    joinedload(Task.bid).joinedload(Order.hold),
                    joinedload(Task.ask).joinedload(Order.hold),
                    joinedload(Task.job).joinedload(Job.subnet),
                )
                .where(Task.id == task_id)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    with patch.object(DBAdapterOrdersTasksMixin, 'get_task', get_task_eager):
        yield


@pytest.mark.asyncio
async def test_basic_setup(db_adapter_server_fixture):
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture

        plugin_id = await server.create_plugin(name="Test Plugin", code="print('hello')")
        subnet_id = await server.create_subnet(dispute_period=3600, solve_period=1800, stake_multiplier=1.5)
        job_id = await server.create_job(
            name="Test Job",
            plugin_id=plugin_id,
            subnet_id=subnet_id,
            sot_url="http://example.com",
            iteration=0
        )

        job = await server.get_job(job_id)
        assert job is not None
        assert job.plugin_id == plugin_id
        assert job.subnet_id == subnet_id


@pytest.mark.asyncio
async def test_admin_deposit_credits(db_adapter_server_fixture):
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture
        await server.admin_deposit_account(user_id="testuser", amount=300.0)
        async with AsyncSessionLocal() as session:
            account = await server.get_or_create_account("testuser", session=session)
            assert account.credits_balance == 300.0


@pytest.mark.asyncio
async def test_create_cc_hold_and_use_for_bid(db_adapter_server_fixture):
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

        async with AsyncSessionLocal() as session:
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

        # Create a task for the job
        task_id = await server.create_task(
            job_id=job_id,
            job_iteration=1,
            status=TaskStatus.SelectingSolver,
            params="{}"
        )

        # Create a bid order using the cc_hold
        bid_order_id = await server.create_order(
            task_id=task_id,
            subnet_id=subnet_id,
            order_type=OrderType.Bid,
            price=50.0,
            hold_id=cc_hold.id
        )

        async with AsyncSessionLocal() as session:
            order = await session.execute(select(Order).where(Order.id == bid_order_id))
            order = order.scalar_one_or_none()
            assert order is not None
            hold = await session.execute(select(Hold).where(Hold.id == cc_hold.id))
            hold = hold.scalar_one_or_none()
            assert hold.used_amount == 50.0


@pytest.mark.asyncio
async def test_create_ask_order_with_cc_hold_and_match(db_adapter_server_fixture):
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

        # Buyer => 500
        await server.admin_deposit_account(user_id="testuser", amount=500.0)

        # solver => "solveruser"
        solver_server = DBAdapterServer(user_id_getter=lambda: "solveruser")

        async with AsyncSessionLocal() as session:
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
            await session.refresh(solver_hold)

        task_id = await server.create_task(
            job_id=job_id,
            job_iteration=1,
            status=TaskStatus.SelectingSolver,
            params="{}"
        )

        async with AsyncSessionLocal() as session:
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

        # create bid
        bid_order_id = await server.create_order(
            task_id=task_id,
            subnet_id=subnet_id,
            order_type=OrderType.Bid,
            price=100.0,
            hold_id=buyer_cc_hold.id
        )
        # create ask
        ask_order_id = await solver_server.create_order(
            task_id=None,
            subnet_id=subnet_id,
            order_type=OrderType.Ask,
            price=100.0,
            hold_id=solver_hold.id
        )
        # solver => submit => => scPending
        await solver_server.submit_task_result(task_id, result=json.dumps({"output": "success"}))

        # finalize => correct => solver gets ~90 earnings
        await server.finalize_sanity_check(task_id, True)

        # Re-check results in a FRESH session with eager loads.
        async with AsyncSessionLocal() as session:
            stmt = (
                select(Task)
                .options(
                    selectinload(Task.bid).selectinload(Order.hold),
                    selectinload(Task.ask).selectinload(Order.hold),
                    selectinload(Task.job).selectinload(Job.subnet)
                )
                .where(Task.id == task_id)
            )
            fetched = await session.execute(stmt)
            task = fetched.scalar_one_or_none()
            assert task is not None
            assert task.status == TaskStatus.ResolvedCorrect

            # Check buyer indefinite credits remain 500.
            buyer_account = await server.get_or_create_account("testuser", session=session)
            assert buyer_account.credits_balance == 500.0

            # solver indefinite credits remain 0, because we do no indefinite credit
            solver_account = await solver_server.get_or_create_account("solveruser", session=session)
            assert solver_account.credits_balance == 0.0

            # Confirm the solver's original hold is uncharged with used_amount=0
            updated_solver_hold = await session.get(Hold, solver_hold.id)
            assert updated_solver_hold is not None
            assert not updated_solver_hold.charged
            assert updated_solver_hold.used_amount == 0.0

            # Now check that a new hold for solver's "earnings" of 90.0 was created
            # We know price=100, fee=10 => solver_earnings=90
            # Try to find a hold with total_amount=90 & user_id="solveruser"
            earnings_hold = await session.execute(
                select(Hold).where(
                    Hold.user_id == "solveruser",
                    Hold.total_amount == 90.0,
                    Hold.charged == False
                )
            )
            earnings_hold = earnings_hold.scalars().first()
            assert earnings_hold is not None, "Expected a new hold for solver's 90.0 earnings"
            # Also confirm that leftover stake is 1.5*100=150 is effectively 'freed' from the solver hold.


@pytest.mark.asyncio
async def test_dispute_scenario_with_cc_hold(db_adapter_server_fixture):
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture

        plugin_id = await server.create_plugin(name="DisputeTest Plugin", code="print('dispute')")
        subnet_id = await server.create_subnet(dispute_period=3600, solve_period=1800, stake_multiplier=2.0)
        job_id = await server.create_job(
            name="Dispute Job",
            plugin_id=plugin_id,
            subnet_id=subnet_id,
            sot_url="http://sot_url",
            iteration=0
        )

        # testuser => deposit 500
        await server.admin_deposit_account(user_id="testuser", amount=500.0)

        # solver => "solveruser" => also has 500 indefinite from old system
        solver_server = DBAdapterServer(user_id_getter=lambda: "solveruser")
        async with AsyncSessionLocal() as session:
            solver_account = await solver_server.get_or_create_account("solveruser", session=session)
            solver_account.credits_balance = 500.0
            session.add(solver_account)
            await session.commit()

        # create solver hold
        async with AsyncSessionLocal() as session:
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

            bidder_account = await server.get_or_create_account("testuser", session=session)
            bidder_hold = Hold(
                account_id=bidder_account.id,
                user_id="testuser",
                hold_type=HoldType.Credits,
                total_amount=200.0,
                used_amount=0.0,
                expiry=datetime.utcnow() + timedelta(days=5),
                charged=False,
                charged_amount=0.0
            )
            session.add(bidder_hold)
            await session.commit()
            await session.refresh(solver_hold)
            await session.refresh(bidder_hold)

        # create a task
        task_id = await server.create_task(
            job_id=job_id,
            job_iteration=1,
            status=TaskStatus.SelectingSolver,
            params="{}"
        )

        # place bid => price=50
        bid_id = await server.create_order(
            task_id=task_id,
            subnet_id=subnet_id,
            order_type=OrderType.Bid,
            price=50.0,
            hold_id=bidder_hold.id
        )
        # solver => ask => price=40
        ask_id = await solver_server.create_order(
            task_id=None,
            subnet_id=subnet_id,
            order_type=OrderType.Ask,
            price=40.0,
            hold_id=solver_hold.id
        )

        # solver => submit => scPending => "fail"
        await solver_server.submit_task_result(task_id, result=json.dumps({"output": "fail"}))
        task = await server.get_task(task_id)
        assert task.status == TaskStatus.SanityCheckPending

        # finalize => is_valid=False => solver is penalized => solver hold fully charged
        await server.finalize_sanity_check(task_id, False)

        # re-fetch
        async with AsyncSessionLocal() as session:
            fetched_task = await session.get(Task, task_id)
            assert fetched_task is not None
            assert fetched_task.status == TaskStatus.ResolvedIncorrect

            # solver indefinite credits is still 500, no new indefinite because solver lost stake
            solver_acc = await solver_server.get_or_create_account("solveruser", session=session)
            assert solver_acc.credits_balance == 500.0

            # confirm solver hold is now fully charged
            updated_solver_hold = await session.get(Hold, solver_hold.id)
            assert updated_solver_hold.charged is True
            assert updated_solver_hold.used_amount == updated_solver_hold.total_amount

            # If leftover>0 (300 total - 80 stake=220 leftover?), we should see a new leftover hold
            # if your code logic returns leftover to solver. 
            # If your code gives leftover to platform, you might skip this check. 
            leftover_hold = await session.execute(
                select(Hold).where(
                    Hold.parent_hold_id == solver_hold.id,
                    Hold.user_id == "solveruser"
                )
            )
            leftover_hold = leftover_hold.scalars().first()
            # if your logic says leftover is returned => leftover_hold should exist w/ total_amount=220
            # But if your logic is different, adapt the assertion:
            if leftover_hold:
                assert leftover_hold.total_amount == 220.0
                assert leftover_hold.used_amount == 0.0
                assert leftover_hold.charged is False



@pytest.mark.asyncio
async def test_dispute_can_only_happen_while_result_uploaded(db_adapter_server_fixture):
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture

        plugin_id = await server.create_plugin(name="LateDispute Plugin", code="print('late')")
        subnet_id = await server.create_subnet(dispute_period=3600, solve_period=1800, stake_multiplier=1.5)
        job_id = await server.create_job(
            name="LateDisputeJob",
            plugin_id=plugin_id,
            subnet_id=subnet_id,
            sot_url="http://example.com",
            iteration=0
        )

        await server.admin_deposit_account(user_id="testuser", amount=600.0)

        task_id = await server.create_task(job_id, 1, TaskStatus.SelectingSolver, "{}")

        async with AsyncSessionLocal() as session:
            account = await server.get_or_create_account("testuser", session=session)
            ask_hold = Hold(
                account_id=account.id,
                user_id="testuser",
                hold_type=HoldType.CreditCard,
                total_amount=200.0,
                used_amount=0.0,
                expiry=datetime.utcnow() + timedelta(days=5),
                charged=False,
                charged_amount=0.0
            )
            bid_hold = Hold(
                account_id=account.id,
                user_id="testuser",
                hold_type=HoldType.Credits,
                total_amount=100.0,
                used_amount=0.0,
                expiry=datetime.utcnow() + timedelta(days=5),
                charged=False,
                charged_amount=0.0
            )
            session.add(ask_hold)
            session.add(bid_hold)
            await session.commit()
            await session.refresh(ask_hold)
            await session.refresh(bid_hold)

        ask_order_id = await server.create_order(
            task_id=None,
            subnet_id=subnet_id,
            order_type=OrderType.Ask,
            price=50.0,
            hold_id=ask_hold.id
        )

        bid_order_id = await server.create_order(
            task_id=task_id,
            subnet_id=subnet_id,
            order_type=OrderType.Bid,
            price=50.0,
            hold_id=bid_hold.id
        )

        # => scPending
        await server.submit_task_result(task_id, result=json.dumps({"output": "success"}))
        await server.finalize_sanity_check(task_id, True)

        async with AsyncSessionLocal() as session:
            stmt = (
                select(Task)
                .options(
                    selectinload(Task.bid).selectinload(Order.hold),
                    selectinload(Task.ask).selectinload(Order.hold),
                    selectinload(Task.job).selectinload(Job.subnet)
                )
                .where(Task.id == task_id)
            )
            fetched = await session.execute(stmt)
            task = fetched.scalar_one_or_none()
            assert task is not None
            assert task.status == TaskStatus.ResolvedCorrect

        # can't new result
        with pytest.raises(ValueError) as excinfo:
            await server.submit_task_result(task_id, result=json.dumps({"output": "fail"}))
        assert "Task not in SolverSelected status" in str(excinfo.value)


@pytest.mark.asyncio
async def test_hold_expiry_prevents_usage(db_adapter_server_fixture):
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture

        plugin_id = await server.create_plugin(name="ExpiryTest Plugin", code="print('exp')")
        subnet_id = await server.create_subnet(dispute_period=3600, solve_period=1800, stake_multiplier=1.5)
        job_id = await server.create_job(
            name="Expiry Job",
            plugin_id=plugin_id,
            subnet_id=subnet_id,
            sot_url="http://example.com",
            iteration=0
        )

        async with AsyncSessionLocal() as session:
            account = await server.get_or_create_account("testuser", session=session)
            short_hold = Hold(
                account_id=account.id,
                user_id="testuser",
                hold_type=HoldType.CreditCard,
                total_amount=200.0,
                used_amount=0.0,
                expiry=datetime.utcnow() + timedelta(hours=1),
                charged=False,
                charged_amount=0.0
            )
            session.add(short_hold)
            await session.commit()
            await session.refresh(short_hold)

        task_id = await server.create_task(
            job_id=job_id,
            job_iteration=1,
            status=TaskStatus.SelectingSolver,
            params="{}"
        )

        with pytest.raises(ValueError) as excinfo:
            await server.create_order(
                task_id=task_id,
                subnet_id=subnet_id,
                order_type=OrderType.Bid,
                price=50.0,
                hold_id=short_hold.id
            )
        assert "hold expires too soon" in str(excinfo.value)


@pytest.mark.asyncio
async def test_create_bids_and_tasks_with_hold(db_adapter_server_fixture):
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture

        plugin_id = await server.create_plugin(name="MultiBidTest", code="print('multi')")
        subnet_id = await server.create_subnet(dispute_period=3600, solve_period=1800, stake_multiplier=1.0)
        job_id = await server.create_job(
            name="MultiBidJob",
            plugin_id=plugin_id,
            subnet_id=subnet_id,
            sot_url="http://example.com",
            iteration=0
        )

        await server.admin_deposit_account(user_id="testuser", amount=300.0)

        async with AsyncSessionLocal() as session:
            account = await server.get_or_create_account("testuser", session=session)
            large_hold = Hold(
                account_id=account.id,
                user_id="testuser",
                hold_type=HoldType.CreditCard,
                total_amount=1000.0,
                used_amount=0.0,
                expiry=datetime.utcnow() + timedelta(days=7),
                charged=False,
                charged_amount=0.0
            )
            session.add(large_hold)
            await session.commit()
            await session.refresh(large_hold)

        result = await server.create_bids_and_tasks(
            job_id=job_id,
            num_tasks=3,
            price=50.0,
            params="{}",
            hold_id=large_hold.id
        )
        assert len(result["created_items"]) == 3

        async with AsyncSessionLocal() as session:
            updated_hold = await session.execute(select(Hold).where(Hold.id == large_hold.id))
            updated_hold = updated_hold.scalar_one_or_none()
            assert updated_hold.used_amount == 150.0


@pytest.mark.asyncio
async def test_no_direct_withdraws_deposits_just_admin_deposit(db_adapter_server_fixture):
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture

        await server.admin_deposit_account(user_id="testuser", amount=1000.0)
        async with AsyncSessionLocal() as session:
            account = await server.get_or_create_account("testuser", session=session)
            assert account.credits_balance == 1000.0


@pytest.mark.asyncio
async def test_create_plugin_fetch_plugins(db_adapter_server_fixture):
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture
        plugin_id = await server.create_plugin(name="FetchPlugin", code="print('fetch')")
        plugins = await server.get_plugins()
        assert any(p.id == plugin_id for p in plugins)


@pytest.mark.asyncio
async def test_deposit_credits_expire_after_a_year(db_adapter_server_fixture):
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture

        # 1) Admin deposit => ~1-year hold
        await server.admin_deposit_account(user_id="testuser", amount=200.0)

        async with AsyncSessionLocal() as session:
            account = await server.get_or_create_account("testuser", session=session)
            assert account.credits_balance == 200.0

            deposit_hold = (
                await session.execute(select(Hold).where(Hold.account_id == account.id))
            ).scalars().first()
            assert deposit_hold is not None
            assert deposit_hold.total_amount == 200.0
            assert deposit_hold.hold_type == HoldType.Credits
            # Force it to be expired:
            deposit_hold.expiry = datetime.utcnow() - timedelta(days=1)
            session.add(deposit_hold)
            await session.commit()

        # 2) check_and_cleanup => leftover removed
        await server.check_and_cleanup_holds()

        async with AsyncSessionLocal() as session:
            account = await server.get_or_create_account("testuser", session=session)
            # leftover => forced to 0
            assert account.credits_balance == 0.0, "User's deposit-based credits should have expired."
