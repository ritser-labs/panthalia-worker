import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from sqlalchemy import select

from spl.models import (
    AsyncSessionLocal, TaskStatus, OrderType, HoldType,
    CreditTxnType, EarningsTxnType, PlatformRevenueTxnType, PermType,
    ServiceType, Base, Hold, Order, Task, PlatformRevenue, Account,
    EarningsTransaction, CreditTransaction
)
from spl.db.server.app import original_app
from spl.db.server.adapter import DBAdapterServer

@pytest_asyncio.fixture(autouse=True)
async def clear_database():
    """
    Clear the database before each test to ensure test isolation.
    """
    async with AsyncSessionLocal() as session:
        for table in reversed(Base.metadata.sorted_tables):
            await session.execute(table.delete())
        await session.commit()
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

        # Simulate a CC hold created externally
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
        # "testuser" server instance
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

        # Buyer (testuser) has 500 credits
        await server.admin_deposit_account(user_id="testuser", amount=500.0)

        # Solver user server instance (returns "solveruser" as user_id)
        solver_server = DBAdapterServer(user_id_getter=lambda: "solveruser")

        # Solver CC hold
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

        # Create a task as testuser
        task_id = await server.create_task(
            job_id=job_id,
            job_iteration=1,
            status=TaskStatus.SelectingSolver,
            params="{}"
        )

        # Buyer also uses a CC hold (not credits) for their bid
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

        # Create bid order as testuser
        bid_order_id = await server.create_order(
            task_id=task_id,
            subnet_id=subnet_id,
            order_type=OrderType.Bid,
            price=100.0,
            hold_id=buyer_cc_hold.id
        )

        # Create ask order as solveruser
        ask_order_id = await solver_server.create_order(
            task_id=None,
            subnet_id=subnet_id,
            order_type=OrderType.Ask,
            price=100.0,
            hold_id=solver_hold.id
        )

        # Submit correct result as solveruser
        await solver_server.submit_task_result(task_id, result={"output": "success"})

        async with AsyncSessionLocal() as session:
            task = await server.get_task(task_id)  # task belongs to testuser's job, but it's the same DB
            assert task.status == TaskStatus.ResolvedCorrect

            buyer_account = await server.get_or_create_account("testuser", session=session)
            solver_account = await solver_server.get_or_create_account("solveruser", session=session)

            # Buyer used a CC hold, so their credits remain unchanged
            assert buyer_account.credits_balance == 500.0
            # Solver got earnings after platform fee deduction
            assert solver_account.earnings_balance == 90.0
            # The solver hold was charged fully, leftover added to solver's credits
            assert solver_account.credits_balance == 150.0

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

        # testuser has 500 credits
        await server.admin_deposit_account(user_id="testuser", amount=500.0)

        async with AsyncSessionLocal() as session:
            solver_account = await server.get_or_create_account("testuser", session=session)
            # Solver CC hold
            solver_hold = Hold(
                account_id=solver_account.id,
                user_id="testuser",
                hold_type=HoldType.CreditCard,
                total_amount=300.0,
                used_amount=0.0,
                expiry=datetime.utcnow() + timedelta(days=5),
                charged=False,
                charged_amount=0.0
            )
            session.add(solver_hold)

            bidder_account = await server.get_or_create_account("testuser", session=session)
            # Bidder Credits hold
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

        task_id = await server.create_task(
            job_id=job_id,
            job_iteration=1,
            status=TaskStatus.SelectingSolver,
            params="{}"
        )

        bid_order_id = await server.create_order(
            task_id=task_id,
            subnet_id=subnet_id,
            order_type=OrderType.Bid,
            price=50.0,
            hold_id=bidder_hold.id
        )

        ask_order_id = await server.create_order(
            task_id=None,
            subnet_id=subnet_id,
            order_type=OrderType.Ask,
            price=40.0,
            hold_id=solver_hold.id
        )

        # Force a dispute scenario
        # Just override the method on server for testing:
        original_should_dispute = server.should_dispute
        async def mock_should_dispute(task):
            return True
        server.should_dispute = mock_should_dispute

        await server.submit_task_result(task_id, result={"output": "fail"})

        # Restore original method
        server.should_dispute = original_should_dispute

        async with AsyncSessionLocal() as session:
            task = await server.get_task(task_id)
            assert task.status == TaskStatus.ResolvedIncorrect

            # Check platform revenue got the stake (80.0 = 2.0 * 40.0)
            platform_revenues = (await session.execute(select(PlatformRevenue))).scalars().all()
            assert any(r.amount == 80.0 and r.txn_type == PlatformRevenueTxnType.Add for r in platform_revenues)

            solver_account = await server.get_or_create_account("testuser", session=session)
            bidder_account = await server.get_or_create_account("testuser", session=session)

            # Credits calculation: solver_account gets leftover after stake removal
            assert solver_account.credits_balance == 720.0

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

        # testuser has enough credits
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

        await server.submit_task_result(task_id, result={"output": "success"})

        # Trying another result after it was already resolved should fail
        with pytest.raises(ValueError) as excinfo:
            await server.submit_task_result(task_id, result={"output": "fail"})
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
