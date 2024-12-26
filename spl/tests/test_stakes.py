import pytest
from datetime import datetime, timedelta
from spl.db.server.app import original_app
from spl.models import AsyncSessionLocal, Hold, HoldType, Task
from spl.models.enums import OrderType  # needed for create_order calls
from spl.common import TaskStatus  # fix: ensure we import this so there's no "UnboundLocalError"

from sqlalchemy import select


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

        # Create a CC hold
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
            order_type=OrderType.Bid,  # fix: use enum
            price=50.0,
            hold_id=cc_hold.id
        )

        # Confirm used_amount=50
        async with AsyncSessionLocal() as session:
            hold_obj = await session.get(Hold, cc_hold.id)
            assert hold_obj.used_amount == 50.0, f"Expected used_amount=50, got {hold_obj.used_amount}"


@pytest.mark.asyncio
async def test_create_ask_order_with_cc_hold_and_match(db_adapter_server_fixture):
    """
    Full scenario:
     - testuser => buyer with some credits
     - solveruser => solver, has a CC hold
     - They place matching orders => finalize => solver’s deposit partially returned, gets paid, etc.
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
        from spl.db.server.adapter import DBAdapterServer
        solver_server = DBAdapterServer(user_id_getter=lambda: "solveruser")

        # solver => has a hold
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

        # Create a task => testuser’s job
        task_id = await server.create_task(
            job_id=job_id,
            job_iteration=1,
            status=TaskStatus.SelectingSolver,
            params="{}"
        )

        # Buyer => place BID=100 with new CC hold
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

        bid_order_id = await server.create_order(
            task_id=task_id,
            subnet_id=subnet_id,
            order_type=OrderType.Bid,  # fix
            price=100.0,
            hold_id=buyer_cc_hold.id
        )

        # solver => create ASK => same price=100 => using solver_hold
        ask_order_id = await solver_server.create_order(
            task_id=None,
            subnet_id=subnet_id,
            order_type=OrderType.Ask,  # fix
            price=100.0,
            hold_id=solver_hold.id
        )

        # solver => submit => scPending
        await solver_server.submit_task_result(task_id, result='{"output": "success"}')
        # finalize => correct => solver gets (100 - 10% fee=90) + stake(1.5*100=150 if needed)
        await server.finalize_sanity_check(task_id, True)

        # confirm
        async with AsyncSessionLocal() as session:
            # solver => check final balances
            solver_acc = await solver_server.get_or_create_account("solveruser", session=session)
            assert solver_acc.earnings_balance == 90.0, f"Expected 90 earnings, got {solver_acc.earnings_balance}"
            assert solver_acc.credits_balance == 150.0, f"Expected 150 credits (the stake), got {solver_acc.credits_balance}"


@pytest.mark.asyncio
async def test_stake_released_for_reuse(db_adapter_server_fixture):
    """
    After we finalize correct resolution, the solver's hold is freed => can re-stake.
    We'll do 2 tasks in a row => same solver hold => used_amount resets to 0 each time.
    """
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture

        plugin_id = await server.create_plugin(name="StakeReleaseTest", code="print('test-stake-release')")
        subnet_id = await server.create_subnet(dispute_period=3600, solve_period=1800, stake_multiplier=2.0)
        job_id = await server.create_job(
            name="StakeReleaseJob",
            plugin_id=plugin_id,
            subnet_id=subnet_id,
            sot_url="http://release_stake_test",
            iteration=0
        )

        # testuser => deposit 500 for bidding
        await server.admin_deposit_account(user_id="testuser", amount=500.0)

        # solver => "solveruser"
        from spl.db.server.adapter import DBAdapterServer
        solver_server = DBAdapterServer(user_id_getter=lambda: "solveruser")

        # big hold => 1000
        async with AsyncSessionLocal() as session:
            solver_acc = await solver_server.get_or_create_account("solveruser", session=session)
            solver_hold = Hold(
                account_id=solver_acc.id,
                user_id="solveruser",
                hold_type=HoldType.CreditCard,
                total_amount=1000.0,
                used_amount=0.0,
                expiry=datetime.utcnow() + timedelta(days=5),
                charged=False,
                charged_amount=0.0
            )
            session.add(solver_hold)
            await session.commit()
            solver_hold_id = solver_hold.id

        for i in range(1, 3):
            task_id = await server.create_task(
                job_id=job_id,
                job_iteration=i,
                status=TaskStatus.SelectingSolver,
                params=f'{{"step": {i}}}'
            )
            # testuser => create new hold each iteration => user_hold=200 => place bid=100
            bid_hold_id = None
            async with AsyncSessionLocal() as session:
                user_acc = await server.get_or_create_account("testuser", session=session)
                user_hold = Hold(
                    account_id=user_acc.id,
                    user_id="testuser",
                    hold_type=HoldType.Credits,
                    total_amount=200.0,
                    used_amount=0.0,
                    expiry=datetime.utcnow() + timedelta(days=5),
                    charged=False,
                    charged_amount=0.0
                )
                session.add(user_hold)
                await session.commit()
                bid_hold_id = user_hold.id

            # place bid
            bid_id = await server.create_order(
                task_id=task_id,
                subnet_id=subnet_id,
                order_type=OrderType.Bid,  # fix
                price=100.0,
                hold_id=bid_hold_id
            )
            # solver => ask => 100 => uses same solver_hold
            ask_id = await solver_server.create_order(
                task_id=None,
                subnet_id=subnet_id,
                order_type=OrderType.Ask,  # fix
                price=100.0,
                hold_id=solver_hold_id
            )

            # solver => scPending => finalize correct => hold freed
            await solver_server.submit_task_result(task_id, f'{{"step": {i}}}')
            await server.finalize_sanity_check(task_id, True)

            # confirm solver_hold used_amount=0 again
            async with AsyncSessionLocal() as session:
                updated_hold = await session.get(Hold, solver_hold_id)
                assert updated_hold.used_amount == 0.0, f"Iteration {i}: expected used_amount=0, got {updated_hold.used_amount}"
                assert not updated_hold.charged, f"Iteration {i}: expected hold not charged, but was charged."


@pytest.mark.asyncio
async def test_dispute_scenario_with_cc_hold(db_adapter_server_fixture):
    """
    If solver's solution is incorrect => stake fully charged => leftover => platform, etc.
    """
    from spl.db.server.adapter import DBAdapterServer

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

        solver_server = DBAdapterServer(user_id_getter=lambda: "solveruser")

        # solver => also has 500
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

            # bidder hold
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
            order_type=OrderType.Bid,  # fix
            price=50.0,
            hold_id=bidder_hold.id
        )
        # solver => ask => price=40
        ask_id = await solver_server.create_order(
            task_id=None,
            subnet_id=subnet_id,
            order_type=OrderType.Ask,  # fix
            price=40.0,
            hold_id=solver_hold.id
        )

        # solver => submit => scPending => "fail"
        await solver_server.submit_task_result(task_id, result='{"output": "fail"}')
        # finalize => incorrect => stake is lost
        await server.finalize_sanity_check(task_id, False)

        # check final
        async with AsyncSessionLocal() as session:
            fetched_task = await session.get(Task, task_id)
            assert fetched_task.status == TaskStatus.ResolvedIncorrect


@pytest.mark.asyncio
async def test_dispute_can_only_happen_while_result_uploaded(db_adapter_server_fixture):
    """
    Once a task is resolved, can't re-submit new results.
    """
    import json
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

        task_id = await server.create_task(
            job_id=job_id,
            job_iteration=1,
            status=TaskStatus.SelectingSolver,
            params="{}"
        )

        # create some holds
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

        # ask => 50
        ask_id = await server.create_order(
            task_id=None,
            subnet_id=subnet_id,
            order_type=OrderType.Ask,
            price=50.0,
            hold_id=ask_hold.id
        )
        # bid => 50
        bid_id = await server.create_order(
            task_id=task_id,
            subnet_id=subnet_id,
            order_type=OrderType.Bid,
            price=50.0,
            hold_id=bid_hold.id
        )

        # solver => scPending => finalize => correct
        await server.submit_task_result(task_id, result=json.dumps({"output": "success"}))
        await server.finalize_sanity_check(task_id, True)

        # now resolved => can't re-submit
        with pytest.raises(ValueError) as excinfo:
            await server.submit_task_result(task_id, result='{"output": "fail"}')
        assert "Task not in SolverSelected status" in str(excinfo.value)


@pytest.mark.asyncio
async def test_hold_expiry_prevents_usage(db_adapter_server_fixture):
    """
    If the hold expires too soon, we can't use it for placing an order.
    """
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

        # short hold => expires soon
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
                order_type=OrderType.Bid,  # fix
                price=50.0,
                hold_id=short_hold.id
            )
        # check substring
        assert "expires too soon" in str(excinfo.value).lower()
