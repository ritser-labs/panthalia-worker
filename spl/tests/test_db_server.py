import pytest
import pytest_asyncio
import json
from datetime import datetime, timedelta
from sqlalchemy import select
from sqlalchemy.orm import selectinload, joinedload
from unittest.mock import patch
import asyncio

from spl.models import (
    TaskStatus, OrderType, HoldType,
    CreditTxnType, EarningsTxnType, PlatformRevenueTxnType,
    Hold, Order, Task, PlatformRevenue, Account, Job
)
from spl.db.server.app import original_app
from spl.db.server.adapter.orders_tasks import DBAdapterOrdersTasksMixin
from spl.db.server.adapter import DBAdapterServer

from spl.db.init import AsyncSessionLocal


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

        balance_info = await server.get_balance_details_for_user()
        assert balance_info["credits_balance"] == 300.0


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

        # Buyer => deposit 500
        await server.admin_deposit_account(user_id="testuser", amount=500.0)

        # solver => solveruser
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

        # finalize => correct => solver gets new earnings hold, etc.
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

        # we can also check the derived balances if desired
        # e.g. buyer has leftover 500; solver has 0 indefinite credits, etc.
        # but the main logic here is matching & we trust the new approach.
