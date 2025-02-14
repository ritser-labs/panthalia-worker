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


@pytest.mark.asyncio
async def test_basic_setup(db_adapter_server_fixture):
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture

        subnet_id = await server.create_subnet(dispute_period=3600, solve_period=1800, stake_multiplier=1.5)
        plugin_id = await server.create_plugin(name="Test Plugin", code="print('hello')", subnet_id=subnet_id)
        job_id = await server.create_job(
            name="Test Job",
            plugin_id=plugin_id,
            sot_url="http://panthalia.com",
            iteration=0,
            limit_price=1
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

        subnet_id = await server.create_subnet(dispute_period=3600, solve_period=1800, stake_multiplier=2.0)
        plugin_id = await server.create_plugin(name="BidTest Plugin", code="print('bid')", subnet_id=subnet_id)
        job_id = await server.create_job(
            name="Bid Job",
            plugin_id=plugin_id,
            sot_url="http://sot_url",
            iteration=0,
            limit_price=1
        )

        async with server.get_async_session() as session:
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

        async with server.get_async_session() as session:
            order = await session.execute(select(Order).where(Order.id == bid_order_id))
            order = order.scalar_one_or_none()
            assert order is not None
            hold = await session.execute(select(Hold).where(Hold.id == cc_hold.id))
            hold = hold.scalar_one_or_none()
            assert hold.used_amount == 50.0
