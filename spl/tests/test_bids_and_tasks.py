import pytest
from spl.db.server.app import original_app
from spl.models import AsyncSessionLocal, Hold, HoldType
from datetime import datetime, timedelta
from sqlalchemy import select

from spl.models.enums import OrderType  # needed so we can pass e.g. OrderType.Bid

@pytest.mark.asyncio
async def test_create_bids_and_tasks_with_hold(db_adapter_server_fixture):
    """
    We can call create_bids_and_tasks(...) => multiple tasks & bids
    referencing the same hold (which must have enough leftover).
    """
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

        # deposit 300 => fresh DB => so the user has 300 credits
        await server.admin_deposit_account(user_id="testuser", amount=300.0)

        # Create a large hold
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

        # create 3 tasks + bids => price=50 each => used_amount => 3*50=150
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
