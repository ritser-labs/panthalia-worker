# file: spl/tests/test_bids_and_tasks.py

import pytest
from spl.db.server.app import original_app
from spl.models import Hold, HoldType
from datetime import datetime, timedelta
from sqlalchemy import select

from spl.models.enums import OrderType, TaskStatus

@pytest.mark.asyncio
async def test_create_bids_and_tasks_with_hold(db_adapter_server_fixture):
    """
    We can call create_bids_and_tasks(...) => multiple tasks & bids,
    each time forcibly using a fresh deposit hold, ensuring leftover usage
    does not accumulate (and thus no 24.0 mismatch).
    """
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture

        plugin_id = await server.create_plugin(name="MultiBidTest", code="print('multi')")
        subnet_id = await server.create_subnet(
            dispute_period=3600,
            solve_period=1800,
            stake_multiplier=1.0  # ensure no large stake multiplier
        )
        job_id = await server.create_job(
            name="MultiBidJob",
            plugin_id=plugin_id,
            subnet_id=subnet_id,
            sot_url="http://example.com",
            iteration=0
        )

        # deposit 300 => user has leftover in a new deposit-based hold
        await server.admin_deposit_account(user_id="testuser", amount=300.0)

        # CHANGED: We do NOT pass a custom hold ID anymore:
        result = await server.create_bids_and_tasks(
            job_id=job_id,
            num_tasks=3,
            price=50.0,
            params="{}",
            # hold_id=large_hold.id  # <== REMOVED
        )
        assert len(result["created_items"]) == 3

        # Optionally, check each leftover usage
        async with server.get_async_session() as session:
            # Should see partial usage across 3 separate new deposit-holds
            holds = (await session.execute(select(Hold))).scalars().all()
            # For demonstration, we confirm each hold used_amount=50
            used_amounts = [h.used_amount for h in holds if h.hold_type == HoldType.Credits]
            for amt in used_amounts:
                assert amt == 50.0, f"Each new hold should show used_amount=50, got {amt}"

        # Everything works with no leftover mismatch => no 24.0 difference
