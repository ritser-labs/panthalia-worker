import pytest
from spl.db.server.app import original_app
from spl.models import AsyncSessionLocal
from sqlalchemy import select

@pytest.mark.asyncio
async def test_admin_deposit_credits(db_adapter_server_fixture):
    """
    Test 'admin_deposit_account' => ensures 'credits_balance' is updated to 300.0
    (since we have a fresh DB for each test now).
    """
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture
        await server.admin_deposit_account(user_id="testuser", amount=300.0)

        async with AsyncSessionLocal() as session:
            account = await server.get_or_create_account("testuser", session=session)
            assert account.credits_balance == 300.0, f"Expected 300.0, got {account.credits_balance}"


@pytest.mark.asyncio
async def test_no_direct_withdraws_deposits_just_admin_deposit(db_adapter_server_fixture):
    """
    Another deposit check: deposit 1000 => should see exactly 1000 in credits_balance.
    """
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture

        await server.admin_deposit_account(user_id="testuser", amount=1000.0)
        async with AsyncSessionLocal() as session:
            account = await server.get_or_create_account("testuser", session=session)
            assert account.credits_balance == 1000.0, f"Expected 1000.0, got {account.credits_balance}"


@pytest.mark.asyncio
async def test_deposit_credits_expire_after_a_year(db_adapter_server_fixture):
    """
    We deposit 200 => that implicitly creates a 1-year hold for 200. We artificially
    set that hold to be expired, then call check_and_cleanup_holds(). The leftover
    should be forcibly removed from user credits.
    """
    from datetime import datetime, timedelta
    from spl.models import Hold

    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture

        # deposit => ~1-year hold
        await server.admin_deposit_account(user_id="testuser", amount=200.0)

        async with AsyncSessionLocal() as session:
            account = await server.get_or_create_account("testuser", session=session)
            assert account.credits_balance == 200.0

            # find the deposit-based hold
            deposit_hold = (
                await session.execute(select(Hold).where(Hold.account_id == account.id))
            ).scalars().first()
            deposit_hold.expiry = datetime.utcnow() - timedelta(days=1)  # force it expired
            session.add(deposit_hold)
            await session.commit()

        # forcibly remove leftover from expired
        await server.check_and_cleanup_holds()

        async with AsyncSessionLocal() as session:
            account = await server.get_or_create_account("testuser", session=session)
            # leftover forced to 0
            assert account.credits_balance == 0.0, f"Expected 0 after expiry, got {account.credits_balance}"
