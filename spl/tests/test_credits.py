# file: spl/tests/test_credits.py

import pytest
from spl.db.server.app import original_app
from sqlalchemy import select
from spl.models import Hold
from spl.models.enums import HoldType
from datetime import datetime, timedelta


@pytest.mark.asyncio
async def test_admin_deposit_credits(db_adapter_server_fixture):
    """
    Test 'admin_deposit_account' => ensures the user has
    300 leftover in a deposit-based hold (thus derived credits_balance=300).
    """
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture
        await server.admin_deposit_account(user_id="testuser", amount=300.0)

        # Now check via get_balance_details_for_user():
        balance_info = await server.get_balance_details_for_user()
        assert balance_info["credits_balance"] == 300.0, f"Expected 300.0, got {balance_info['credits_balance']}"


@pytest.mark.asyncio
async def test_no_direct_withdraws_deposits_just_admin_deposit(db_adapter_server_fixture):
    """
    Another deposit check: deposit 1000 => derived credits_balance => 1000.0
    """
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture

        await server.admin_deposit_account(user_id="testuser", amount=1000.0)
        balance_info = await server.get_balance_details_for_user()
        assert balance_info["credits_balance"] == 1000.0, f"Expected 1000.0, got {balance_info['credits_balance']}"


@pytest.mark.asyncio
async def test_cc_hold_charge_creates_leftover_credits_hold(db_adapter_server_fixture):
    """
    Ensure that when a credit-card hold is charged, it always charges the entire hold
    and creates a leftover Credits hold if 'hold.total_amount > price'.
    """
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture

        # STEP 1: Create a credit-card hold with total_amount=300
        user_id = "cc_test_user"
        async with server.get_async_session() as session:
            account = await server.get_or_create_account(user_id, session=session)
            cc_hold = Hold(
                account_id=account.id,
                user_id=user_id,
                hold_type=HoldType.CreditCard,
                total_amount=300.0,
                used_amount=80.0,  # Suppose part is used for an order
                expiry=datetime.utcnow() + timedelta(days=10),
                charged=False,
                charged_amount=0.0
            )
            session.add(cc_hold)
            await session.commit()
            await session.refresh(cc_hold)

        # STEP 2: Charge the hold for 80 -> triggers the "fully consume" logic for CC
        price_to_charge = 80.0
        async with server.get_async_session() as session:
            # Re-fetch hold in a new session
            stmt = select(Hold).where(Hold.id == cc_hold.id)
            original_hold_obj = (await session.execute(stmt)).scalar_one()

            # Use the adapter’s method to charge it
            await server.charge_hold_for_price(session, original_hold_obj, price_to_charge)
            await session.commit()

        # STEP 3: Confirm the old CC hold is fully consumed => total_amount=0, used_amount=0, charged_amount=80
        async with server.get_async_session() as session:
            stmt = select(Hold).where(Hold.id == cc_hold.id)
            reloaded_cc_hold = (await session.execute(stmt)).scalar_one()
            assert reloaded_cc_hold is not None
            assert reloaded_cc_hold.hold_type == HoldType.CreditCard
            assert reloaded_cc_hold.total_amount == 0.0, "Expected the CC hold to be fully zeroed out"
            assert reloaded_cc_hold.used_amount == 0.0, "Expected used_amount to be reset to 0"
            assert reloaded_cc_hold.charged_amount == price_to_charge, "Should match the charged price"
            assert reloaded_cc_hold.charged is True, "CC hold is now marked as charged"

        # STEP 4: Because old total_amount=300, leftover=300 - 80=220 => a new Credits hold must exist
        async with server.get_async_session() as session:
            leftover_holds_stmt = (
                select(Hold)
                .where(Hold.parent_hold_id == cc_hold.id, Hold.hold_type == HoldType.Credits)
            )
            leftover_credits_hold = (await session.execute(leftover_holds_stmt)).scalar_one_or_none()
            assert leftover_credits_hold is not None, "A new leftover 'Credits' hold was expected"
            assert leftover_credits_hold.total_amount == 220.0, "Leftover hold must match leftover=300-80"
            assert leftover_credits_hold.used_amount == 0.0
            assert leftover_credits_hold.charged is False
            assert leftover_credits_hold.charged_amount == 0.0

            # Also confirm leftover hold belongs to same user
            assert leftover_credits_hold.account_id == cc_hold.account_id
            assert leftover_credits_hold.user_id == cc_hold.user_id

        print("test_cc_hold_charge_creates_leftover_credits_hold => PASS")


@pytest.mark.asyncio
async def test_credits_expire_after_year(db_adapter_server_fixture):
    """
    Ensures that when a 'Credits' hold expires (past its expiry time),
    the hold is forcibly "charged" during check_and_cleanup_holds()
    and leftover is zeroed out.
    """
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture
        user_id = "test_expire_credits"

        # STEP 1: First, deposit 200 "credits" so the ledger is balanced
        await server.admin_deposit_account(user_id=user_id, amount=200.0)

        # We'll forcibly set the hold's expiry in the past and partially used
        async with server.get_async_session() as session:
            # Find the newly created deposit-based hold
            stmt = (
                select(Hold)
                .where(Hold.user_id == user_id, Hold.hold_type == HoldType.Credits)
                .order_by(Hold.id.desc())
            )
            deposit_hold = (await session.execute(stmt)).scalars().first()
            assert deposit_hold, "Expected a deposit-based hold after admin_deposit_account"

            # For demonstration: set expiry 10 seconds in the past
            deposit_hold.expiry = datetime.utcnow() - timedelta(seconds=10)
            deposit_hold.used_amount = 50.0  # leftover=150
            await session.commit()
            await session.refresh(deposit_hold)

        # STEP 2: Run the background hold cleanup => detects the expired hold & forcibly charges leftover
        await server.check_and_cleanup_holds()

        # STEP 3: Confirm the hold is now fully "charged" => leftover=0
        async with server.get_async_session() as session:
            stmt = select(Hold).where(Hold.id == deposit_hold.id)
            reloaded_hold = (await session.execute(stmt)).scalar_one_or_none()
            assert reloaded_hold is not None
            assert reloaded_hold.total_amount == 0.0, "Expired credits hold should be fully charged"
            assert reloaded_hold.used_amount == 0.0,  "Used amount should be zeroed out"
            assert reloaded_hold.charged,            "Hold must be marked as 'charged'"
            assert reloaded_hold.charged_amount == 200.0, (
                "charged_amount should match the original total_amount of 200.0, "
                "since leftover was forcibly charged"
            )

        print("test_credits_expire_after_year => PASS")
