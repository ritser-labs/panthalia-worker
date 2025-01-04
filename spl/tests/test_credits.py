import pytest
from spl.db.server.app import original_app
from sqlalchemy import select

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


