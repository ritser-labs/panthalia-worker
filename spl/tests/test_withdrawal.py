# file: spl/tests/test_withdrawal.py

import pytest
from spl.db.server.app import original_app
from spl.models import Hold, Account
from spl.models.enums import HoldType, WithdrawalStatus
from sqlalchemy import select

@pytest.mark.asyncio
async def test_withdrawal_flow(db_adapter_server_fixture):
    """
    End-to-end test of the new withdrawal flow when user only has an EARNINGS hold:
      1) deposit => user gets a credits hold
      2) forcibly remove that credits hold + create an EARNINGS hold with leftover=300
      3) confirm user's 'earnings_balance' is 300
      4) user requests withdrawal => 100 => leftover=200
      5) admin calls complete_withdrawal => hold is charged => leftover=200
    """
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture

        user_id = "test_withdrawer"
        server._user_id_getter = lambda: user_id

        # 1) deposit => user gets a deposit-based credits hold of 300
        await server.admin_deposit_account(user_id, 300)

        # 2) forcibly remove that credits hold & create a new EARNINGS hold
        async with server.get_async_session() as session:
            # fetch the user's account
            stmt_acc = select(Account).where(Account.user_id == user_id)
            acc_result = await session.execute(stmt_acc)
            account = acc_result.scalar_one_or_none()
            if not account:
                raise ValueError(f"No account found for user_id={user_id}")

            # find the deposit-based hold (credits)
            stmt_hold = select(Hold).where(
                Hold.account_id == account.id,
                Hold.hold_type == HoldType.Credits
            )
            hold_result = await session.execute(stmt_hold)
            deposit_hold = hold_result.scalar_one_or_none()
            if not deposit_hold:
                raise ValueError("Didn't find the deposit-based credits hold after admin_deposit_account")

            # remove the credits hold
            await session.delete(deposit_hold)
            await session.flush()

            # create a new EARNINGS hold with leftover=300
            earnings_hold = Hold(
                account_id=account.id,
                user_id=user_id,
                hold_type=HoldType.Earnings,
                total_amount=300.0,
                used_amount=0.0,
                expiry=deposit_hold.expiry,
                charged=False,
                charged_amount=0.0
            )
            session.add(earnings_hold)
            await session.commit()

        # 3) confirm user has 'earnings_balance'=300
        balance_before = await server.get_balance_details_for_user()
        assert balance_before["earnings_balance"] == 300, (
            f"Expected 300 earnings, got {balance_before['earnings_balance']}"
        )

        # 4) create a withdrawal => 100 => leftover=200
        withdrawal_id = await server.create_withdrawal_request(user_id, 100)
        w_obj = await server.get_withdrawal(withdrawal_id)
        assert w_obj is not None
        assert w_obj.status == WithdrawalStatus.PENDING
        assert w_obj.amount == 100

        after_req_balance = await server.get_balance_details_for_user()
        assert after_req_balance["earnings_balance"] == 200, "We reserved 100 from the EARNINGS hold"

        # 5) admin calls complete_withdrawal => sets status=FINALIZED, charges the hold
        await server.complete_withdrawal_flow(withdrawal_id)
        w_obj = await server.get_withdrawal(withdrawal_id)
        assert w_obj.status == WithdrawalStatus.FINALIZED

        final_balance = await server.get_balance_details_for_user()
        # leftover remains 200
        assert final_balance["earnings_balance"] == 200, (
            f"Expected 200 leftover in earnings, got {final_balance['earnings_balance']}"
        )

        print("Withdrawal from EARNINGS hold test passed!")

@pytest.mark.asyncio
async def test_reject_withdrawal_flow(db_adapter_server_fixture):
    """
    Ensures that rejecting a withdrawal:
      - sets the status to REJECTED
      - unreserves the previously used EARNINGS hold
      - leftover in the EARNINGS hold reverts to the original
    """
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture

        user_id = "reject_test_user"
        server._user_id_getter = lambda: user_id

        # 1) Let’s deposit into an EARNINGS hold for simplicity.
        #    Or deposit as credits & forcibly morph it into Earnings, etc.
        #    For brevity, let's replicate the approach from the existing withdrawal test:
        await server.admin_deposit_account(user_id, 500)

        # replace the user's single credits hold with an Earnings hold
        async with server.get_async_session() as session:
            acc_stmt = select(Account).where(Account.user_id == user_id)
            acc_res = await session.execute(acc_stmt)
            account = acc_res.scalar_one_or_none()
            assert account, "No account found after deposit?"

            # find the credits hold
            holds_q = select(Hold).where(Hold.account_id == account.id, Hold.hold_type == HoldType.Credits)
            hold_res = await session.execute(holds_q)
            deposit_credits_hold = hold_res.scalar_one_or_none()

            # remove it
            if deposit_credits_hold:
                await session.delete(deposit_credits_hold)
                await session.flush()

                # create an EARNINGS hold with leftover=500
                earnings_hold = Hold(
                    account_id=account.id,
                    user_id=user_id,
                    hold_type=HoldType.Earnings,
                    total_amount=500.0,
                    used_amount=0.0,
                    expiry=deposit_credits_hold.expiry,
                    charged=False,
                    charged_amount=0.0
                )
                session.add(earnings_hold)
                await session.commit()

        # 2) create a withdrawal request => say 200 => leftover=300
        withdrawal_id = await server.create_withdrawal_request(user_id, 200)
        w_obj = await server.get_withdrawal(withdrawal_id)
        assert w_obj.status == WithdrawalStatus.PENDING
        balance_before_rejection = await server.get_balance_details_for_user()
        assert balance_before_rejection["earnings_balance"] == 300.0

        # 3) Now "admin" rejects the withdrawal
        #    (we can just call the method directly, or call the route if you prefer)
        await server.reject_withdrawal_flow(withdrawal_id)

        w_obj = await server.get_withdrawal(withdrawal_id)
        assert w_obj.status == WithdrawalStatus.REJECTED, "Withdrawal should be REJECTED after reject call"

        # 4) check that leftover in hold is restored => 500
        balance_after_rejection = await server.get_balance_details_for_user()
        assert balance_after_rejection["earnings_balance"] == 500.0, (
            f"Expected 500 leftover, got {balance_after_rejection['earnings_balance']}"
        )

        print("[test_reject_withdrawal_flow] => PASS")
