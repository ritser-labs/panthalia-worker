# spl/db/server/adapter/balance_details.py

import logging
import sqlalchemy
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from ....models import (
    CreditTransaction, CreditTxnType, 
    PlatformRevenue, PlatformRevenueTxnType,
    PendingWithdrawal, WithdrawalStatus,
    Hold, HoldType, Account
)
from ....models.enums import HoldType
from datetime import datetime

logger = logging.getLogger(__name__)

class DBAdapterBalanceDetailsMixin:
    async def get_balance_details_for_user(self) -> dict:
        """
        Returns a dictionary that includes:
         - credits_balance (derived)
         - earnings_balance (derived)
         - locked_hold_amounts (dict by hold_type)
         - detailed_holds (array of hold info)
        Raises ValueError if user/account not found.
        """
        user_id = self.get_user_id()

        async with self.get_async_session() as session:
            stmt = (
                select(Account)
                .where(Account.user_id == user_id)
                .options(joinedload(Account.holds))
            )
            result = await session.execute(stmt)
            account = result.scalars().unique().one_or_none()

            if not account:
                raise ValueError(f"No account found for user_id={user_id}")

            holds = account.holds

            # Derive balances from uncharged leftover holds
            derived_credits_balance = 0.0
            derived_earnings_balance = 0.0
            locked_hold_amounts = {}
            detailed_holds = []

            for hold in holds:
                hold_type = (
                    hold.hold_type.value
                    if hasattr(hold.hold_type, "value")
                    else str(hold.hold_type)
                )

                locked_amount = hold.used_amount
                locked_hold_amounts[hold_type] = (
                    locked_hold_amounts.get(hold_type, 0.0) + locked_amount
                )

                leftover = hold.total_amount - hold.used_amount
                # If it's uncharged leftover, it contributes to "balance" if it's a "credits" or "earnings" hold
                if hold.hold_type == HoldType.Credits:
                    derived_credits_balance += leftover
                elif hold.hold_type == HoldType.Earnings:
                    derived_earnings_balance += leftover

                hold_data = {
                    "hold_id": hold.id,
                    "hold_type": hold_type,
                    "total_amount": hold.total_amount,
                    "used_amount": hold.used_amount,
                    "leftover": leftover,
                    "expiry": hold.expiry.isoformat() if hold.expiry else None,
                    "charged": hold.charged,
                    "charged_amount": hold.charged_amount,
                    "status": "charged" if hold.charged else "active",
                }
                detailed_holds.append(hold_data)

        total_locked = sum(locked_hold_amounts.values())

        return {
            "credits_balance": derived_credits_balance,
            "earnings_balance": derived_earnings_balance,
            "locked_hold_amounts": locked_hold_amounts,
            "detailed_holds": detailed_holds,
        }

    async def check_invariant(self, session=None) -> dict:
        """
        For the invariant check, we ignore locked amounts and platform revenue.
        We just verify that:

        sum_of_all_credits_total_amount + sum_of_all_earnings_total_amount
        == total_deposited - total_withdrawn

        Where:
        - total_deposited = sum of all CreditTxnType.Add transactions
        - total_withdrawn = sum of all APPROVED withdrawals
        - sum_of_all_credits_total_amount = sum(Hold.total_amount) for hold_type=Credits
        - sum_of_all_earnings_total_amount = sum(Hold.total_amount) for hold_type=Earnings
        """
        import sqlalchemy
        from sqlalchemy import select
        from ....models import (
            CreditTransaction, CreditTxnType,
            PendingWithdrawal, WithdrawalStatus, Hold, HoldType
        )

        own_session = False
        if session is None:
            session = self.get_async_session()
            own_session = True
        try:
            # 1) Sum all deposit-based credit transactions (CreditTxnType.Add)
            stmt_deposits = select(
                sqlalchemy.func.sum(CreditTransaction.amount)
            ).where(CreditTransaction.txn_type == CreditTxnType.Add)
            total_deposited = (await session.execute(stmt_deposits)).scalar() or 0.0

            # 2) Sum all APPROVED withdrawals
            stmt_withdrawals = select(
                sqlalchemy.func.sum(PendingWithdrawal.amount)
            ).where(PendingWithdrawal.status == WithdrawalStatus.APPROVED)
            total_withdrawn = (await session.execute(stmt_withdrawals)).scalar() or 0.0

            # 3) Sum total_amount (NOT leftover or locked) for each hold with type=Credits or Earnings
            holds_stmt = select(Hold)
            all_holds = (await session.execute(holds_stmt)).scalars().all()

            sum_credits_total = 0.0
            sum_earnings_total = 0.0

            for hold in all_holds:
                if hold.hold_type == HoldType.Credits:
                    sum_credits_total += hold.total_amount
                elif hold.hold_type == HoldType.Earnings:
                    sum_earnings_total += hold.total_amount

            # 4) Check the desired equality:
            #    sum_credits_total + sum_earnings_total == total_deposited - total_withdrawn
            lhs = sum_credits_total + sum_earnings_total
            rhs = total_deposited - total_withdrawn
            difference = lhs - rhs
            invariant_holds = abs(difference) < 1e-9

            return {
                "total_deposited": float(total_deposited),
                "total_withdrawn": float(total_withdrawn),
                "sum_credits_total": float(sum_credits_total),
                "sum_earnings_total": float(sum_earnings_total),
                "invariant_holds": invariant_holds,
                "difference": float(difference),
            }
        finally:
            if own_session:
                await session.close()
