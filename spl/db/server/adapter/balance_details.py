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
        NEW VERSION:
        Check the invariant by ignoring locked amounts entirely. We verify that:
        
           (deposits - withdrawals) == (sum_of_total_amounts_for_Credits_and_Earnings + total_platform_revenue)

        1) total_deposited is the sum of CreditTxnType.Add credit transactions
        2) total_withdrawn is the sum of APPROVED PendingWithdrawal amounts
        3) total_platform_revenue sums up all PlatformRevenue (Add => +, Subtract => -)
        4) sum_of_total_amounts_for_Credits_and_Earnings is the sum of hold.total_amount
           for all holds with hold_type in [HoldType.Credits, HoldType.Earnings],
           ignoring used_amount entirely.

        Returns a dict with the intermediate sums + a boolean 'invariant_holds'.
        """
        own_session = False
        if session is None:
            session = self.get_async_session()
            own_session = True

        try:
            # (1) Sum all deposit-based credit transactions (CreditTxnType.Add)
            stmt_deposits = select(
                sqlalchemy.func.sum(CreditTransaction.amount)
            ).where(CreditTransaction.txn_type == CreditTxnType.Add)
            total_deposited = (await session.execute(stmt_deposits)).scalar() or 0.0

            # (2) Sum all APPROVED withdrawals
            stmt_withdrawals = select(
                sqlalchemy.func.sum(PendingWithdrawal.amount)
            ).where(PendingWithdrawal.status == WithdrawalStatus.APPROVED)
            total_withdrawn = (await session.execute(stmt_withdrawals)).scalar() or 0.0

            # (3) Sum platform revenue (Add => +, Subtract => -)
            stmt_revenue = select(
                sqlalchemy.func.sum(
                    PlatformRevenue.amount *
                    sqlalchemy.case(
                        (PlatformRevenue.txn_type == PlatformRevenueTxnType.Add, 1),
                        (PlatformRevenue.txn_type == PlatformRevenueTxnType.Subtract, -1),
                        else_=0
                    )
                )
            )
            total_platform_revenue = (await session.execute(stmt_revenue)).scalar() or 0.0

            # (4) Sum total_amount for all 'Credits' or 'Earnings' holds
            holds_stmt = select(Hold)
            all_holds = (await session.execute(holds_stmt)).scalars().all()

            sum_credits_and_earnings = 0.0
            for hold in all_holds:
                if hold.hold_type in [HoldType.Credits, HoldType.Earnings]:
                    sum_credits_and_earnings += hold.total_amount

            # LHS vs RHS
            lhs = total_deposited - total_withdrawn
            rhs = sum_credits_and_earnings + total_platform_revenue

            difference = lhs - rhs
            invariant_holds = abs(difference) < 1e-9

            return {
                "invariant_holds": invariant_holds,
                "difference": int(difference),
                "total_deposited": int(total_deposited),
                "total_withdrawn": int(total_withdrawn),
                "total_platform_revenue": int(total_platform_revenue),
                "sum_credits_and_earnings": int(sum_credits_and_earnings),
                "lhs_deposits_minus_withdrawals": int(lhs),
                "rhs_credits_earnings_plus_platform": int(rhs),
            }
        finally:
            if own_session:
                await session.close()
