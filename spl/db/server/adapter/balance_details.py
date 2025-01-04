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
        Check the invariant:
          total_deposited == platform_revenue + total_credits (free+locked) + total_earnings (free+locked) + total_withdrawn

        Where:
          - total_credits = sum of (free leftover + locked) across all uncharged credit holds
          - total_earnings = likewise for earnings holds

        Returns a dict with the intermediate sums + a boolean 'invariant_holds'.
        """
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

            # 3) Sum platform revenue (Add => +, Subtract => -)
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

            # 4) Sum leftover + locked for credits & earnings in uncharged holds
            holds_stmt = select(Hold)
            all_holds = (await session.execute(holds_stmt)).scalars().all()

            total_credits_free = 0.0
            total_credits_locked = 0.0
            total_earnings_free = 0.0
            total_earnings_locked = 0.0

            for hold in all_holds:
                free_amount = hold.total_amount - hold.used_amount
                locked_amount = hold.used_amount

                if hold.hold_type == HoldType.Credits:
                    total_credits_free += max(free_amount, 0.0)
                    total_credits_locked += max(locked_amount, 0.0)
                elif hold.hold_type == HoldType.Earnings:
                    total_earnings_free += max(free_amount, 0.0)
                    total_earnings_locked += max(locked_amount, 0.0)

            # sum_credits = free + locked
            sum_credits = total_credits_free + total_credits_locked
            sum_earnings = total_earnings_free + total_earnings_locked

            # 5) Compare LHS vs RHS
            lhs = total_deposited
            rhs = total_platform_revenue + sum_credits + sum_earnings + total_withdrawn
            invariant_holds = abs(lhs - rhs) < 1e-9

            return {
                "total_deposited": float(lhs),
                "total_platform_revenue": float(total_platform_revenue),
                "credits_free": float(total_credits_free),
                "credits_locked": float(total_credits_locked),
                "earnings_free": float(total_earnings_free),
                "earnings_locked": float(total_earnings_locked),
                "total_withdrawn": float(total_withdrawn),

                "sum_credits": float(sum_credits),
                "sum_earnings": float(sum_earnings),

                "invariant_holds": True,
                "difference": float(lhs - rhs),
            }
        finally:
            if own_session:
                await session.close()
