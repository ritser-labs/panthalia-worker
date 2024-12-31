# spl/db/server/adapter/balance_details.py

import logging
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from ....models import AsyncSessionLocal, Account, Hold
from ....models.enums import HoldType
from datetime import datetime

logger = logging.getLogger(__name__)

class DBAdapterBalanceDetailsMixin:
    async def get_balance_details_for_user(self) -> dict:
        """
        Returns a dictionary that includes:
         - credits_balance
         - earnings_balance
         - locked_hold_amounts (dict by hold_type)
         - detailed_holds (array of hold info)
        Raises ValueError if user/account not found.
        """
        user_id = self.get_user_id()

        # We open a new session to load Account and its 'holds' in one query:
        async with AsyncSessionLocal() as session:
            # 1) either re-query the account using joinedload
            stmt = (
                select(Account)
                .where(Account.user_id == user_id)
                .options(joinedload(Account.holds))
            )
            result = await session.execute(stmt)
            account = result.scalars().unique().one_or_none()

            # 2) If the account does not exist, optionally create it, or raise error.
            if not account:
                # If you want "create" behavior, do that here. Otherwise, raise:
                #   raise ValueError(f"No account found for user_id={user_id}")
                # Example "auto-create" logic (comment out if not wanted):
                account = Account(
                    user_id=user_id,
                    credits_balance=0.0,
                    earnings_balance=0.0,
                )
                session.add(account)
                await session.commit()
                await session.refresh(account, ["holds"])  # ensure holds is loaded

            # Now 'account.holds' is attached to this session, so we can iterate safely.
            credits_balance = account.credits_balance
            earnings_balance = account.earnings_balance
            holds = account.holds  # no DetachedInstanceError now!

            locked_hold_amounts = {}
            detailed_holds = []

            for hold in holds:
                locked_amount = hold.used_amount
                hold_type = (
                    hold.hold_type.value
                    if hasattr(hold.hold_type, "value")
                    else str(hold.hold_type)
                )
                locked_hold_amounts[hold_type] = locked_hold_amounts.get(hold_type, 0.0) + locked_amount

                leftover = hold.total_amount - hold.used_amount
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

        return {
            "credits_balance": credits_balance,
            "earnings_balance": earnings_balance,
            "locked_hold_amounts": locked_hold_amounts,
            "detailed_holds": detailed_holds,
        }
