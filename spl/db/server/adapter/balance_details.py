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

        async with AsyncSessionLocal() as session:
            # 1) Load the account + holds
            stmt = (
                select(Account)
                .where(Account.user_id == user_id)
                .options(joinedload(Account.holds))
            )
            result = await session.execute(stmt)
            account = result.scalars().unique().one_or_none()

            if not account:
                # If you want an auto-create behavior, you'd do it here.
                raise ValueError(f"No account found for user_id={user_id}")

            credits_balance = account.credits_balance
            earnings_balance = account.earnings_balance
            holds = account.holds

            locked_hold_amounts = {}
            detailed_holds = []

            for hold in holds:
                hold_type = (
                    hold.hold_type.value
                    if hasattr(hold.hold_type, "value")
                    else str(hold.hold_type)
                )

                # -----------------------------------------
                # CHANGE: skip fully charged holds entirely
                # -----------------------------------------
                if hold.charged:
                    # If the hold is fully charged, it no longer
                    # contributes to "locked" amounts at all.
                    locked_amount = 0.0
                else:
                    # The "locked" portion is hold.used_amount for uncharged holds
                    locked_amount = hold.used_amount

                    # Accumulate by hold type
                    locked_hold_amounts[hold_type] = (
                        locked_hold_amounts.get(hold_type, 0.0) + locked_amount
                    )

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

        # Sum all locked amounts
        total_locked = sum(locked_hold_amounts.values())

        return {
            "credits_balance": credits_balance,
            "earnings_balance": earnings_balance,
            "locked_hold_amounts": locked_hold_amounts,
            "detailed_holds": detailed_holds,
        }
