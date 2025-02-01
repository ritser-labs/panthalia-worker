# file: spl/db/server/adapter/holds.py

from datetime import datetime, timedelta
import logging
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from ....models import (
    OrderType, Hold, HoldTransaction, HoldType, CreditTransaction, CreditTxnType
)
from typing import Optional

DISPUTE_PAYOUT_DELAY_DAYS = 1
logger = logging.getLogger(__name__)

class DBAdapterHoldsMixin:
    async def select_hold_for_order(
        self,
        session: AsyncSession,
        account,
        subnet,
        order_type: OrderType,
        price: int,
        specified_hold_id: Optional[int] = None
    ) -> Hold:
        """
        For Bid orders, the required amount is simply `price`.
        For Ask orders, it is `subnet.stake_multiplier * price`.
        If a hold is explicitly specified via `specified_hold_id`, it must have enough leftover;
        otherwise a ValueError is raised. If not specified, then choose the first hold with
        enough leftover.
        """
        required_amount = price if order_type == OrderType.Bid else subnet.stake_multiplier * price

        if specified_hold_id is not None:
            stmt = (
                select(Hold)
                .where(
                    Hold.id == specified_hold_id,
                    Hold.account_id == account.id,
                    Hold.hold_type.in_([HoldType.CreditCard, HoldType.Credits]),
                    (Hold.total_amount - Hold.used_amount) >= required_amount
                )
                .options(selectinload(Hold.hold_transactions))
            )
            hold = (await session.execute(stmt)).scalar_one_or_none()
            if not hold:
                raise ValueError(
                    f"Specified hold_id={specified_hold_id} is invalid or does not have enough leftover "
                    f"(requires at least {required_amount})."
                )
            return hold

        query = (
            select(Hold)
            .where(
                Hold.account_id == account.id,
                Hold.hold_type.in_([HoldType.CreditCard, HoldType.Credits]),
                (Hold.total_amount - Hold.used_amount) >= required_amount
            )
            .options(selectinload(Hold.hold_transactions))
        )
        hold = (await session.execute(query)).scalars().first()
        if not hold:
            raise ValueError(
                f"No suitable hold found with leftover ≥ {required_amount} to place this {order_type} order."
            )
        return hold



    async def reserve_funds_on_hold(self, session: AsyncSession, hold: Hold, amount: int, order):
        leftover_before = hold.total_amount - hold.used_amount
        if leftover_before < amount:
            raise ValueError(
                f"Hold {hold.id} has leftover={leftover_before}, which is < requested={amount}"
            )
        hold.used_amount += amount
        txn = HoldTransaction(
            hold_id=hold.id,
            order_id=order.id,
            amount=amount
        )
        session.add(txn)

    async def free_funds_from_hold(self, session: AsyncSession, hold: Hold, amount: int, order):
        if hold.used_amount < amount:
            raise ValueError(
                f"Hold {hold.id} used_amount={hold.used_amount}, cannot free {amount}"
            )
        hold.used_amount -= amount
        txn = HoldTransaction(
            hold_id=hold.id,
            order_id=order.id,
            amount=(-amount)
        )
        session.add(txn)

    async def charge_hold_for_price(self, session: AsyncSession, hold: Hold, price: int):
        """
        Partially or fully charge `price` from 'hold'.

        If the hold type is NOT a credit card:
         - If hold.used_amount < price, raise an error.
         - Decrease used_amount by `price` and total_amount by `price`.
         - Increase charged_amount by `price`.
         - Mark hold as charged=True if total_amount goes to 0 or below.
         - If total_amount < 0, raise ValueError (charged more than total).

        If the hold type IS a credit card (HoldType.CreditCard):
         - We treat the entire hold.total_amount as "charged" (zero out total_amount & used_amount).
         - We only actually apply `price` to hold.charged_amount, i.e. hold.charged_amount += price.
         - If leftover > 0 (i.e. hold.total_amount > price), we create a leftover hold of type=Credits 
           in the same account for (hold.total_amount - price).
        """

        from ....models.enums import HoldType
        if hold.hold_type != HoldType.CreditCard:
            # Original partial charge logic for non-credit-card holds
            if hold.used_amount < price:
                raise ValueError(
                    f"Cannot finalize {price} from hold {hold.id}; used_amount={hold.used_amount} is too small."
                )

            hold.used_amount -= price
            hold.charged_amount += price
            hold.total_amount -= price
            hold.charged = True  # we consider it 'charged' at least partially

            if hold.total_amount < 0:
                raise ValueError(f"Charged more than total_amount in hold {hold.id}")

            await session.flush()

        else:
            # Special logic for credit card holds: fully consume the hold
            if hold.used_amount < price:
                raise ValueError(
                    f"Cannot finalize {price} from credit-card hold {hold.id}; "
                    f"used_amount={hold.used_amount} is too small."
                )

            leftover = hold.total_amount - price
            if leftover < 0:
                raise ValueError(
                    f"Cannot charge {price} from credit-card hold {hold.id} because total_amount={hold.total_amount} "
                    "would go negative."
                )

            # Mark the original CC hold as fully consumed
            hold.charged_amount += price
            hold.total_amount = 0.0
            hold.used_amount = 0.0
            hold.charged = True

            await session.flush()

            # If there's leftover from the CC hold, create a new Credits-type hold
            if leftover > 0:
                from ....models import Hold
                from datetime import datetime, timedelta

                expiry_date = datetime.utcnow() + timedelta(days=365)
                new_credits_hold = Hold(
                    account_id=hold.account_id,
                    user_id=hold.user_id,
                    hold_type=HoldType.Credits,
                    total_amount=leftover,
                    used_amount=0.0,
                    expiry=expiry_date,
                    charged=False,
                    charged_amount=0.0,
                    parent_hold_id=hold.id  # optional: track lineage
                )
                session.add(new_credits_hold)
                # 1) Also log a "deposit" transaction for leftover
                new_tx = CreditTransaction(
                    account_id=hold.account_id,
                    user_id=hold.user_id,
                    amount=leftover,
                    txn_type=CreditTxnType.Add,
                    reason="CC leftover"
                )
                session.add(new_tx)

            # Done
            await session.flush()

