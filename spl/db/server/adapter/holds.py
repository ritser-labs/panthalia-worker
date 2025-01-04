# file: spl/db/server/adapter/holds.py

from datetime import datetime, timedelta
import logging
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from ....models import (
    Account, Hold, HoldTransaction, HoldType
)

DISPUTE_PAYOUT_DELAY_DAYS = 1
logger = logging.getLogger(__name__)

class DBAdapterHoldsMixin:
    async def select_hold_for_order(
        self,
        session: AsyncSession,
        account: Account,
        subnet,
        order_type,
        price: float,
        specified_hold_id: int = None
    ) -> Hold:
        required_amount = (
            price if order_type.value == 'bid' else (subnet.stake_multiplier * price)
        )

        if specified_hold_id is not None:
            stmt = (
                select(Hold)
                .where(
                    Hold.id == specified_hold_id,
                    Hold.account_id == account.id,
                    Hold.hold_type == HoldType.Credits,
                    (Hold.total_amount - Hold.used_amount) >= required_amount
                )
                .options(selectinload(Hold.hold_transactions))
            )
            hold = (await session.execute(stmt)).scalar_one_or_none()
            if hold:
                return hold
            else:
                logger.debug(
                    f"[select_hold_for_order] specified hold_id={specified_hold_id} "
                    "not valid or insufficient leftover. Falling back to any suitable hold."
                )

        q = (
            select(Hold)
            .where(
                Hold.account_id == account.id,
                Hold.hold_type == HoldType.Credits,
                (Hold.total_amount - Hold.used_amount) >= required_amount
            )
            .options(selectinload(Hold.hold_transactions))
        )
        hold = (await session.execute(q)).scalars().first()
        if not hold:
            raise ValueError(
                "No deposit hold found with enough leftover to place this order."
            )
        return hold

    async def reserve_funds_on_hold(self, session: AsyncSession, hold: Hold, amount: float, order):
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

    async def free_funds_from_hold(self, session: AsyncSession, hold: Hold, amount: float, order):
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
