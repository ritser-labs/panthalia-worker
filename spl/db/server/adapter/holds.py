# spl/db/server/adapter/holds.py

from datetime import datetime, timedelta
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from ....models import (
    CreditTransaction, Hold, HoldTransaction, Account, CreditTxnType, HoldType
)
import logging

DISPUTE_PAYOUT_DELAY_DAYS = 1

class DBAdapterHoldsMixin:
    async def select_hold_for_order(
        self,
        session: AsyncSession,
        account: Account,
        subnet,
        order_type,
        price: float,
        specified_hold_id: int = None
    ):
        # no changes re: removing credits_balance references. We just ensure we do not
        # rely on or update any account.credits_balance. This method is about picking a hold.
        required_expiry_buffer = timedelta(days=DISPUTE_PAYOUT_DELAY_DAYS)
        min_expiry = (
            datetime.utcnow()
            + timedelta(seconds=subnet.solve_period + subnet.dispute_period)
            + required_expiry_buffer
        )

        if order_type.value == 'bid':
            required_amount = price
        else:
            required_amount = subnet.stake_multiplier * price

        if specified_hold_id is not None:
            stmt = select(Hold).where(
                Hold.id == specified_hold_id,
                Hold.account_id == account.id
            ).options(selectinload(Hold.hold_transactions))
            result = await session.execute(stmt)
            hold = result.scalar_one_or_none()

            if not hold:
                raise ValueError("Specified hold not found or does not belong to you.")
            if hold.charged:
                raise ValueError("This hold is already fully charged (cannot reuse).")
            if hold.expiry < min_expiry:
                raise ValueError("Specified hold expires too soon for this order.")
            if (hold.total_amount - hold.used_amount) < required_amount:
                raise ValueError("Not enough leftover amount on the specified hold.")

            return hold

        stmt = select(Hold).where(
            Hold.account_id == account.id,
            Hold.charged == False,
            Hold.expiry > min_expiry,
            (Hold.total_amount - Hold.used_amount) >= required_amount
        ).options(selectinload(Hold.hold_transactions))

        result = await session.execute(stmt)
        hold = result.scalars().first()
        if hold:
            return hold

        raise ValueError("No suitable hold found. The deposit-based hold might be expired or used up.")

    async def reserve_funds_on_hold(self, session: AsyncSession, hold: Hold, amount: float, order):
        if hold.total_amount - hold.used_amount < amount:
            raise ValueError("not enough hold funds")

        hold.used_amount += amount
        hold_txn = HoldTransaction(
            hold_id=hold.id,
            order_id=order.id,
            amount=amount
        )
        session.add(hold_txn)

        # Removed any references to "account.credits_balance -= amount" or the like.

    async def free_funds_from_hold(self, session: AsyncSession, hold: Hold, amount: float, order):
        if hold.used_amount < amount:
            raise ValueError("not enough used amount in hold to free")
        hold.used_amount -= amount
        hold_txn = HoldTransaction(
            hold_id=hold.id,
            order_id=order.id,
            amount=-amount
        )
        session.add(hold_txn)
        # Removed "account.credits_balance += amount"

    async def charge_hold_fully(self, session: AsyncSession, hold: Hold):
        if hold.charged:
            raise ValueError("hold already charged")

        hold.charged = True
        hold.charged_amount = hold.total_amount

        leftover = hold.total_amount - hold.used_amount
        if leftover > 0:
            if hold.hold_type == HoldType.CreditCard:
                new_hold_type = HoldType.Credits
                new_expiry = datetime.utcnow() + timedelta(days=365)
            else:
                new_hold_type = hold.hold_type
                new_expiry = hold.expiry

            leftover_hold = Hold(
                account_id=hold.account_id,
                user_id=hold.user_id,
                hold_type=new_hold_type,
                total_amount=leftover,
                used_amount=0.0,
                expiry=new_expiry,
                charged=False,
                charged_amount=0.0,
                parent_hold_id=hold.id
            )
            session.add(leftover_hold)
            hold.used_amount = hold.total_amount
        else:
            hold.used_amount = hold.total_amount
