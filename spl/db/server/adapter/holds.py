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
        Integrates with Stripe capture for credit-card holds
        by calling self.capture_stripe_payment_intent(...).
        """

        # Non-CC logic
        if hold.hold_type != HoldType.CreditCard:
            if hold.used_amount < price:
                raise ValueError(
                    f"Cannot finalize {price} from hold {hold.id}; used_amount={hold.used_amount} is too small."
                )
            hold.used_amount -= price
            hold.charged_amount += price
            hold.total_amount -= price
            hold.charged = True
            if hold.total_amount < 0:
                raise ValueError(f"Charged more than total_amount in hold {hold.id}")

            await session.flush()
            return

        # ---------- CC hold logic ----------
        leftover = hold.total_amount - price
        if leftover < 0:
            raise ValueError(
                f"Cannot charge {price} from CC hold {hold.id}; total_amount={hold.total_amount} < price."
            )
        if hold.used_amount < price:
            raise ValueError(
                f"Cannot finalize {price} from CC hold {hold.id}; used_amount={hold.used_amount} is too small."
            )

        # (A) Actually capture on Stripe using the new function in DBAdapterStripeBillingMixin
        # The same `self` here has the method `capture_stripe_payment_intent` because
        # DBAdapterServer inherits from both DBAdapterHoldsMixin & DBAdapterStripeBillingMixin.
        await self.capture_stripe_payment_intent(hold, price, session)

        # (B) Now proceed with the local logic. We treat CC holds as fully consumed
        hold.charged_amount += price
        hold.total_amount = 0.0
        hold.used_amount = 0.0
        hold.charged = True
        await session.flush()

        # (C) If leftover > 0 => create leftover Credits hold
        if leftover > 0:
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
                parent_hold_id=hold.id
            )
            session.add(new_credits_hold)

            new_tx = CreditTransaction(
                account_id=hold.account_id,
                user_id=hold.user_id,
                amount=leftover,
                txn_type=CreditTxnType.Add,
                reason="CC leftover"
            )
            session.add(new_tx)

        await session.flush()
