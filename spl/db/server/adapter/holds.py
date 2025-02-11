# file: spl/db/server/adapter/holds.py

from datetime import datetime, timedelta
import logging
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from ....models import (
    OrderType, Hold, HoldTransaction, HoldType, CreditTransaction, CreditTxnType
)

DISPUTE_PAYOUT_DELAY_DAYS = 1
logger = logging.getLogger(__name__)

def allowed_fallback(original_hold, candidate=None):
    """
    DRY helper that defines the allowed fallback hold constraint.
    
    If candidate is None, returns a SQLAlchemy filter expression.
    Otherwise, returns a Boolean indicating if the candidate hold is allowed.

    Constraint:
      - If the original hold is a Deposit hold (i.e. hold_type==Credits and parent_hold_id is None),
        then candidate must also be a Credits hold with no parent.
      - Otherwise (if the original is CreditCard or a non-deposit Credits hold),
        candidate must have hold_type in [CreditCard, Credits].
    """
    if candidate is None:
        if original_hold.hold_type == HoldType.Credits and original_hold.parent_hold_id is None:
            return and_(Hold.hold_type == HoldType.Credits, Hold.parent_hold_id.is_(None))
        else:
            return Hold.hold_type.in_([HoldType.CreditCard, HoldType.Credits])
    else:
        if original_hold.hold_type == HoldType.Credits and original_hold.parent_hold_id is None:
            return candidate.hold_type == HoldType.Credits and candidate.parent_hold_id is None
        else:
            return candidate.hold_type in [HoldType.CreditCard, HoldType.Credits]

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
                f"No suitable hold found with leftover ΓëÑ {required_amount} to place this {order_type} order."
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
        """
        Frees `amount` from the user's holds. We first try the given `hold`:
          - If `hold.used_amount >= amount`, free it all from there.
          - Otherwise, search other uncharged holds owned by the same user for one that can fully free `amount`.
        We do NOT partially free from multiple holds.

        New constraint: If the original hold is a Deposit hold (Credits with no parent),
        any fallback candidate must also be a Deposit hold. Otherwise, fallback candidates
        are allowed if they are of type CreditCard or Credits.
        """
        if hold.used_amount >= amount:
            hold.used_amount -= amount
            txn = HoldTransaction(
                hold_id=hold.id,
                order_id=order.id,
                amount=(-amount)
            )
            session.add(txn)
            return

        user_id = hold.user_id
        # Get the SQLAlchemy filter from our helper
        allowed_filter = allowed_fallback(hold)
        stmt = (
            select(Hold)
            .where(
                Hold.user_id == user_id,
                Hold.id != hold.id,
                allowed_filter,
                Hold.used_amount >= amount
            )
            .order_by(Hold.id.asc())
        )
        candidates = (await session.execute(stmt)).scalars().all()
        for candidate in candidates:
            if candidate.used_amount >= amount and allowed_fallback(hold, candidate):
                candidate.used_amount -= amount
                txn = HoldTransaction(
                    hold_id=candidate.id,
                    order_id=order.id,
                    amount=(-amount)
                )
                session.add(txn)
                return

        raise ValueError(
            f"Cannot free {amount} from any hold. The specified hold had used_amount={hold.used_amount}, "
            f"and no other hold can fully free {amount} either."
        )

    async def charge_hold_for_price(self, session: AsyncSession, hold: Hold, price: int):
        """
        Captures a payment for the given price from one of the user's holds.
        
        For non-CreditCard holds we require hold.used_amount >= price.
        For CreditCard holds we require both hold.total_amount >= price 
        and hold.used_amount >= price.
        
        If the given hold meets the condition, we charge it via _charge_one_hold.
        Otherwise, we search (in order) for another uncharged hold owned by the user that
        can cover the entire price.

        New constraint: If the original hold was a Deposit hold (Credits with no parent),
        any fallback candidate must also be a Deposit hold; otherwise, fallback candidate must be
        of type CreditCard or Credits.
        """
        can_charge = False
        if hold.hold_type != HoldType.CreditCard:
            if hold.used_amount >= price:
                can_charge = True
        else:
            if hold.total_amount >= price and hold.used_amount >= price:
                can_charge = True

        if can_charge:
            await self._charge_one_hold(session, hold, price)
            return

        user_id = hold.user_id
        stmt = (
            select(Hold)
            .where(
                Hold.user_id == user_id,
                Hold.id != hold.id
            )
            .order_by(Hold.id.asc())
        )
        candidates = (await session.execute(stmt)).scalars().all()
        for candidate in candidates:
            if not allowed_fallback(hold, candidate):
                continue
            if candidate.hold_type != HoldType.CreditCard:
                if candidate.used_amount >= price:
                    await self._charge_one_hold(session, candidate, price)
                    return
            else:
                if candidate.total_amount >= price and candidate.used_amount >= price:
                    await self._charge_one_hold(session, candidate, price)
                    return

        raise ValueError(
            f"No single hold can cover price={price}. "
            f"Specified hold {hold.id} didn't have enough funds (total_amount or used_amount)."
        )

    async def _charge_one_hold(self, session: AsyncSession, hold: Hold, price: int):
        """
        Charges exactly one hold for the specified price.
        
        For non-CreditCard holds, subtracts price from both used_amount and total_amount,
        marks the hold as charged, and flushes.
        
        For CreditCard holds, the logic is as follows:
          1. Check that hold.total_amount >= price and hold.used_amount >= price.
          2. Save original values, compute leftover.
          3. Capture the payment on Stripe (via capture_stripe_payment_intent).
          4. Mark the original CC hold as charged.
          5. If there is leftover, create a new Credits hold with that leftover and record a credit transaction.
        """
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

        if hold.total_amount < price:
            raise ValueError(
                f"Cannot charge {price} from CC hold {hold.id}; total_amount={hold.total_amount} < price."
            )
        if hold.used_amount < price:
            raise ValueError(
                f"Cannot finalize {price} from CC hold {hold.id}; used_amount={hold.used_amount} is too small."
            )

        original_total = hold.total_amount
        original_used = hold.used_amount
        leftover = original_total - price
        leftover_used = original_used - price

        await self.capture_stripe_payment_intent(hold, price, session)

        hold.charged_amount += price
        hold.total_amount = 0.0
        hold.used_amount = 0.0
        hold.charged = True
        await session.flush()

        if leftover > 0:
            expiry_date = datetime.utcnow() + timedelta(days=365)
            new_credits_hold = Hold(
                account_id=hold.account_id,
                user_id=hold.user_id,
                hold_type=HoldType.Credits,
                total_amount=leftover,
                used_amount=leftover_used,
                expiry=expiry_date,
                charged=False,
                charged_amount=0.0,
                parent_hold_id=hold.id
            )
            session.add(new_credits_hold)

            from ....models.enums import CreditTxnType
            from ....models import CreditTransaction
            new_tx = CreditTransaction(
                account_id=hold.account_id,
                user_id=hold.user_id,
                amount=leftover,
                txn_type=CreditTxnType.Add,
                reason="CC leftover"
            )
            session.add(new_tx)
        await session.flush()
