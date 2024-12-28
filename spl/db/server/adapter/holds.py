from datetime import datetime, timedelta
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from ....models import (
    AsyncSessionLocal, Hold, HoldTransaction, Account, CreditTxnType, HoldType
)

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
        """
        Looks up an existing uncharged hold for the given account that has enough leftover
        amount, does not expire too soon, and matches the needed 'required_amount'.
        If 'specified_hold_id' is given, it must refer to a suitable hold.
        Otherwise, we search for any suitable hold. If none found, we raise an error.

        NOTE: This version removes the old fallback that created a brand-new hold from leftover credits.
        """
        # Dispute buffer for the entire solve+dispute window
        required_expiry_buffer = timedelta(days=DISPUTE_PAYOUT_DELAY_DAYS)
        # Must remain valid for solve_period + dispute_period + buffer
        min_expiry = (
            datetime.utcnow()
            + timedelta(seconds=subnet.solve_period + subnet.dispute_period)
            + required_expiry_buffer
        )

        # For a 'bid' we need exactly 'price' credits
        # For an 'ask' we need 'stake_multiplier * price'
        if order_type.value == 'bid':
            required_amount = price
        else:
            required_amount = subnet.stake_multiplier * price

        # If user explicitly specified a hold, check that hold only
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

        # Otherwise, search for any suitable uncharged hold with enough leftover
        # that isn't expiring too soon.
        stmt = select(Hold).where(
            Hold.account_id == account.id,
            Hold.charged == False,
            Hold.expiry > min_expiry,
            (Hold.total_amount - Hold.used_amount) >= required_amount
        ).options(selectinload(Hold.hold_transactions))

        result = await session.execute(stmt)
        hold = result.scalars().first()
        if hold:
            # Found a suitable hold
            return hold

        # If nothing found, raise error. We do NOT create a brand-new hold.
        raise ValueError("No suitable hold found. The deposit-based hold might be expired or used up.")

    
    async def reserve_funds_on_hold(self, session: AsyncSession, hold: Hold, amount: float, order):
        """
        Increases hold.used_amount by 'amount', ensuring leftover can cover it.
        """
        if hold.total_amount - hold.used_amount < amount:
            raise ValueError("not enough hold funds")
        hold.used_amount += amount
        hold_txn = HoldTransaction(
            hold_id=hold.id,
            order_id=order.id,
            amount=amount
        )
        session.add(hold_txn)

    async def free_funds_from_hold(self, session: AsyncSession, hold: Hold, amount: float, order):
        """
        Decreases hold.used_amount by 'amount'.
        """
        if hold.used_amount < amount:
            raise ValueError("not enough used amount in hold to free")
        hold.used_amount -= amount
        hold_txn = HoldTransaction(
            hold_id=hold.id,
            order_id=order.id,
            amount=-amount
        )
        session.add(hold_txn)

    async def charge_hold_fully(self, session: AsyncSession, hold: Hold):
        """
        Mark hold as fully charged. 
        If leftover > 0 => create a new hold with the same expiry, same user, etc. 
        We also link it by leftover_hold.parent_hold_id = hold.id
        """
        if hold.charged:
            raise ValueError("hold already charged")
        hold.charged = True
        hold.charged_amount = hold.total_amount

        leftover = hold.total_amount - hold.used_amount
        if leftover > 0:
            leftover_hold = Hold(
                account_id=hold.account_id,
                user_id=hold.user_id,
                hold_type=hold.hold_type,
                total_amount=leftover,
                used_amount=0.0,
                expiry=hold.expiry,  # same expiry
                charged=False,
                charged_amount=0.0,
                parent_hold_id=hold.id  # link them
            )
            session.add(leftover_hold)

            hold.used_amount = hold.total_amount
        else:
            hold.used_amount = hold.total_amount
