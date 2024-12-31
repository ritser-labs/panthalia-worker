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
        if hold.total_amount - hold.used_amount < amount:
            raise ValueError("not enough hold funds")

        hold.used_amount += amount
        hold_txn = HoldTransaction(
            hold_id=hold.id,
            order_id=order.id,
            amount=amount
        )
        session.add(hold_txn)

        # NEW: If this is a deposit-based (credits) hold, also subtract from the account’s credits_balance
        if hold.hold_type == HoldType.Credits:
            account_stmt = select(Account).where(Account.id == hold.account_id)
            acc_res = await session.execute(account_stmt)
            account = acc_res.scalar_one_or_none()
            if not account:
                raise ValueError("No account found for hold")

            if account.credits_balance < amount:
                raise ValueError("Not enough credits in the account to reserve on this hold")

            # Deduct from the account's main credits balance
            account.credits_balance -= amount

            # Also, record a credit-transaction for auditing
            credit_tx = CreditTransaction(
                account_id=account.id,
                user_id=account.user_id,
                amount=amount,
                txn_type=CreditTxnType.Subtract,
                reason="reserve_on_hold"
            )
            session.add(credit_tx)


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

        # NEW: If this is a deposit-based hold, increment the user’s credits_balance
        if hold.hold_type == HoldType.Credits:
            account_stmt = select(Account).where(Account.id == hold.account_id)
            acc_res = await session.execute(account_stmt)
            account = acc_res.scalar_one_or_none()
            if not account:
                raise ValueError("No account found for hold")

            # Return that freed portion back to the account balance
            account.credits_balance += amount

            # Record a credit transaction for clarity
            credit_tx = CreditTransaction(
                account_id=account.id,
                user_id=account.user_id,
                amount=amount,
                txn_type=CreditTxnType.Add,
                reason="free_from_hold"
            )
            session.add(credit_tx)


    async def charge_hold_fully(self, session: AsyncSession, hold: Hold):
        """
        Mark 'hold' as fully charged. If leftover > 0 => create a new hold.
        The new leftover hold's type and expiry may differ from the original:
        - If the original hold is CC => leftover becomes deposit-based Credits
        with a 1-year expiry, so the solver can re-use it more flexibly.
        - If leftover == 0 => do nothing special.
        """
        if hold.charged:
            raise ValueError("hold already charged")

        hold.charged = True
        hold.charged_amount = hold.total_amount

        leftover = hold.total_amount - hold.used_amount
        if leftover > 0:
            # Decide the new hold type + expiry based on the original hold type
            if hold.hold_type == HoldType.CreditCard:
                new_hold_type = HoldType.Credits
                new_expiry = datetime.utcnow() + timedelta(days=365)
            else:
                # For other hold types (Credits, Earnings, etc.), you can keep the same type/expiry
                # or adapt logic to your preference. Example:
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

            # Mark the original hold's used_amount = total_amount
            hold.used_amount = hold.total_amount
        else:
            # No leftover
            hold.used_amount = hold.total_amount

        logging.info(
            f"[charge_hold_fully] hold.id={hold.id} fully charged. leftover={leftover:.2f}. "
            f"Created leftover hold => type={new_hold_type if leftover > 0 else None}"
        )
