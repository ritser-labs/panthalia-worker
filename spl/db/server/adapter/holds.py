from datetime import datetime, timedelta
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from ....models import (
    AsyncSessionLocal, Hold, HoldTransaction, Account, CreditTxnType, HoldType
)

DISPUTE_PAYOUT_DELAY_DAYS = 1

class DBAdapterHoldsMixin:
    async def select_hold_for_order(self, session: AsyncSession, account: Account, subnet, order_type, price: float, specified_hold_id: int = None):
        required_expiry_buffer = timedelta(days=DISPUTE_PAYOUT_DELAY_DAYS)
        min_expiry = datetime.utcnow() + timedelta(seconds=subnet.solve_period + subnet.dispute_period) + required_expiry_buffer

        if order_type.value == 'bid':
            required_amount = price
        else:
            required_amount = subnet.stake_multiplier * price

        if specified_hold_id is not None:
            stmt = select(Hold).where(Hold.id == specified_hold_id, Hold.account_id == account.id).options(selectinload(Hold.hold_transactions))
            result = await session.execute(stmt)
            hold = result.scalar_one_or_none()
            if not hold:
                raise ValueError("specified hold not found or not yours")
            if hold.charged:
                raise ValueError("this hold is already charged, cannot use")
            if hold.expiry < min_expiry:
                raise ValueError("hold expires too soon")
            if (hold.total_amount - hold.used_amount) < required_amount:
                raise ValueError("not enough hold amount")
            if hold.hold_type == HoldType.Credits and not hold.charged:
                used_before = any(ht.amount > 0 for ht in hold.hold_transactions)
                if not used_before:
                    if account.credits_balance < hold.total_amount:
                        raise ValueError("insufficient credits to back this externally created credits hold")
                    await self.add_credits_transaction(session, account, hold.total_amount, CreditTxnType.Subtract)
                    await session.flush()
            return hold
        else:
            stmt = select(Hold).where(
                Hold.account_id == account.id,
                Hold.charged == False,
                Hold.expiry > min_expiry,
                (Hold.total_amount - Hold.used_amount) >= required_amount
            ).options(selectinload(Hold.hold_transactions))
            result = await session.execute(stmt)
            hold = result.scalars().first()
            if hold:
                if hold.hold_type == HoldType.Credits and not hold.charged:
                    used_before = any(ht.amount > 0 for ht in hold.hold_transactions)
                    if not used_before:
                        if account.credits_balance < hold.total_amount:
                            raise ValueError("insufficient credits to back this externally created credits hold")
                        await self.add_credits_transaction(session, account, hold.total_amount, CreditTxnType.Subtract)
                        await session.flush()
                return hold

            if account.credits_balance >= required_amount:
                new_hold = Hold(
                    account_id=account.id,
                    user_id=account.user_id,
                    hold_type=HoldType.Credits,
                    total_amount=required_amount,
                    used_amount=0.0,
                    expiry=min_expiry,
                    charged=False,
                    charged_amount=0.0
                )
                session.add(new_hold)
                account.credits_balance -= required_amount
                await session.flush()
                return new_hold

            raise ValueError("no suitable hold found, please specify a hold or create one")

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

    async def charge_hold_fully(self, session: AsyncSession, hold: Hold, add_leftover_to_account: bool = True):
        if hold.charged:
            raise ValueError("hold already charged")
        hold.charged = True
        hold.charged_amount = hold.total_amount

        if add_leftover_to_account:
            leftover = hold.total_amount - hold.used_amount
            if leftover > 0:
                stmt = select(Account).where(Account.id == hold.account_id)
                result = await session.execute(stmt)
                account = result.scalar_one_or_none()
                if account:
                    await self.add_credits_transaction(session, account, leftover, CreditTxnType.Add)
