# file: spl/db/server/adapter/withdrawals.py

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from datetime import datetime
from ....models.enums import WithdrawalStatus, HoldType
from ....models import PendingWithdrawal, Account, Hold, HoldTransaction

class DBAdapterWithdrawalsMixin:
    """
    Mixin for handling user withdrawals from their Earnings balance.
    """

    async def create_withdrawal_request(self, user_id: str, amount: int) -> int:
        """
        Creates a new PendingWithdrawal in status=PENDING. Also 'reserves' that amount
        in an Earnings hold (so the user cannot double-spend those same earnings).

        Raises ValueError if there's no single Earnings hold with enough leftover.
        """
        if amount <= 0:
            raise ValueError("Cannot withdraw a non-positive amount.")

        async with self.get_async_session() as session:
            # Eager-load the user's Account plus its holds:
            stmt = (
                select(Account)
                .where(Account.user_id == user_id)
                .options(joinedload(Account.holds))
            )
            # Important => .unique() to avoid "InvalidRequestError" with joined eager loads
            result = await session.execute(stmt)
            result = result.unique()           # <--- fix here
            account = result.scalar_one_or_none()
            if not account:
                raise ValueError(f"No account found for user_id={user_id}")

            # Find a single Earnings hold with enough leftover
            suitable_hold = None
            for hold in account.holds:
                if hold.hold_type == HoldType.Earnings and not hold.charged:
                    leftover = hold.total_amount - hold.used_amount
                    if leftover >= amount:
                        suitable_hold = hold
                        break

            if not suitable_hold:
                raise ValueError("Insufficient earnings to request this withdrawal.")

            # Reserve the amount in that hold
            suitable_hold.used_amount += amount
            hold_txn = HoldTransaction(
                hold_id=suitable_hold.id,
                order_id=None,  # No 'order' here
                amount=amount
            )
            session.add(hold_txn)

            # Create new PendingWithdrawal
            new_withdrawal = PendingWithdrawal(
                account_id=account.id,
                user_id=user_id,
                amount=amount,
                status=WithdrawalStatus.PENDING,
            )
            session.add(new_withdrawal)

            await session.commit()
            await session.refresh(new_withdrawal)
            return new_withdrawal.id

    async def get_withdrawal(self, withdrawal_id: int) -> PendingWithdrawal | None:
        """
        Fetch a single PendingWithdrawal by ID.
        """
        async with self.get_async_session() as session:
            stmt = select(PendingWithdrawal).where(PendingWithdrawal.id == withdrawal_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_withdrawals_for_user(self, user_id: str) -> list[PendingWithdrawal]:
        """
        Return all withdrawals for a given user.
        """
        async with self.get_async_session() as session:
            stmt = select(PendingWithdrawal).where(PendingWithdrawal.user_id == user_id)
            result = await session.execute(stmt)
            return result.scalars().all()

    async def update_withdrawal_status(self, withdrawal_id: int, new_status: WithdrawalStatus):
        """
        For final 'approval' or 'rejection' steps.
        """
        async with self.get_async_session() as session:
            stmt = select(PendingWithdrawal).where(PendingWithdrawal.id == withdrawal_id)
            result = await session.execute(stmt)
            withdrawal = result.scalar_one_or_none()
            if not withdrawal:
                raise ValueError(f"Withdrawal {withdrawal_id} not found.")

            withdrawal.status = new_status
            withdrawal.updated_at = datetime.utcnow()
            session.add(withdrawal)
            await session.commit()

    async def complete_withdrawal_flow(self, withdrawal_id: int) -> bool:
        """
        Marks the withdrawal as FINALIZED, and charges the user's Earnings hold 
        for the requested amount. Typically called by an admin or payment system.
        """
        async with self.get_async_session() as session:
            stmt = select(PendingWithdrawal).where(PendingWithdrawal.id == withdrawal_id)
            result = await session.execute(stmt)
            withdrawal = result.scalar_one_or_none()
            if not withdrawal:
                raise ValueError(f"Withdrawal {withdrawal_id} not found.")

            if withdrawal.status != WithdrawalStatus.PENDING:
                raise ValueError(f"Withdrawal {withdrawal_id} must be PENDING before finalizing.")

            # Mark as FINALIZED
            withdrawal.status = WithdrawalStatus("FINALIZED")
            withdrawal.updated_at = datetime.utcnow()
            session.add(withdrawal)

            required = withdrawal.amount
            # fetch the account + holds with joinedload => use .unique()
            account_stmt = (
                select(Account)
                .where(Account.id == withdrawal.account_id)
                .options(joinedload(Account.holds))
            )
            acc_res = await session.execute(account_stmt)
            acc_res = acc_res.unique()         # <--- fix again
            account = acc_res.scalar_one_or_none()
            if not account:
                raise ValueError("Missing account for withdrawal?")

            found_hold = None
            for hold in account.holds:
                if hold.hold_type == HoldType.Earnings and not hold.charged:
                    if hold.used_amount >= required:
                        found_hold = hold
                        break
            if not found_hold:
                raise ValueError(
                    "Could not find an Earnings hold with enough used_amount to finalize this withdrawal."
                )

            # Charge that hold for 'required'
            await self.charge_hold_for_price(session, found_hold, required)
            await session.commit()
            return True
