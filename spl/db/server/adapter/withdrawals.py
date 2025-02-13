# file: spl/db/server/adapter/withdrawals.py

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from datetime import datetime
from ....models.enums import WithdrawalStatus, HoldType
from ....models import WithdrawalRequest, Account, Hold, HoldTransaction

class DBAdapterWithdrawalsMixin:
    """
    Mixin for handling user withdrawals from their Earnings balance.
    """

    async def create_withdrawal_request(self, amount: int, payment_instructions: str) -> int:
        """
        Creates a new WithdrawalRequest in status=PENDING. Also 'reserves' that amount
        in an Earnings hold (so the user cannot double-spend those same earnings).

        Raises ValueError if there's no single Earnings hold with enough leftover.
        """
        if amount <= 0:
            raise ValueError("Cannot withdraw a non-positive amount.")

        async with self.get_async_session() as session:
            # Eager-load the user's Account plus its holds:
            user_id = self.get_user_id()
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

            # Create new WithdrawalRequest
            new_withdrawal = WithdrawalRequest(
                account_id=account.id,
                user_id=user_id,
                amount=amount,
                status=WithdrawalStatus.PENDING,
                payment_instructions=payment_instructions
            )
            session.add(new_withdrawal)

            await session.commit()
            await session.refresh(new_withdrawal)
            return new_withdrawal.id

    async def get_withdrawal(self, withdrawal_id: int) -> WithdrawalRequest | None:
        """
        Fetch a single WithdrawalRequest by ID.
        """
        async with self.get_async_session() as session:
            stmt = select(WithdrawalRequest).where(WithdrawalRequest.id == withdrawal_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_withdrawals_for_user(self, offset: int = 0, limit: int = 20) -> list[WithdrawalRequest]:
        """
        Return a paginated list of withdrawals for the current user.
        """
        async with self.get_async_session() as session:
            user_id = self.get_user_id()
            stmt = (
                select(WithdrawalRequest)
                .where(WithdrawalRequest.user_id == user_id)
                .offset(offset)
                .limit(limit)
            )
            result = await session.execute(stmt)
            return result.scalars().all()


    async def complete_withdrawal_flow(self, withdrawal_id: int, payment_record: str) -> bool:
        """
        Marks the withdrawal as FINALIZED, and charges the user's Earnings hold 
        for the requested amount. Typically called by an admin or payment system.
        """
        async with self.get_async_session() as session:
            stmt = select(WithdrawalRequest).where(WithdrawalRequest.id == withdrawal_id)
            result = await session.execute(stmt)
            withdrawal = result.scalar_one_or_none()
            if not withdrawal:
                raise ValueError(f"Withdrawal {withdrawal_id} not found.")

            if withdrawal.status != WithdrawalStatus.PENDING:
                raise ValueError(f"Withdrawal {withdrawal_id} must be PENDING before finalizing.")

            # Mark as FINALIZED
            withdrawal.status = WithdrawalStatus.FINALIZED
            withdrawal.updated_at = datetime.utcnow()
            withdrawal.payment_record = payment_record
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

    async def reject_withdrawal_flow(self, withdrawal_id: int, rejection_reason: str) -> bool:
        """
        Marks the withdrawal as REJECTED, and 'unreserves' the user's Earnings hold
        by decrementing used_amount by the withdrawal amount.

        Typically called by an admin or payment system when a withdrawal is declined.
        """
        async with self.get_async_session() as session:
            # 1) Fetch the WithdrawalRequest
            stmt = select(WithdrawalRequest).where(WithdrawalRequest.id == withdrawal_id)
            result = await session.execute(stmt)
            withdrawal = result.scalar_one_or_none()
            if not withdrawal:
                raise ValueError(f"Withdrawal {withdrawal_id} not found.")

            if withdrawal.status != WithdrawalStatus.PENDING:
                raise ValueError(f"Withdrawal {withdrawal_id} must be PENDING before rejection.")

            # 2) Mark as REJECTED
            withdrawal.status = WithdrawalStatus.REJECTED
            withdrawal.updated_at = datetime.utcnow()
            withdrawal.rejection_reason = rejection_reason
            session.add(withdrawal)

            # 3) Unreserve the previously used amount from the user’s Earnings hold
            required = withdrawal.amount
            account_stmt = (
                select(Account)
                .where(Account.id == withdrawal.account_id)
                .options(joinedload(Account.holds))
            )
            acc_res = await session.execute(account_stmt)
            acc_res = acc_res.unique()
            account = acc_res.scalar_one_or_none()
            if not account:
                raise ValueError(f"Missing account for withdrawal {withdrawal_id}?")

            # find the Earnings hold that was used by create_withdrawal_request
            found_hold = None
            for hold in account.holds:
                if hold.hold_type == HoldType.Earnings and not hold.charged:
                    # Because create_withdrawal_request increments used_amount by 'withdrawal.amount'
                    # we only need to find a hold with enough used_amount that covers this withdrawal
                    if hold.used_amount >= required:
                        found_hold = hold
                        break
            if not found_hold:
                raise ValueError(
                    "Could not find an Earnings hold with enough used_amount to unreserve for this withdrawal."
                )

            found_hold.used_amount -= required  # unreserve
            if found_hold.used_amount < 0:
                raise ValueError(
                    f"Earnings hold {found_hold.id} used_amount became negative after rejection. Data mismatch!"
                )

            # Insert a negative hold transaction to reflect the unreserve
            txn = HoldTransaction(
                hold_id=found_hold.id,
                order_id=None,
                amount=-required
            )
            session.add(txn)

            await session.commit()
            return True

    async def get_withdrawals(self, status: str = "all", offset: int = 0, limit: int = 20) -> list[WithdrawalRequest]:
        """
        Return a paginated list of withdrawals filtered by status.
        If status is "all", no filtering is applied.
        """
        from ....models.enums import WithdrawalStatus
        async with self.get_async_session() as session:
            stmt = select(WithdrawalRequest)
            if status.lower() != "all":
                try:
                    # Convert the provided status to the enum (expects values like "pending", "finalized", or "rejected")
                    status_enum = WithdrawalStatus(status.lower())
                except ValueError:
                    raise ValueError("Invalid withdrawal status provided. Allowed values: pending, finalized, rejected, or 'all'.")
                stmt = stmt.where(WithdrawalRequest.status == status_enum)
            stmt = stmt.offset(offset).limit(limit)
            result = await session.execute(stmt)
            return result.scalars().all()
