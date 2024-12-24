# spl/db/server/adapter/accounts.py

import logging
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from eth_account import Account as EthAccount

from ....models import (
    AsyncSessionLocal, Account, CreditTransaction, CreditTxnType,
    EarningsTransaction, EarningsTxnType, PlatformRevenue, PlatformRevenueTxnType,
    AccountKey, PendingWithdrawal, WithdrawalStatus
)

logger = logging.getLogger(__name__)

class DBAdapterAccountsMixin:
    ##
    # We'll define a minimum amount needed for any withdrawal
    ##
    MINIMUM_PAYOUT_AMOUNT = 50.0

    async def get_or_create_account(self, user_id: str, session: AsyncSession = None):
        own_session = False
        if session is None:
            session = AsyncSessionLocal()
            own_session = True
        try:
            stmt = select(Account).where(Account.user_id == user_id)
            result = await session.execute(stmt)
            account = result.scalar_one_or_none()

            if account:
                return account

            new_account = Account(
                user_id=user_id,
                credits_balance=0.0,
                earnings_balance=0.0,
                deposited_at=datetime.utcnow()
            )
            session.add(new_account)
            await session.commit()
            await session.refresh(new_account)
            return new_account
        finally:
            if own_session:
                await session.close()

    async def add_credits_transaction(self, session: AsyncSession, account: Account, amount: float, txn_type: CreditTxnType):
        if txn_type == CreditTxnType.Subtract:
            if account.credits_balance < amount:
                raise ValueError("insufficient credits to subtract")
            account.credits_balance -= amount
        elif txn_type == CreditTxnType.Add:
            account.credits_balance += amount

        new_credit_txn = CreditTransaction(
            account_id=account.id,
            user_id=account.user_id,
            amount=amount,
            txn_type=txn_type,
        )
        session.add(new_credit_txn)

    async def add_earnings_transaction(self, session: AsyncSession, account: Account, amount: float, txn_type: EarningsTxnType):
        if txn_type == EarningsTxnType.Subtract:
            if account.earnings_balance < amount:
                raise ValueError("not enough earnings to subtract")
            account.earnings_balance -= amount
        else:
            account.earnings_balance += amount

        new_earnings_txn = EarningsTransaction(
            account_id=account.id,
            user_id=account.user_id,
            amount=amount,
            txn_type=txn_type,
        )
        session.add(new_earnings_txn)

    async def add_platform_revenue(self, session: AsyncSession, amount: float, txn_type: PlatformRevenueTxnType):
        new_rev = PlatformRevenue(
            amount=amount,
            txn_type=txn_type
        )
        session.add(new_rev)

    ##
    # REMOVED maybe_payout_earnings
    # (This system is replaced by create_withdrawal_request.)
    ##

    async def admin_deposit_account(self, user_id: str, amount: float):
        async with AsyncSessionLocal() as session:
            account = await self.get_or_create_account(user_id, session)
            await self.add_credits_transaction(session, account, amount, CreditTxnType.Add)
            await session.commit()

    # Account Key Methods
    async def admin_create_account_key(self, user_id: str):
        return await self._create_account_key(user_id)

    async def create_account_key(self):
        user_id = self.get_user_id()
        return await self._create_account_key(user_id)

    async def _create_account_key(self, user_id: str):
        account = EthAccount.create()
        private_key = account.key.hex()
        public_key = account.address.lower()

        async with AsyncSessionLocal() as session:
            new_account_key = AccountKey(
                user_id=user_id,
                public_key=public_key
            )
            session.add(new_account_key)
            await session.commit()
            await session.refresh(new_account_key)

        return {
            "private_key": private_key,
            "public_key": public_key,
            "account_key_id": new_account_key.id
        }

    async def account_key_from_public_key(self, public_key: str):
        async with AsyncSessionLocal() as session:
            stmt = select(AccountKey).filter_by(public_key=public_key.lower())
            result = await session.execute(stmt)
            account_key = result.scalar_one_or_none()
            return account_key

    async def get_account_keys(self):
        user_id = self.get_user_id()
        async with AsyncSessionLocal() as session:
            stmt = select(AccountKey).filter_by(user_id=user_id)
            result = await session.execute(stmt)
            account_keys = result.scalars().all()
            return account_keys

    async def delete_account_key(self, account_key_id: int):
        user_id = self.get_user_id()
        async with AsyncSessionLocal() as session:
            stmt = select(AccountKey).filter_by(id=account_key_id)
            result = await session.execute(stmt)
            account_key = result.scalar_one_or_none()
            if not account_key or account_key.user_id != user_id:
                raise PermissionError("No access to delete this account key.")
            await session.delete(account_key)
            await session.commit()

    ##
    # NEW WITHDRAWALS:
    ##
    async def create_withdrawal_request(self, user_id: str, amount: float) -> int:
        """
        Creates a new PendingWithdrawal for the given user with the requested amount,
        only subtracting from earnings_balance.
        Also enforces MINIMUM_PAYOUT_AMOUNT.
        """
        if amount < self.MINIMUM_PAYOUT_AMOUNT:
            raise ValueError(f"Cannot withdraw less than the minimum {self.MINIMUM_PAYOUT_AMOUNT}")

        async with AsyncSessionLocal() as session:
            account = await self.get_or_create_account(user_id, session)
            # Only subtract from earnings_balance:
            if account.earnings_balance < amount:
                raise ValueError("insufficient earnings to request withdrawal")

            account.earnings_balance -= amount

            new_withdrawal = PendingWithdrawal(
                account_id=account.id,
                user_id=user_id,
                amount=amount,
            )
            session.add(new_withdrawal)

            await session.commit()
            await session.refresh(new_withdrawal)
            return new_withdrawal.id

    async def get_withdrawal(self, withdrawal_id: int) -> PendingWithdrawal | None:
        async with AsyncSessionLocal() as session:
            stmt = select(PendingWithdrawal).where(PendingWithdrawal.id == withdrawal_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_withdrawals_for_user(self, user_id: str) -> list[PendingWithdrawal]:
        async with AsyncSessionLocal() as session:
            stmt = select(PendingWithdrawal).where(PendingWithdrawal.user_id == user_id)
            result = await session.execute(stmt)
            return result.scalars().all()

    async def update_withdrawal_status(self, withdrawal_id: int, new_status: WithdrawalStatus):
        """
        Approve or reject a pending withdrawal. If rejected, refund user. If approved, do nothing else here
        (you might do a real payment or queue it).
        """
        async with AsyncSessionLocal() as session:
            stmt = select(PendingWithdrawal).where(PendingWithdrawal.id == withdrawal_id)
            result = await session.execute(stmt)
            withdrawal = result.scalar_one_or_none()
            if not withdrawal:
                raise ValueError(f"Withdrawal {withdrawal_id} not found.")

            if new_status == WithdrawalStatus.REJECTED and withdrawal.status == WithdrawalStatus.PENDING:
                # Return the funds to user (earnings_balance).
                acct_stmt = select(Account).where(Account.id == withdrawal.account_id)
                acct_result = await session.execute(acct_stmt)
                account = acct_result.scalar_one_or_none()
                if account:
                    account.earnings_balance += withdrawal.amount

            withdrawal.status = new_status
            withdrawal.updated_at = datetime.utcnow()
            session.add(withdrawal)
            await session.commit()
