import logging
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from eth_account import Account as EthAccount

from ....models import (
    AsyncSessionLocal, Account, CreditTransaction, CreditTxnType,
    EarningsTransaction, EarningsTxnType, PlatformRevenue, PlatformRevenueTxnType,
    AccountKey
)

logger = logging.getLogger(__name__)

class DBAdapterAccountsMixin:
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

    async def maybe_payout_earnings(self, session: AsyncSession, account: Account):
        MINIMUM_PAYOUT_AMOUNT = 50.0
        if account.earnings_balance >= MINIMUM_PAYOUT_AMOUNT:
            payout_amount = account.earnings_balance
            account.earnings_balance = 0.0
            logger.debug(f"Payout {payout_amount} to {account.user_id}")

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
