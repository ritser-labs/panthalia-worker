# file: spl/db/server/adapter/accounts.py

import logging
from datetime import datetime
from sqlalchemy import update, delete, select
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from eth_account import Account as EthAccount
from datetime import timedelta

from ....models import (
    Account, CreditTransaction, CreditTxnType,
    EarningsTransaction, EarningsTxnType, PlatformRevenue, PlatformRevenueTxnType,
    AccountKey, WithdrawalRequest, WithdrawalStatus, StripeDeposit, Hold, HoldType,
    Task, Order, AccountTransaction
)
import uuid
from typing import Optional
from spl.auth.auth0_management import delete_auth0_user

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class DBAdapterAccountsMixin:
    MINIMUM_PAYOUT_AMOUNT = 50.0

    async def get_or_create_account(self, user_id: str, session: AsyncSession = None):
        own_session = False
        if session is None:
            session = self.get_async_session()
            own_session = True
        try:
            stmt = select(Account).where(Account.user_id == user_id)
            result = await session.execute(stmt)
            account = result.scalar_one_or_none()

            if account:
                return account

            new_account = Account(
                user_id=user_id,
                deposited_at=datetime.utcnow()
            )
            session.add(new_account)
            await session.commit()
            await session.refresh(new_account)
            return new_account
        finally:
            if own_session:
                await session.close()

    async def add_credits_transaction(self, session: AsyncSession, account: Account, amount: int, txn_type: CreditTxnType):
        """
        Records a new credit transaction in the DB for auditing, but does NOT adjust
        any account.credits_balance because that no longer exists.
        """
        new_credit_txn = CreditTransaction(
            account_id=account.id,
            user_id=account.user_id,
            amount=amount,
            txn_type=txn_type,
        )
        session.add(new_credit_txn)
        # The actual "balance" is derived from the uncharged leftover in holds.

    async def add_earnings_transaction(self, session: AsyncSession, account: Account, amount: int, txn_type: EarningsTxnType):
        """
        Records a new earnings transaction for auditing, does NOT mutate
        any ephemeral earnings_balance in the Account.
        """
        new_earnings_txn = EarningsTransaction(
            account_id=account.id,
            user_id=account.user_id,
            amount=amount,
            txn_type=txn_type,
        )
        session.add(new_earnings_txn)

    async def add_platform_revenue(self, session: AsyncSession, amount: int, txn_type: PlatformRevenueTxnType):
        new_rev = PlatformRevenue(
            amount=amount,
            txn_type=txn_type
        )
        session.add(new_rev)

    async def admin_deposit_account(self, user_id: str, amount: int):
        """
        Instead of touching any ephemeral account.credits_balance, we simply create a
        credit transaction and a hold for deposit-based credits, so the user
        has an uncharged leftover in the hold.
        """
        await self.create_credit_transaction_for_user(
            user_id=user_id,
            amount=amount,
            reason="1year_deposit",
            txn_type=CreditTxnType.Add
        )

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

        async with self.get_async_session() as session:
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
        async with self.get_async_session() as session:
            stmt = select(AccountKey).filter_by(public_key=public_key.lower())
            result = await session.execute(stmt)
            account_key = result.scalar_one_or_none()
            return account_key

    async def get_account_keys(self):
        user_id = self.get_user_id()
        async with self.get_async_session() as session:
            stmt = select(AccountKey).filter_by(user_id=user_id)
            result = await session.execute(stmt)
            account_keys = result.scalars().all()
            return account_keys

    async def delete_account_key(self, account_key_id: int):
        user_id = self.get_user_id()
        async with self.get_async_session() as session:
            stmt = select(AccountKey).filter_by(id=account_key_id)
            result = await session.execute(stmt)
            account_key = result.scalar_one_or_none()
            if not account_key or account_key.user_id != user_id:
                raise PermissionError("No access to delete this account key.")
            await session.delete(account_key)
            await session.commit()

    async def create_credit_transaction_for_user(
        self,
        user_id: str,
        amount: int,
        reason: str,
        txn_type: CreditTxnType,
        session: Optional[AsyncSession] = None
    ) -> CreditTransaction:
        """
        Creates a credit transaction (Add or Subtract) for the user. If txn_type=Add,
        also creates a new deposit-based hold so that leftover in that hold is effectively
        their "credits" balance.

        If `session` is provided, it reuses that session and does NOT commit/close inside
        this function. Otherwise, it creates its own session context, commits, and closes.
        """
        # If caller did not pass an existing session, open our own context
        if session is None:
            async with self.get_async_session() as new_session:
                result = await self._create_credit_transaction_for_user_internal(
                    new_session, user_id, amount, reason, txn_type
                )
                await new_session.commit()
                return result
        else:
            # Use caller's session
            return await self._create_credit_transaction_for_user_internal(
                session, user_id, amount, reason, txn_type
            )

    async def _create_credit_transaction_for_user_internal(
        self,
        session: AsyncSession,
        user_id: str,
        amount: int,
        reason: str,
        txn_type: CreditTxnType
    ) -> CreditTransaction:
        account = await self.get_or_create_account(user_id, session=session)

        # If we're adding credits, create a new deposit-based hold
        if txn_type == CreditTxnType.Add:
            expiry_date = datetime.utcnow() + timedelta(days=365)
            new_hold = Hold(
                account_id=account.id,
                user_id=user_id,
                hold_type=HoldType.Credits,
                total_amount=amount,
                used_amount=0.0,
                expiry=expiry_date,
                charged=False,
                charged_amount=0.0
            )
            session.add(new_hold)
            logger.info(
                f"[create_credit_transaction_for_user] user={user_id}, +{amount} credits => "
                f"hold with expiry {expiry_date}"
            )

        # Insert the credit transaction
        new_tx = CreditTransaction(
            account_id=account.id,
            user_id=user_id,
            amount=amount,
            txn_type=txn_type,
            reason=reason
        )
        session.add(new_tx)

        logger.info(
            f"[create_credit_transaction_for_user] user={user_id}, created txn => reason={reason}, "
            f"txn_type={txn_type}, amount={amount}"
        )
        return new_tx

    async def link_deposit_to_transaction(self, deposit_id: int, credit_tx_id: int) -> StripeDeposit:
        async with self.get_async_session() as session:
            stmt = select(StripeDeposit).where(StripeDeposit.id == deposit_id)
            result = await session.execute(stmt)
            dep_obj = result.scalar_one_or_none()
            if not dep_obj:
                raise ValueError(f"No StripeDeposit found with id={deposit_id}")

            dep_obj.credit_transaction_id = credit_tx_id
            await session.commit()
            await session.refresh(dep_obj)
            logger.info(f"[link_deposit_to_transaction] deposit {deposit_id} linked to credit_tx {credit_tx_id}")
            return dep_obj

    async def delete_account(self):
        """
        Anonymizes the current user’s account and associated personal data.
        This implementation “deletes” the account by replacing the user's identifier
        with an anonymized value and removing any account keys. Other tables that hold
        the user_id (such as tasks, orders, transactions, withdrawals, holds) are updated
        so that the personal identifier is no longer stored.

        Additionally, it calls the Auth0 Management API to delete the Auth0 user.
        """
        # Get the current user's original id (assumed to be the Auth0 user id, e.g. "auth0|...")
        user_id = self.get_user_id()

        async with self.get_async_session() as session:
            # Fetch the account record for this user
            stmt = select(Account).where(Account.user_id == user_id)
            result = await session.execute(stmt)
            account = result.scalar_one_or_none()
            if not account:
                raise ValueError("Account not found.")

            # Preserve the original user_id (to use for the Auth0 deletion call)
            original_user_id = account.user_id

            # Generate an anonymized user_id—for example, "deleted_<account.id>_<uuid>"
            new_user_id = f"deleted_{account.id}_{uuid.uuid4().hex}"
            account.user_id = new_user_id  # update the account record

            # Update all other tables that store the personal user_id.
            tables = [
                Order,
                AccountTransaction,
                CreditTransaction,
                EarningsTransaction,
                WithdrawalRequest,
                Hold
            ]
            for table in tables:
                upd = (
                    update(table)
                    .where(table.user_id == original_user_id)
                    .values(user_id=new_user_id)
                )
                await session.execute(upd)

            # Delete any account keys for this user (since these contain cryptographic material)
            del_stmt = delete(AccountKey).where(AccountKey.user_id == original_user_id)
            await session.execute(del_stmt)

            await session.commit()

        # Now delete the user from Auth0 using your management API helper.
        try:
            delete_auth0_user(original_user_id)
            logger.info(f"Auth0 user {original_user_id} deleted successfully.")
        except Exception as e:
            logger.error(f"Failed to delete Auth0 user {original_user_id}: {e}")
            # Depending on your application's policy, you might choose to re-raise here.

        return True
