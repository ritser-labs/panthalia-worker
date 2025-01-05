# spl/db/server/adapter/accounts.py

import logging
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from eth_account import Account as EthAccount
from datetime import timedelta

from ....models import (
    Account, CreditTransaction, CreditTxnType,
    EarningsTransaction, EarningsTxnType, PlatformRevenue, PlatformRevenueTxnType,
    AccountKey, PendingWithdrawal, WithdrawalStatus, StripeDeposit, Hold, HoldType
)
from typing import Optional

logger = logging.getLogger(__name__)

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

    async def create_withdrawal_request(self, user_id: str, amount: int) -> int:
        """
        Creates a new PendingWithdrawal. The actual "balance" check is done
        by summing leftover in Earnings holds, so if there's insufficient leftover,
        we raise an error.
        """
        if amount < self.MINIMUM_PAYOUT_AMOUNT:
            raise ValueError(f"Cannot withdraw less than the minimum {self.MINIMUM_PAYOUT_AMOUNT}")

        async with self.get_async_session() as session:
            account = await self.get_or_create_account(user_id, session)
            # Instead of subtracting from account.earnings_balance, we check if
            # there's enough leftover in "Earnings" holds.
            # If not enough leftover => raise.
            total_earnings_leftover = 0.0
            for hold in account.holds:
                if hold.hold_type == HoldType.Earnings and not hold.charged:
                    leftover = hold.total_amount - hold.used_amount
                    if leftover > 0:
                        total_earnings_leftover += leftover

            if total_earnings_leftover < amount:
                raise ValueError("insufficient earnings to request withdrawal")

            # We do not physically subtract from a column. It's enough that we
            # create a PendingWithdrawal. The future code that approves or rejects
            # can forcibly charge or free from an Earnings hold, if desired.
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
        async with self.get_async_session() as session:
            stmt = select(PendingWithdrawal).where(PendingWithdrawal.id == withdrawal_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_withdrawals_for_user(self, user_id: str) -> list[PendingWithdrawal]:
        async with self.get_async_session() as session:
            stmt = select(PendingWithdrawal).where(PendingWithdrawal.user_id == user_id)
            result = await session.execute(stmt)
            return result.scalars().all()

    async def update_withdrawal_status(self, withdrawal_id: int, new_status: WithdrawalStatus):
        """
        If rejected, we simply do not finalize the withdrawal. 
        If approved, you might do your real payment or external queue, etc.
        """
        async with self.get_async_session() as session:
            stmt = select(PendingWithdrawal).where(PendingWithdrawal.id == withdrawal_id)
            result = await session.execute(stmt)
            withdrawal = result.scalar_one_or_none()
            if not withdrawal:
                raise ValueError(f"Withdrawal {withdrawal_id} not found.")

            if new_status == WithdrawalStatus.REJECTED and withdrawal.status == WithdrawalStatus.PENDING:
                # We do not "refund" any column-based balance, because we only rely on hold leftover for logic.
                pass

            withdrawal.status = new_status
            withdrawal.updated_at = datetime.utcnow()
            session.add(withdrawal)
            await session.commit()

    async def create_stripe_deposit(self, user_id: str, deposit_amount: int, session_id: str) -> int:
        async with self.get_async_session() as session:
            new_dep = StripeDeposit(
                user_id=user_id,
                deposit_amount=deposit_amount,
                stripe_session_id=session_id,
                status='pending'
            )
            session.add(new_dep)
            await self.create_credit_transaction_for_user(
                user_id=user_id,
                amount=deposit_amount,
                reason="1year_deposit",
                txn_type=CreditTxnType.Add,
                session=session
            )
            await session.commit()
            await session.refresh(new_dep)
            logger.info(f"[create_stripe_deposit] Created deposit id={new_dep.id} for user={user_id}, amt={deposit_amount}")
            return new_dep.id

    async def mark_stripe_deposit_completed(self, stripe_session_id: str) -> StripeDeposit | None:
        async with self.get_async_session() as session:
            stmt = select(StripeDeposit).where(StripeDeposit.stripe_session_id == stripe_session_id)
            result = await session.execute(stmt)
            dep_obj = result.scalar_one_or_none()
            if not dep_obj:
                return None

            if dep_obj.status == 'completed':
                return dep_obj

            dep_obj.status = 'completed'
            await session.commit()
            await session.refresh(dep_obj)
            logger.info(f"[mark_stripe_deposit_completed] Deposit {dep_obj.id} marked completed.")
            return dep_obj

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
                return await self._create_credit_transaction_for_user_internal(
                    new_session, user_id, amount, reason, txn_type
                )
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
        """
        Internal helper that assumes an already-open session. This is where the actual
        DB logic happens. The caller handles commit if needed.
        """
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

        # Note: We don't do `await session.commit()` here because the caller
        # (either our own `async with` block or the external caller) will handle it.

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
