# spl/db/server/adapter/accounts.py

import logging
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from eth_account import Account as EthAccount
from datetime import timedelta

from ....models import (
    AsyncSessionLocal, Account, CreditTransaction, CreditTxnType,
    EarningsTransaction, EarningsTxnType, PlatformRevenue, PlatformRevenueTxnType,
    AccountKey, PendingWithdrawal, WithdrawalStatus, StripeDeposit, Hold, HoldType
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
        """
        Instead of duplicating logic, just call create_credit_transaction_for_user.
        Mark the reason="1year_deposit" so we know it's deposit-based.
        """
        # You might want to do a quick check that amount > 0, etc.
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

    async def create_stripe_deposit(self, user_id: str, deposit_amount: float, session_id: str) -> int:
        """
        Insert a row in stripe_deposits with status='pending'.
        Returns the deposit ID.
        """
        async with AsyncSessionLocal() as session:
            new_dep = StripeDeposit(
                user_id=user_id,
                deposit_amount=deposit_amount,
                stripe_session_id=session_id,
                status='pending'
            )
            session.add(new_dep)
            await session.commit()
            await session.refresh(new_dep)
            logger.info(f"[create_stripe_deposit] Created deposit id={new_dep.id} for user={user_id}, amt={deposit_amount}")
            return new_dep.id

    async def mark_stripe_deposit_completed(self, stripe_session_id: str) -> StripeDeposit | None:
        """
        Mark the deposit row as 'completed' if it's pending.
        Returns the StripeDeposit object or None if not found.
        If already completed, returns the deposit object unchanged.
        """
        async with AsyncSessionLocal() as session:
            stmt = select(StripeDeposit).where(StripeDeposit.stripe_session_id == stripe_session_id)
            result = await session.execute(stmt)
            dep_obj = result.scalar_one_or_none()
            if not dep_obj:
                return None

            if dep_obj.status == 'completed':
                # Already done
                return dep_obj

            dep_obj.status = 'completed'
            await session.commit()
            await session.refresh(dep_obj)
            logger.info(f"[mark_stripe_deposit_completed] Deposit {dep_obj.id} marked completed.")
            return dep_obj

    async def create_credit_transaction_for_user(
        self,
        user_id: str,
        amount: float,
        reason: str,
        txn_type: CreditTxnType
    ) -> CreditTransaction:
        """
        Creates a credit transaction (Add or Subtract) for the given user, updates the user's
        credits_balance, and if txn_type=Add, also creates a new hold that expires in 1 year.
        """
        async with AsyncSessionLocal() as session:
            # fetch or create the account row
            account = await self.get_or_create_account(user_id, session)

            if txn_type == CreditTxnType.Add:
                # Increase balance
                account.credits_balance += amount

                # Also create a hold that expires in 1 year
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

                # Log for debugging; 'new_hold.id' may still be None until session.flush() or commit
                logging.info(
                    f"[create_credit_transaction_for_user] user={user_id}, +{amount} credits. "
                    f"Created 1-year hold (will expire {expiry_date})"
                )

            elif txn_type == CreditTxnType.Subtract:
                if account.credits_balance < amount:
                    raise ValueError(
                        f"Insufficient balance to subtract {amount} for user {user_id}"
                    )
                account.credits_balance -= amount

            # **FIX**: Set 'account_id' to avoid null constraint error
            new_tx = CreditTransaction(
                account_id=account.id,    # <-- CRITICAL
                user_id=user_id,
                amount=amount,
                txn_type=txn_type,
                reason=reason
            )
            session.add(new_tx)

            await session.commit()
            await session.refresh(new_tx)

            logging.info(
                f"[create_credit_transaction_for_user] user={user_id}, txn={new_tx.id} "
                f"({txn_type.name} {amount}) => success."
            )

            return new_tx


    async def link_deposit_to_transaction(self, deposit_id: int, credit_tx_id: int) -> StripeDeposit:
        """
        Store credit_transaction_id in the StripeDeposit row.
        Returns the updated deposit object.
        """
        async with AsyncSessionLocal() as session:
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

    async def get_balance(self) -> dict:
        """
        Returns a dict with credits_balance and earnings_balance
        for the given user_id, or an error.
        """
        user_id = self.get_user_id()
        account = await self.get_or_create_account(user_id)
        if not account:
            # Return a dict containing an "error" key if no account
            return {'error': f"No account found for user_id={user_id}"}
        return {
            'credits_balance': account.credits_balance,
            'earnings_balance': account.earnings_balance
        }