# file: spl/db/server/adapter/billing/stripe.py

import os
import logging
import stripe
from datetime import datetime, timedelta
from sqlalchemy import select, func, asc
from typing import Optional

from .....models.enums import CreditTxnType, HoldType, CENT_AMOUNT
from .....models import StripeDeposit, Account, Hold

# The concurrency limit for pending Stripe checkouts
MAX_CONCURRENT_STRIPE_SESSIONS = 3

# If an account's `max_credits_balance` is null, use this fallback default
FALLBACK_DEPOSIT_LIMIT = 1000 * 100 * CENT_AMOUNT  # e.g. $1000

# We keep the same minimum checkout logic from your original code:
MINIMUM_CHECKOUT_AMOUNT = 10 * 100 * CENT_AMOUNT  # e.g. $10 minimum, if 1 cent == 1e6

logger = logging.getLogger(__name__)

class DBAdapterStripeBillingMixin:
    """
    A mixin that adds stripe-billing-related methods, updated to handle:
      1) A user-specific deposit limit from account.max_credits_balance
      2) Concurrency-limiting for pending sessions
      3) Final deposit-limit check in webhook
      4) Automatic expiration of stale sessions
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

        self.stripe_api_key = os.environ.get("STRIPE_SECRET_KEY", "")
        self.webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
        self.success_url = os.environ.get("STRIPE_SUCCESS_URL", "https://panthalia.com/success")
        self.cancel_url = os.environ.get("STRIPE_CANCEL_URL", "https://panthalia.com/cancel")

        stripe.api_key = self.stripe_api_key

    # -------------------------------------------------------------------------
    # 1) CREATE A STRIPE CREDITS SESSION => concurrency & deposit-limit check
    # -------------------------------------------------------------------------
    async def create_stripe_credits_session(self, amount: int) -> dict:
        """
        Creates a Stripe Checkout Session for a deposit, ensuring:
          - We prune oldest pending sessions if user is over concurrency
          - We do a best-effort deposit-limit check using account.max_credits_balance
        """
        user_id = self.get_user_id()

        # (A) Minimum deposit
        if amount < MINIMUM_CHECKOUT_AMOUNT:
            return {
                "error": f"Minimum checkout amount is {MINIMUM_CHECKOUT_AMOUNT} (internal units)",
                "status_code": 400
            }
        if amount <= 0:
            return {"error": "Invalid amount", "status_code": 400}

        # (B) Prune older sessions if concurrency is exceeded
        pruned_ok = await self._prune_pending_sessions_if_needed(user_id)
        if not pruned_ok:
            return {"error": "Failed to prune older sessions", "status_code": 500}

        # (C) Best-effort deposit-limit check
        leftover_credits = await self._get_user_credits_leftover(user_id)
        if leftover_credits is None:
            return {"error": "No account found for user", "status_code": 400}

        deposit_limit = await self._get_user_deposit_limit(user_id)
        if deposit_limit is None:
            return {"error": "No account deposit limit found", "status_code": 400}

        # sum up pending
        total_pending = await self._get_pending_deposit_sum_for_user(user_id)
        if leftover_credits + total_pending + amount > deposit_limit:
            return {
                "error": (
                    f"Deposit limit exceeded. leftover={leftover_credits}, "
                    f"pending={total_pending}, request={amount}, limit={deposit_limit}"
                ),
                "status_code": 400
            }

        # (D) Create the Stripe Checkout Session
        try:
            amount_in_cents = int(amount / CENT_AMOUNT)
            session_obj = stripe.checkout.Session.create(
                payment_method_types=["card"],
                mode="payment",
                line_items=[{
                    "price_data": {
                        "currency": "usd",
                        "product_data": {"name": "Panthalia Credits"},
                        "unit_amount": amount_in_cents,
                    },
                    "quantity": 1,
                }],
                success_url=self.success_url,
                cancel_url=self.cancel_url,
            )
        except Exception as e:
            self.logger.error(f"[create_stripe_credits_session] error from Stripe => {e}")
            return {"error": str(e), "status_code": 500}

        # (E) Insert a new StripeDeposit row => status='pending'
        try:
            await self.create_stripe_deposit(user_id, amount, session_obj.id)
        except Exception as e:
            self.logger.error(f"[create_stripe_credits_session] DB error => {e}")
            return {"error": str(e), "status_code": 500}

        return {"session_id": session_obj.id, "url": session_obj.url}

    # -------------------------------------------------------------------------
    # PRUNE OLDEST PENDING SESSIONS if user is over concurrency limit
    # -------------------------------------------------------------------------
    async def _prune_pending_sessions_if_needed(self, user_id: str) -> bool:
        stmt_count = select(func.count(StripeDeposit.id)).where(
            StripeDeposit.user_id == user_id,
            StripeDeposit.status == 'pending'
        )
        async with self.get_async_session() as session:
            res = await session.execute(stmt_count)
            pending_count = res.scalar() or 0

            if pending_count < MAX_CONCURRENT_STRIPE_SESSIONS:
                return True

            # free up slots so user can create a new one
            to_remove = pending_count - (MAX_CONCURRENT_STRIPE_SESSIONS - 1)
            if to_remove <= 0:
                return True

            # oldest first
            stmt_oldest = (
                select(StripeDeposit)
                .where(
                    StripeDeposit.user_id == user_id,
                    StripeDeposit.status == 'pending'
                )
                .order_by(StripeDeposit.created_at.asc())
                .limit(to_remove)
            )
            dep_res = await session.execute(stmt_oldest)
            old_list = dep_res.scalars().all()

            for dep_obj in old_list:
                sess_id = dep_obj.stripe_session_id
                try:
                    stripe.checkout.Session.expire(sess_id)
                except Exception as ex:
                    self.logger.warning(f"_prune_pending_sessions_if_needed => cannot expire sess={sess_id}: {ex}")
                dep_obj.status = 'cancelled'
                session.add(dep_obj)
            await session.commit()
        return True

    # -------------------------------------------------------------------------
    # 2) STRIPE WEBHOOK => final deposit-limit check
    # -------------------------------------------------------------------------
    async def handle_stripe_webhook(self, payload: bytes, sig_header: str) -> dict:
        """
        If type='checkout.session.completed', we do a final deposit-limit check using
        account.max_credits_balance. If it’s truly over limit, we cancel the deposit.
        Otherwise we finalize => apply_stripe_deposit.
        """
        if not self.webhook_secret:
            return {"error": "No webhook secret set", "status_code": 500}

        try:
            event = stripe.Webhook.construct_event(payload, sig_header, self.webhook_secret)
        except stripe.error.SignatureVerificationError:
            return {"error": "Invalid signature", "status_code": 400}
        except Exception as e:
            self.logger.error(f"[handle_stripe_webhook] parse error => {e}")
            return {"error": str(e), "status_code": 400}

        if event["type"] == "checkout.session.completed":
            session_obj = event["data"]["object"]
            stripe_session_id = session_obj["id"]

            deposit_obj = await self.mark_stripe_deposit_completed(stripe_session_id)
            if not deposit_obj:
                self.logger.warning(f"[handle_stripe_webhook] no deposit found for sess={stripe_session_id}")
                return {"status": "ok"}

            if deposit_obj.status == 'completed' and deposit_obj.credit_transaction_id:
                # Already credited => do nothing
                return {"status": "ok"}

            if deposit_obj.is_authorization:
                # For authorize-only => create a credit-card hold
                await self.apply_credit_card_authorization(deposit_obj)
                return {"status": "ok"}

            # normal deposit => final deposit-limit check
            user_id = deposit_obj.user_id
            leftover_credits = await self._get_user_credits_leftover(user_id)
            deposit_limit = await self._get_user_deposit_limit(user_id)
            if leftover_credits is None or deposit_limit is None:
                # fallback => just cancel deposit if we can't read the data
                self.logger.warning(f"[handle_stripe_webhook] cannot read leftover or deposit_limit => cancel deposit")
                await self._mark_stripe_deposit_cancelled(deposit_obj)
                return {"status": "ok"}

            # exclude this session from 'pending' sum
            sum_pending = await self._get_pending_deposit_sum_for_user(
                user_id, exclude_session_id=stripe_session_id
            )
            if leftover_credits + sum_pending + deposit_obj.deposit_amount > deposit_limit:
                # Over the user’s limit => cancel deposit
                self.logger.warning(
                    f"[handle_stripe_webhook] deposit {deposit_obj.id} would exceed limit => canceling"
                )
                await self._mark_stripe_deposit_cancelled(deposit_obj)
                return {"status": "ok"}
            else:
                # OK => finalize deposit => apply
                await self.apply_stripe_deposit(deposit_obj)

        return {"status": "ok"}

    # Cancel deposit => status='cancelled'
    async def _mark_stripe_deposit_cancelled(self, deposit_obj: StripeDeposit):
        async with self.get_async_session() as session:
            deposit_obj.status = 'cancelled'
            session.add(deposit_obj)
            await session.commit()
        self.logger.info(f"[mark_stripe_deposit_cancelled] deposit {deposit_obj.id} => cancelled")

    # -------------------------------------------------------------------------
    # 3) EXPIRE OLD 'pending' sessions from background
    # -------------------------------------------------------------------------
    async def expire_old_stripe_deposits(self, older_than_minutes: int = 60):
        """
        Called from background tasks => for each 'pending' deposit older than X minutes,
        call stripe.checkout.Session.expire(...) then set status='cancelled' in DB.
        """
        cutoff_time = datetime.utcnow() - timedelta(minutes=older_than_minutes)

        async with self.get_async_session() as session:
            stmt = select(StripeDeposit).where(
                StripeDeposit.status == 'pending',
                StripeDeposit.created_at < cutoff_time
            )
            result = await session.execute(stmt)
            old_deps = result.scalars().all()

            for dep_obj in old_deps:
                try:
                    stripe.checkout.Session.expire(dep_obj.stripe_session_id)
                except Exception as e:
                    self.logger.warning(
                        f"[expire_old_stripe_deposits] cannot expire session={dep_obj.stripe_session_id}: {e}"
                    )
                dep_obj.status = 'cancelled'
                session.add(dep_obj)

            await session.commit()
            self.logger.info(f"[expire_old_stripe_deposits] => cancelled {len(old_deps)} stale sessions")

    # -------------------------------------------------------------------------
    # 4) HELPER => Summation of leftover credits
    # -------------------------------------------------------------------------
    async def _get_user_credits_leftover(self, user_id: str) -> Optional[int]:
        """
        Return the user’s leftover credits. We reuse get_balance_details_for_user
        by temporarily overriding self._user_id_getter.
        """
        try:
            old_fn = self._user_id_getter
            self._user_id_getter = lambda: user_id
            bal_details = await self.get_balance_details_for_user()
            return bal_details["credits_balance"]
        except Exception as e:
            self.logger.error(f"[_get_user_credits_leftover] => {e}")
            return None
        finally:
            self._user_id_getter = old_fn

    # -------------------------------------------------------------------------
    # HELPER => Return the user's deposit limit from account.max_credits_balance
    # -------------------------------------------------------------------------
    async def _get_user_deposit_limit(self, user_id: str) -> Optional[int]:
        """
        Fetch the user's Account row => read account.max_credits_balance.
        If it's None, fallback to FALLBACK_DEPOSIT_LIMIT.
        If no account found => return None, so we can handle it.
        """
        async with self.get_async_session() as session:
            stmt = select(Account).where(Account.user_id == user_id)
            res = await session.execute(stmt)
            account_obj = res.scalar_one_or_none()
            if not account_obj:
                return None
            if account_obj.max_credits_balance is not None:
                return account_obj.max_credits_balance
            else:
                return FALLBACK_DEPOSIT_LIMIT

    # -------------------------------------------------------------------------
    # HELPER => Sum of 'pending' deposit_amount for user
    # -------------------------------------------------------------------------
    async def _get_pending_deposit_sum_for_user(self, user_id: str, exclude_session_id: str = None) -> int:
        stmt = select(func.sum(StripeDeposit.deposit_amount)).where(
            StripeDeposit.user_id == user_id,
            StripeDeposit.status == 'pending'
        )
        if exclude_session_id:
            stmt = stmt.where(StripeDeposit.stripe_session_id != exclude_session_id)

        async with self.get_async_session() as session:
            result = await session.execute(stmt)
            s = result.scalar()
            return s or 0

    # -------------------------------------------------------------------------
    # The rest of your existing methods from the old code remain:
    #  - create_stripe_deposit(...)
    #  - mark_stripe_deposit_completed(...)
    #  - apply_credit_card_authorization(...)
    #  - apply_stripe_deposit(...)
    #  - create_stripe_authorization_session(...) etc.
    # We'll show them below for completeness.
    # -------------------------------------------------------------------------
    async def create_stripe_deposit(self, user_id: str, deposit_amount: int, session_id: str) -> int:
        async with self.get_async_session() as session:
            new_dep = StripeDeposit(
                user_id=user_id,
                deposit_amount=deposit_amount,
                stripe_session_id=session_id,
                status='pending'
            )
            session.add(new_dep)
            await session.commit()
            await session.refresh(new_dep)
            logging.info(f"[create_stripe_deposit] Created deposit id={new_dep.id} for user={user_id}, amt={deposit_amount}")
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
            logging.info(f"[mark_stripe_deposit_completed] Deposit {dep_obj.id} marked completed.")
            return dep_obj

    async def apply_credit_card_authorization(self, deposit_obj: StripeDeposit):
        """
        For an 'authorize-only' deposit => create a hold with hold_type=CreditCard, etc.
        """
        from datetime import datetime, timedelta
        user_id = deposit_obj.user_id
        amount = deposit_obj.deposit_amount

        try:
            async with self.get_async_session() as session:
                account = await self.get_or_create_account(user_id, session)
                cc_hold = Hold(
                    account_id=account.id,
                    user_id=user_id,
                    hold_type=HoldType.CreditCard,
                    total_amount=amount,
                    used_amount=0.0,
                    expiry=datetime.utcnow() + timedelta(days=6),
                    charged=False,
                    charged_amount=0.0,
                    parent_hold_id=None,
                    stripe_deposit_id=deposit_obj.id
                )
                session.add(cc_hold)

                deposit_obj.status = 'completed'
                session.add(deposit_obj)

                await session.commit()
                self.logger.info(f"[apply_credit_card_authorization] Created CC hold={cc_hold.id} for user={user_id}")
        except Exception as e:
            self.logger.error(f"[apply_credit_card_authorization] error: {e}")

    async def apply_stripe_deposit(self, deposit_obj: StripeDeposit):
        """
        If final deposit-limit check passes => credit the user’s account
        by creating a deposit-based hold or a direct credit transaction.
        """
        from .....models.enums import CreditTxnType
        user_id = deposit_obj.user_id
        deposit_amount = deposit_obj.deposit_amount
        deposit_id = deposit_obj.id

        try:
            tx = await self.create_credit_transaction_for_user(
                user_id=user_id,
                amount=deposit_amount,
                reason="stripe_deposit",
                txn_type=CreditTxnType.Add
            )
            await self.link_deposit_to_transaction(deposit_id, tx.id)
            self.logger.info(
                f"[apply_stripe_deposit] deposit {deposit_id} credited user={user_id} with {deposit_amount}"
            )
        except Exception as e:
            self.logger.error(f"[apply_stripe_deposit] error: {e}")

    async def create_stripe_authorization_session(self, amount: int) -> dict:
        """
        Used for 'authorize-only' flow. We keep the old logic, skipping deposit-limit check
        for an auth hold. If you want to add deposit-limit checks, you can do so similarly.
        """
        user_id = self.get_user_id()
        if amount < MINIMUM_CHECKOUT_AMOUNT:
            return {
                "error": f"Minimum checkout amount is {MINIMUM_CHECKOUT_AMOUNT} (internal units)",
                "status_code": 400
            }
        if amount <= 0:
            return {"error": "Invalid amount", "status_code": 400}

        try:
            amount_in_cents = int(amount / CENT_AMOUNT)
            session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                mode="payment",
                line_items=[{
                    "price_data": {
                        "currency": "usd",
                        "product_data": {"name": "Card Authorization Only"},
                        "unit_amount": amount_in_cents,
                    },
                    "quantity": 1,
                }],
                payment_intent_data={
                    "capture_method": "manual"
                },
                success_url=self.success_url,
                cancel_url=self.cancel_url,
            )
        except Exception as e:
            self.logger.error(f"[create_stripe_authorization_session] Error from Stripe: {e}")
            return {"error": str(e), "status_code": 500}

        try:
            deposit_id = await self._create_stripe_authorize_deposit(
                user_id=user_id,
                amount=amount,
                session_id=session.id,
                payment_intent_id=session.payment_intent
            )
        except Exception as e:
            self.logger.error(f"[create_stripe_authorization_session] DB error: {e}")
            return {"error": str(e), "status_code": 500}

        return {"session_id": session.id, "url": session.url}

    async def _create_stripe_authorize_deposit(self, user_id: str, amount: int, session_id: str, payment_intent_id: int) -> int:
        """
        For the 'authorize-only' scenario => store is_authorization=True
        """
        async with self.get_async_session() as session:
            new_dep = StripeDeposit(
                user_id=user_id,
                deposit_amount=amount,
                stripe_session_id=session_id,
                status='pending',
                is_authorization=True,
                payment_intent_id=payment_intent_id
            )
            session.add(new_dep)
            await session.commit()
            await session.refresh(new_dep)
            self.logger.info(
                f"[_create_stripe_authorize_deposit] Created authorize-only deposit id={new_dep.id} "
                f"for user={user_id}, amt={amount}"
            )
            return new_dep.id

    async def capture_stripe_payment_intent(self, hold: Hold, price: int, session):
        """
        For capturing authorized payments => old logic stays unchanged.
        """
        from .....models import StripeDeposit
        stmt = select(StripeDeposit).where(StripeDeposit.id == hold.stripe_deposit_id)
        result = await session.execute(stmt)
        deposit_obj = result.scalar_one_or_none()
        if not deposit_obj or not deposit_obj.payment_intent_id:
            self.logger.debug(f"[capture_stripe_payment_intent] hold={hold.id} => deposit missing payment_intent_id")
            return

        try:
            capture_amount_in_cents = int(price / CENT_AMOUNT)
            stripe.PaymentIntent.capture(
                deposit_obj.payment_intent_id,
                amount_to_capture=capture_amount_in_cents
            )
            self.logger.info(
                f"[capture_stripe_payment_intent] Captured PaymentIntent={deposit_obj.payment_intent_id} "
                f"for {capture_amount_in_cents} cents, hold_id={hold.id}, deposit_id={deposit_obj.id}"
            )
        except stripe.error.StripeError as e:
            self.logger.error(f"[capture_stripe_payment_intent] Stripe capture failed: {e}")
            raise ValueError(f"Stripe capture failed: {e}")
