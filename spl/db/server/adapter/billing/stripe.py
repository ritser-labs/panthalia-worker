# file: spl/db/server/adapter/billing/stripe.py

import os
import logging
import stripe
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy import select, func, asc

from .....models import StripeDeposit, Account, Hold, CENT_AMOUNT
from .....models.enums import CreditTxnType, HoldType

#############################################
# Configuration
#############################################
# Maximum number of pending Stripe checkout sessions allowed per user.
MAX_CONCURRENT_STRIPE_SESSIONS = 3
# If account.max_credits_balance is NULL, fall back to this value.
FALLBACK_DEPOSIT_LIMIT = 1000 * 100 * CENT_AMOUNT  # e.g. $1000 in internal units
# Minimum deposit amount (your original minimum).
MINIMUM_CHECKOUT_AMOUNT = 10 * 100 * CENT_AMOUNT  # e.g. $10

logger = logging.getLogger(__name__)


class DBAdapterStripeBillingMixin:
    """
    This mixin extends the Stripe-billing methods so that both immediate deposits
    and authorization-only flows obey a per‑user deposit limit (stored in Account.max_credits_balance),
    count existing pending deposits and leftover from CC holds toward usage, and limit
    concurrent pending checkouts. It also supports expiring stale checkouts.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.stripe_api_key = os.environ.get("STRIPE_SECRET_KEY", "")
        self.webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
        self.success_url = os.environ.get(
            "STRIPE_SUCCESS_URL", "https://panthalia.com/success")
        self.cancel_url = os.environ.get(
            "STRIPE_CANCEL_URL", "https://panthalia.com/cancel")
        stripe.api_key = self.stripe_api_key

    # -------------------------------------------------------------------------
    # 1) CREATE A STRIPE CREDITS SESSION (Immediate Deposit)
    # -------------------------------------------------------------------------
    async def create_stripe_credits_session(self, amount: int) -> dict:
        user_id = self.get_user_id()

        # (A) Basic validations
        if amount < MINIMUM_CHECKOUT_AMOUNT:
            return {"error": f"Minimum deposit is {MINIMUM_CHECKOUT_AMOUNT}", "status_code": 400}
        if amount <= 0:
            return {"error": "Invalid amount", "status_code": 400}

        # (B) Enforce concurrency: prune older pending sessions if needed.
        if not await self._prune_pending_sessions_if_needed(user_id):
            return {"error": "Failed to prune older sessions", "status_code": 500}

        # (C) Calculate user deposit usage:
        #    usage = leftover credits + pending deposits + leftover in CC holds.
        usage_before = await self._get_user_deposit_usage(user_id)
        if usage_before is None:
            return {"error": "Unable to compute deposit usage", "status_code": 400}

        deposit_limit = await self._get_user_deposit_limit(user_id)
        if deposit_limit is None:
            return {"error": "Unable to retrieve deposit limit", "status_code": 400}

        if usage_before + amount > deposit_limit:
            return {
                "error": f"Deposit limit exceeded. Current usage={usage_before}, requested={amount}, limit={deposit_limit}",
                "status_code": 400
            }

        # (D) Create the Stripe Checkout Session.
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
            self.logger.error(
                f"[create_stripe_credits_session] Stripe error: {e}")
            return {"error": str(e), "status_code": 500}

        # (E) Record a new pending deposit in the DB.
        try:
            await self.create_stripe_deposit(user_id, amount, session_obj.id)
        except Exception as ex:
            self.logger.error(
                f"[create_stripe_credits_session] DB error: {ex}")
            return {"error": str(ex), "status_code": 500}

        return {"session_id": session_obj.id, "url": session_obj.url}

    # -------------------------------------------------------------------------
    # 2) CREATE A STRIPE AUTHORIZATION SESSION (Authorize-Only Flow)
    # -------------------------------------------------------------------------
    async def create_stripe_authorization_session(self, amount: int) -> dict:
        user_id = self.get_user_id()
        if amount < MINIMUM_CHECKOUT_AMOUNT:
            return {"error": f"Minimum deposit is {MINIMUM_CHECKOUT_AMOUNT}", "status_code": 400}
        if amount <= 0:
            return {"error": "Invalid amount", "status_code": 400}

        if not await self._prune_pending_sessions_if_needed(user_id):
            return {"error": "Failed to prune older sessions", "status_code": 500}

        usage_before = await self._get_user_deposit_usage(user_id)
        if usage_before is None:
            return {"error": "Unable to compute deposit usage", "status_code": 400}

        dep_limit = await self._get_user_deposit_limit(user_id)
        if dep_limit is None:
            return {"error": "Unable to retrieve deposit limit", "status_code": 400}

        if usage_before + amount > dep_limit:
            return {
                "error": f"Authorization limit exceeded. usage={usage_before}, request={amount}, limit={dep_limit}",
                "status_code": 400
            }

        try:
            amount_in_cents = int(amount / CENT_AMOUNT)
            session_obj = stripe.checkout.Session.create(
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
                payment_intent_data={"capture_method": "manual"},
                success_url=self.success_url,
                cancel_url=self.cancel_url,
            )
        except Exception as e:
            self.logger.error(
                f"[create_stripe_authorization_session] Stripe error: {e}")
            return {"error": str(e), "status_code": 500}

        try:
            dep_id = await self._create_stripe_authorize_deposit(
                user_id=user_id,
                amount=amount,
                session_id=session_obj.id,
                payment_intent_id=session_obj.payment_intent
            )
            self.logger.info(
                f"[create_stripe_authorization_session] created deposit_id={dep_id}")
        except Exception as ex:
            self.logger.error(
                f"[create_stripe_authorization_session] DB error: {ex}")
            return {"error": str(ex), "status_code": 500}

        return {"session_id": session_obj.id, "url": session_obj.url}

    async def _create_stripe_authorize_deposit(self, user_id: str, amount: int, session_id: str, payment_intent_id: str) -> int:
        from .....models import StripeDeposit
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
            return new_dep.id

    # -------------------------------------------------------------------------
    # 3) CONCURRENCY PRUNE: Expire oldest pending sessions if user exceeds limit
    # -------------------------------------------------------------------------
    async def _prune_pending_sessions_if_needed(self, user_id: str) -> bool:
        from .....models import StripeDeposit
        async with self.get_async_session() as session:
            stmt_count = select(func.count(StripeDeposit.id)).where(
                StripeDeposit.user_id == user_id,
                StripeDeposit.status == 'pending'
            )
            res = await session.execute(stmt_count)
            count_pending = res.scalar() or 0
            if count_pending < MAX_CONCURRENT_STRIPE_SESSIONS:
                return True
            to_remove = count_pending - (MAX_CONCURRENT_STRIPE_SESSIONS - 1)
            if to_remove <= 0:
                return True
            stmt_oldest = (
                select(StripeDeposit)
                .where(StripeDeposit.user_id == user_id, StripeDeposit.status == 'pending')
                .order_by(StripeDeposit.created_at.asc())
                .limit(to_remove)
            )
            old_res = await session.execute(stmt_oldest)
            old_list = old_res.scalars().all()
            for dep_obj in old_list:
                sid = dep_obj.stripe_session_id
                try:
                    stripe.checkout.Session.expire(sid)
                except Exception as e:
                    self.logger.warning(
                        f"[_prune_pending_sessions_if_needed] cannot expire sess={sid}: {e}")
                dep_obj.status = 'cancelled'
                session.add(dep_obj)
            await session.commit()
        return True

    # -------------------------------------------------------------------------
    # 4) STRIPE WEBHOOK HANDLER: Final deposit-limit check and apply/cancel deposit
    # -------------------------------------------------------------------------
    async def handle_stripe_webhook(self, payload: bytes, sig_header: str) -> dict:
        from .....models import StripeDeposit
        if not self.webhook_secret:
            return {"error": "No webhook secret set", "status_code": 500}
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, self.webhook_secret)
        except stripe.error.SignatureVerificationError:
            return {"error": "Invalid signature", "status_code": 400}
        except Exception as e:
            self.logger.error(f"[handle_stripe_webhook] parse error: {e}")
            return {"error": str(e), "status_code": 400}
        if event["type"] == "checkout.session.completed":
            session_obj = event["data"]["object"]
            stripe_session_id = session_obj["id"]
            deposit_obj = await self.mark_stripe_deposit_completed(stripe_session_id)
            if not deposit_obj:
                self.logger.warning(
                    f"[handle_stripe_webhook] no deposit found for sess={stripe_session_id}")
                return {"status": "ok"}
            if deposit_obj.status == 'completed' and deposit_obj.credit_transaction_id:
                return {"status": "ok"}
            if deposit_obj.is_authorization:
                await self.apply_credit_card_authorization(deposit_obj)
            else:
                user_id = deposit_obj.user_id
                usage_before = await self._get_user_deposit_usage(user_id, exclude_session_id=stripe_session_id)
                if usage_before is None:
                    await self._mark_stripe_deposit_cancelled(deposit_obj)
                    return {"status": "ok"}
                dep_limit = await self._get_user_deposit_limit(user_id)
                if dep_limit is None:
                    await self._mark_stripe_deposit_cancelled(deposit_obj)
                    return {"status": "ok"}
                new_amt = deposit_obj.deposit_amount
                if usage_before + new_amt > dep_limit:
                    self.logger.warning(
                        f"[handle_stripe_webhook] deposit {deposit_obj.id} would exceed limit; cancelling")
                    await self._mark_stripe_deposit_cancelled(deposit_obj)
                else:
                    await self.apply_stripe_deposit(deposit_obj)
        else:
            self.logger.debug(
                f"[handle_stripe_webhook] ignoring event type {event['type']}")
        return {"status": "ok"}

    async def _mark_stripe_deposit_cancelled(self, deposit_obj: StripeDeposit):
        async with self.get_async_session() as session:
            deposit_obj.status = 'cancelled'
            session.add(deposit_obj)
            await session.commit()
        self.logger.info(
            f"[_mark_stripe_deposit_cancelled] deposit {deposit_obj.id} cancelled")

    # -------------------------------------------------------------------------
    # 5) BACKGROUND: Expire stale pending Stripe sessions.
    # -------------------------------------------------------------------------
    async def expire_old_stripe_deposits(self, older_than_minutes: int = 60):
        cutoff = datetime.utcnow() - timedelta(minutes=older_than_minutes)
        from .....models import StripeDeposit
        async with self.get_async_session() as session:
            stmt = select(StripeDeposit).where(
                StripeDeposit.status == 'pending',
                StripeDeposit.created_at < cutoff
            )
            res = await session.execute(stmt)
            old_deps = res.scalars().all()
            for dep in old_deps:
                sid = dep.stripe_session_id
                try:
                    stripe.checkout.Session.expire(sid)
                except Exception as e:
                    self.logger.warning(
                        f"[expire_old_stripe_deposits] cannot expire sess={sid}: {e}")
                dep.status = 'cancelled'
                session.add(dep)
            await session.commit()
            self.logger.info(
                f"[expire_old_stripe_deposits] cancelled {len(old_deps)} stale sessions")

    # -------------------------------------------------------------------------
    # 6) HELPER: Calculate user deposit usage = leftover credits + pending deposits + CC hold leftover.
    # -------------------------------------------------------------------------
    async def _get_user_deposit_usage(self, user_id: str, exclude_session_id: Optional[str] = None) -> Optional[int]:
        leftover = await self._get_user_credits_leftover(user_id)
        if leftover is None:
            return None
        pending = await self._get_pending_deposit_sum_for_user(user_id, exclude_session_id)
        cc_leftover = await self._get_authorized_cc_sum_for_user(user_id)
        return leftover + pending + cc_leftover

    async def _get_authorized_cc_sum_for_user(self, user_id: str) -> int:
        from .....models import Hold
        async with self.get_async_session() as session:
            stmt = select(Hold).where(
                Hold.user_id == user_id,
                Hold.hold_type == HoldType.CreditCard,
                Hold.charged == False
            )
            res = await session.execute(stmt)
            cc_holds = res.scalars().all()
            total = 0
            for h in cc_holds:
                left = h.total_amount - h.used_amount
                if left > 0:
                    total += left
            return total

    async def _get_user_deposit_limit(self, user_id: str) -> Optional[int]:
        from .....models import Account
        async with self.get_async_session() as session:
            stmt = select(Account).where(Account.user_id == user_id)
            res = await session.execute(stmt)
            acct = res.scalar_one_or_none()
            if not acct:
                return None
            if acct.max_credits_balance is not None:
                return acct.max_credits_balance
            else:
                return FALLBACK_DEPOSIT_LIMIT

    async def _get_user_credits_leftover(self, user_id: str) -> Optional[int]:
        try:
            old_fn = self._user_id_getter
            self._user_id_getter = lambda: user_id
            bal = await self.get_balance_details_for_user()
            return bal["credits_balance"]
        except Exception as e:
            self.logger.error(f"[_get_user_credits_leftover] error: {e}")
            return None
        finally:
            self._user_id_getter = old_fn

    async def _get_pending_deposit_sum_for_user(self, user_id: str, exclude_session_id: Optional[str] = None) -> int:
        stmt = select(func.sum(StripeDeposit.deposit_amount)).where(
            StripeDeposit.user_id == user_id,
            StripeDeposit.status == 'pending'
        )
        if exclude_session_id:
            stmt = stmt.where(
                StripeDeposit.stripe_session_id != exclude_session_id)
        async with self.get_async_session() as session:
            res = await session.execute(stmt)
            s = res.scalar()
            return s or 0

    # -------------------------------------------------------------------------
    # 7) EXISTING DEPOSIT METHODS
    # -------------------------------------------------------------------------
    async def create_stripe_deposit(self, user_id: str, deposit_amount: int, session_id: str) -> int:
        from .....models import StripeDeposit
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
            self.logger.info(
                f"[create_stripe_deposit] Created deposit id={new_dep.id} for user={user_id}, amt={deposit_amount}")
            return new_dep.id

    async def mark_stripe_deposit_completed(self, stripe_session_id: str) -> Optional[StripeDeposit]:
        from .....models import StripeDeposit
        async with self.get_async_session() as session:
            stmt = select(StripeDeposit).where(
                StripeDeposit.stripe_session_id == stripe_session_id)
            res = await session.execute(stmt)
            dep_obj = res.scalar_one_or_none()
            if not dep_obj:
                return None
            if dep_obj.status == 'completed':
                return dep_obj
            dep_obj.status = 'completed'
            session.add(dep_obj)
            await session.commit()
            await session.refresh(dep_obj)
            self.logger.info(
                f"[mark_stripe_deposit_completed] Deposit {dep_obj.id} marked completed.")
            return dep_obj

    async def apply_credit_card_authorization(self, deposit_obj: StripeDeposit):
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
                    used_amount=0,
                    expiry=datetime.utcnow() + timedelta(days=6),
                    charged=False,
                    charged_amount=0,
                    parent_hold_id=None,
                    stripe_deposit_id=deposit_obj.id
                )
                session.add(cc_hold)
                deposit_obj.status = 'completed'
                session.add(deposit_obj)
                await session.commit()
                self.logger.info(
                    f"[apply_credit_card_authorization] Created CC hold {cc_hold.id} for user {user_id}")
        except Exception as e:
            self.logger.error(f"[apply_credit_card_authorization] error: {e}")

    async def apply_stripe_deposit(self, deposit_obj: StripeDeposit):
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
                f"[apply_stripe_deposit] Deposit {deposit_id} credited user {user_id} with {deposit_amount}")
        except Exception as e:
            self.logger.error(f"[apply_stripe_deposit] error: {e}")

    async def capture_stripe_payment_intent(self, hold: Hold, price: int, session):
        from .....models import StripeDeposit
        stmt = select(StripeDeposit).where(
            StripeDeposit.id == hold.stripe_deposit_id)
        res = await session.execute(stmt)
        deposit_obj = res.scalar_one_or_none()
        if not deposit_obj or not deposit_obj.payment_intent_id:
            self.logger.debug(
                f"[capture_stripe_payment_intent] hold {hold.id} missing payment_intent_id, skipping capture")
            return
        try:
            capture_amount_in_cents = int(price / CENT_AMOUNT)
            stripe.PaymentIntent.capture(
                deposit_obj.payment_intent_id,
                amount_to_capture=capture_amount_in_cents
            )
            self.logger.info(
                f"[capture_stripe_payment_intent] Captured PaymentIntent {deposit_obj.payment_intent_id} for {capture_amount_in_cents} cents, hold {hold.id}, deposit {deposit_obj.id}")
        except stripe.error.StripeError as e:
            self.logger.error(
                f"[capture_stripe_payment_intent] Stripe capture failed: {e}")
            raise ValueError(f"Stripe capture failed: {e}")
        if hold.hold_type == HoldType.CreditCard:
            leftover = hold.total_amount - price
            if leftover > 0:
                user_id = hold.user_id
                usage_before = await self._get_user_deposit_usage(user_id)
                if usage_before is None:
                    leftover = 0
                else:
                    dep_limit = await self._get_user_deposit_limit(user_id)
                    if dep_limit is None or usage_before + leftover > dep_limit:
                        self.logger.warning(
                            f"[capture_stripe_payment_intent] leftover would exceed limit; setting leftover=0")
                        leftover = 0
            hold.charged_amount += price
            hold.total_amount = 0
            hold.used_amount = 0
            hold.charged = True
            session.add(hold)
            await session.flush()
            if leftover > 0:
                from datetime import datetime, timedelta
                from .....models.enums import CreditTxnType
                from .....models import CreditTransaction
                new_hold = Hold(
                    account_id=hold.account_id,
                    user_id=hold.user_id,
                    hold_type=HoldType.Credits,
                    total_amount=leftover,
                    used_amount=0,
                    expiry=datetime.utcnow() + timedelta(days=365),
                    charged=False,
                    charged_amount=0,
                    parent_hold_id=hold.id
                )
                session.add(new_hold)
                new_tx = CreditTransaction(
                    account_id=hold.account_id,
                    user_id=hold.user_id,
                    amount=leftover,
                    txn_type=CreditTxnType.Add,
                    reason="CC leftover"
                )
                session.add(new_tx)
            await session.flush()
