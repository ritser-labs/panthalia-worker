# spl/db/server/adapter/billing/stripe.py

import os
import logging
import stripe
from .....models.enums import CreditTxnType
from .....models import StripeDeposit, HoldType, Hold, CENT_AMOUNT
from datetime import datetime, timedelta
from sqlalchemy import select

# Define a minimum checkout amount.
# For example, if CENT_AMOUNT is the internal unit for 1 cent, then 100 * CENT_AMOUNT represents $1.
MINIMUM_CHECKOUT_AMOUNT = 10 * 100 * CENT_AMOUNT  # $10 minimum in internal units

class DBAdapterStripeBillingMixin:
    """
    A mixin that adds stripe-billing-related methods to the DB adapter.
    It depends on DBAdapterAccountsMixin for the actual deposit & credit logic:
      - create_credit_transaction_for_user(...)
      - link_deposit_to_transaction(...)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

        self.stripe_api_key = os.environ.get("STRIPE_SECRET_KEY", "")
        self.webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
        self.success_url = os.environ.get("STRIPE_SUCCESS_URL", "https://panthalia.com/success")
        self.cancel_url = os.environ.get("STRIPE_CANCEL_URL", "https://panthalia.com/cancel")

        stripe.api_key = self.stripe_api_key

    async def create_stripe_credits_session(self, amount: int) -> dict:
        """
        Creates a Stripe Checkout Session. Also calls self.create_stripe_deposit(...) to record a pending deposit.
        Enforces a minimum checkout amount.
        """
        user_id = self.get_user_id()
        # Enforce minimum checkout amount
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
                        "product_data": {"name": "Panthalia Credits"},
                        "unit_amount": amount_in_cents,
                    },
                    "quantity": 1,
                }],
                success_url=self.success_url,
                cancel_url=self.cancel_url,
            )
        except Exception as e:
            self.logger.error(f"[create_stripe_credits_session] Error from Stripe: {e}")
            return {"error": str(e), "status_code": 500}

        try:
            await self.create_stripe_deposit(user_id, amount, session.id)
        except Exception as e:
            self.logger.error(f"[create_stripe_credits_session] DB error create_stripe_deposit: {e}")
            return {"error": str(e), "status_code": 500}

        return {"session_id": session.id, "url": session.url}

    async def handle_stripe_webhook(self, payload: bytes, sig_header: str) -> dict:
        """
        Called from /stripe/webhook route. On checkout.session.completed, we mark the deposit completed
        and apply the deposit to the user's balance.
        """
        if not self.webhook_secret:
            return {"error": "No webhook secret set", "status_code": 500}

        try:
            event = stripe.Webhook.construct_event(payload, sig_header, self.webhook_secret)
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
                self.logger.warning(f"[handle_stripe_webhook] no deposit found for session={stripe_session_id}")
                return {"status": "ok"}

            if deposit_obj.status == 'completed' and deposit_obj.credit_transaction_id:
                self.logger.info(f"[handle_stripe_webhook] deposit {deposit_obj.id} was already credited.")
                return {"status": "ok"}

            if deposit_obj.is_authorization:
                await self.apply_credit_card_authorization(deposit_obj)
            else:
                await self.apply_stripe_deposit(deposit_obj)

        else:
            self.logger.debug(f"[handle_stripe_webhook] ignoring event type {event['type']}")

        return {"status": "ok"}

    async def apply_credit_card_authorization(self, deposit_obj: StripeDeposit):
        """
        For an 'authorize-only' deposit: create a Hold with hold_type=HoldType.CreditCard.
        The leftover is deposit_obj.deposit_amount.
        """
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

    async def apply_stripe_deposit(self, deposit_obj):
        """
        Applies the deposit by creating a credit transaction and linking the deposit to the transaction.
        """
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
            self.logger.info(f"[apply_stripe_deposit] deposit {deposit_id} credited user {user_id} with {deposit_amount}")
        except Exception as e:
            self.logger.error(f"[apply_stripe_deposit] error: {e}")

    async def create_stripe_authorization_session(self, amount: int) -> dict:
        """
        Creates a Stripe Checkout Session with an uncaptured PaymentIntent (authorize-only).
        The card is only authorized, not charged immediately.
        Enforces a minimum checkout amount.
        """
        user_id = self.get_user_id()
        # Enforce minimum checkout amount
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
        Helper that creates a StripeDeposit record flagged as is_authorization=True.
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
            self.logger.info(f"[_create_stripe_authorize_deposit] Created authorize-only deposit id={new_dep.id} for user={user_id}, amt={amount}")
            return new_dep.id

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

    async def capture_stripe_payment_intent(self, hold: Hold, price: int, session) -> None:
        """
        Captures the authorized PaymentIntent for the specified price.
        If capture fails, raises a ValueError.
        """
        if not hold.stripe_deposit_id:
            self.logger.debug(f"[capture_stripe_payment_intent] hold={hold.id} => no stripe_deposit_id, skipping capture.")
            return

        stmt = select(StripeDeposit).where(StripeDeposit.id == hold.stripe_deposit_id)
        result = await session.execute(stmt)
        deposit_obj = result.scalar_one_or_none()
        if not deposit_obj or not deposit_obj.payment_intent_id:
            self.logger.debug(f"[capture_stripe_payment_intent] hold={hold.id} => deposit missing payment_intent_id, skipping capture.")
            return

        try:
            from .....models.enums import CENT_AMOUNT
            capture_amount_in_cents = int(price / CENT_AMOUNT)
            stripe.PaymentIntent.capture(
                deposit_obj.payment_intent_id,
                amount_to_capture=capture_amount_in_cents
            )
            self.logger.info(f"[capture_stripe_payment_intent] Captured PaymentIntent={deposit_obj.payment_intent_id} for {capture_amount_in_cents} cents, hold_id={hold.id}, deposit_id={deposit_obj.id}")
        except stripe.error.StripeError as e:
            self.logger.error(f"[capture_stripe_payment_intent] Stripe capture failed: {e}")
            raise ValueError(f"Stripe capture failed: {e}")

        return
