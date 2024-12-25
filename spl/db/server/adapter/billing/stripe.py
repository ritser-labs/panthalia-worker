# spl/db/server/adapter/billing/stripe.py

import os
import logging
import stripe
from ....models.enums import CreditTxnType

class DBAdapterStripeBillingMixin:
    """
    A mixin that adds stripe-billing-related methods to the DB adapter.
    It depends on DBAdapterAccountsMixin for the actual deposit & credit logic:
      - create_stripe_deposit(...)
      - mark_stripe_deposit_completed(...)
      - create_credit_transaction_for_user(...)
      - link_deposit_to_transaction(...)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

        self.stripe_api_key = os.environ.get("STRIPE_SECRET_KEY", "")
        self.webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
        self.success_url = os.environ.get("STRIPE_SUCCESS_URL", "https://example.com/success")
        self.cancel_url = os.environ.get("STRIPE_CANCEL_URL", "https://example.com/cancel")

        stripe.api_key = self.stripe_api_key

    async def create_stripe_session(self, user_id: str, amount: float) -> dict:
        """
        Creates a Stripe Checkout Session. Also calls self.create_stripe_deposit(...) to record pending deposit.
        """
        if amount <= 0:
            return {"error": "Invalid amount", "status_code": 400}

        try:
            amount_in_cents = int(round(amount * 100))
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
            self.logger.error(f"[create_stripe_session] Error from Stripe: {e}")
            return {"error": str(e), "status_code": 500}

        # Insert the deposit row in DB (pending)
        try:
            await self.create_stripe_deposit(user_id, amount, session.id)
        except Exception as e:
            self.logger.error(f"[create_stripe_session] DB error create_stripe_deposit: {e}")
            return {"error": str(e), "status_code": 500}

        return {"session_id": session.id, "url": session.url}

    async def handle_stripe_webhook(self, payload: bytes, sig_header: str) -> dict:
        """
        Called from /stripe/webhook route. On checkout.session.completed, we mark deposit completed
        and apply the deposit to the user’s balance (creating a credit transaction).
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

            # If deposit already had status='completed' and credit_transaction_id != None, skip
            if deposit_obj.status == 'completed' and deposit_obj.credit_transaction_id:
                self.logger.info(f"[handle_stripe_webhook] deposit {deposit_obj.id} was already credited.")
                return {"status": "ok"}

            # Otherwise, apply the deposit now
            await self.apply_stripe_deposit(deposit_obj)
        else:
            self.logger.debug(f"[handle_stripe_webhook] ignoring event type {event['type']}")

        return {"status": "ok"}

    async def apply_stripe_deposit(self, deposit_obj):
        """
        Actually apply the deposit after it's marked completed:
          - create a credit transaction with txn_type=CreditTxnType.Add
          - update user’s balance
          - link deposit -> transaction
        """
        user_id = deposit_obj.user_id
        deposit_amount = deposit_obj.deposit_amount
        deposit_id = deposit_obj.id

        try:
            # Create a new credit transaction for this deposit
            tx = await self.create_credit_transaction_for_user(
                user_id=user_id,
                amount=deposit_amount,
                reason="stripe_deposit",
                txn_type=CreditTxnType.Add
            )
            # Link deposit -> transaction
            await self.link_deposit_to_transaction(deposit_id, tx.id)
            self.logger.info(f"[apply_stripe_deposit] deposit {deposit_id} credited user {user_id} with {deposit_amount}")
        except Exception as e:
            self.logger.error(f"[apply_stripe_deposit] error: {e}")
