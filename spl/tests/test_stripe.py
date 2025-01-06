import pytest
import json
from unittest.mock import patch
from sqlalchemy import select
from spl.db.server.app import original_app
from spl.models import StripeDeposit, Hold
from spl.models.enums import HoldType

@pytest.mark.asyncio
async def test_stripe_add_credits_flow(db_adapter_server_fixture):
    """
    Simulates creating a normal Stripe "add credits" session and then
    receiving a 'checkout.session.completed' webhook event.

    - create_stripe_credits_session(amount=100)
    - verify a StripeDeposit row (is_authorization=False) is created in status=pending
    - simulate the Stripe webhook => deposit->completed => user gets deposit-based 'Credits' hold
    """
    async with original_app.test_request_context("/"):
        server = db_adapter_server_fixture

        # Force user_id => e.g. "stripe_credits_user"
        test_user_id = "stripe_credits_user"
        server._user_id_getter = lambda: test_user_id

        # 1) Create a new stripe "credits" session
        amount = 100
        session_response = await server.create_stripe_credits_session(amount)
        assert "session_id" in session_response, "Expected a Stripe session_id in response"
        session_id = session_response["session_id"]

        # 2) Check the DB for the newly created StripeDeposit
        async with server.get_async_session() as s:
            deposit_q = await s.execute(
                select(StripeDeposit).where(StripeDeposit.stripe_session_id == session_id)
            )
            deposit_obj = deposit_q.scalar_one_or_none()
            assert deposit_obj is not None, "Expected to find a StripeDeposit record"
            assert deposit_obj.status == "pending"
            assert deposit_obj.is_authorization is False
            assert deposit_obj.deposit_amount == amount
            deposit_id = deposit_obj.id

        # 3) Simulate the Stripe webhook => craft a fake 'checkout.session.completed' event
        fake_stripe_payload = json.dumps({
            "id": "evt_1FAKE_COMPLETED",
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "id": session_id
                }
            }
        }).encode("utf-8")
        fake_signature = "sha256=foo"

        # 4) Patch stripe.Webhook.construct_event so it won't fail signature checks
        with patch("stripe.Webhook.construct_event") as mock_construct_event:
            # Return a dict shaped like a valid event
            mock_construct_event.return_value = {
                "id": "evt_1FAKE_COMPLETED",
                "type": "checkout.session.completed",
                "data": {
                    "object": {
                        "id": session_id
                    }
                }
            }

            # Actually call your webhook handler
            webhook_resp = await server.handle_stripe_webhook(
                payload=fake_stripe_payload,
                sig_header=fake_signature,
            )
            assert "status" in webhook_resp
            assert webhook_resp["status"] == "ok"

        # 5) Confirm deposit is now 'completed' and user has a 'Credits' hold
        async with server.get_async_session() as s:
            # Reload deposit
            deposit_q2 = await s.execute(
                select(StripeDeposit).where(StripeDeposit.id == deposit_id)
            )
            deposit_obj2 = deposit_q2.scalar_one_or_none()
            assert deposit_obj2 is not None
            assert deposit_obj2.status == "completed"

            # Because normal deposits create a deposit-based hold (via create_credit_transaction_for_user)
            # in Stripe’s `create_stripe_deposit`, check a 'Credits' hold exists with leftover == amount.
            holds = await s.execute(
                select(Hold).where(Hold.user_id == test_user_id).order_by(Hold.id.asc())
            )
            all_holds = holds.scalars().all()
            found_deposit_hold = [h for h in all_holds if h.hold_type == HoldType.Credits]
            assert len(found_deposit_hold) > 0, "Expected at least one 'Credits' hold for deposit"
            assert found_deposit_hold[-1].total_amount == amount
            assert found_deposit_hold[-1].used_amount == 0.0

        print("[test_stripe_add_credits_flow] => PASS")


@pytest.mark.asyncio
async def test_stripe_authorization_flow(db_adapter_server_fixture):
    """
    Simulates creating a Stripe "authorize-only" session and then
    receiving the 'checkout.session.completed' webhook event.

    - create_stripe_authorization_session(amount=300)
    - verify StripeDeposit row (is_authorization=True) in status=pending
    - simulate webhook => deposit->completed => user gets a 'CreditCard' hold
    """
    async with original_app.test_request_context("/"):
        server = db_adapter_server_fixture

        # Force user_id => e.g. "stripe_auth_user"
        test_user_id = "stripe_auth_user"
        server._user_id_getter = lambda: test_user_id

        # 1) Create a new stripe authorization session
        amount = 300
        session_response = await server.create_stripe_authorization_session(amount)
        assert "session_id" in session_response, "Expected a Stripe session_id in response"
        session_id = session_response["session_id"]

        # 2) Check the DB for the newly created StripeDeposit (is_authorization=True)
        async with server.get_async_session() as s:
            deposit_q = await s.execute(
                select(StripeDeposit).where(StripeDeposit.stripe_session_id == session_id)
            )
            deposit_obj = deposit_q.scalar_one_or_none()
            assert deposit_obj is not None, "Expected to find a StripeDeposit for auth-only"
            assert deposit_obj.status == "pending"
            assert deposit_obj.is_authorization is True
            assert deposit_obj.deposit_amount == amount
            deposit_id = deposit_obj.id

        # 3) Simulate the Stripe webhook => craft a fake 'checkout.session.completed' event
        fake_stripe_payload = json.dumps({
            "id": "evt_1FAKE_COMPLETED_AUTH",
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "id": session_id
                }
            }
        }).encode("utf-8")
        fake_signature = "sha256=foo"

        # 4) Patch stripe.Webhook.construct_event so it won't fail signature checks
        with patch("stripe.Webhook.construct_event") as mock_construct_event:
            mock_construct_event.return_value = {
                "id": "evt_1FAKE_COMPLETED_AUTH",
                "type": "checkout.session.completed",
                "data": {
                    "object": {
                        "id": session_id
                    }
                }
            }

            webhook_resp = await server.handle_stripe_webhook(
                payload=fake_stripe_payload,
                sig_header=fake_signature,
            )
            assert "status" in webhook_resp
            assert webhook_resp["status"] == "ok"

        # 5) Confirm deposit is now 'completed' and user has a 'CreditCard' hold
        async with server.get_async_session() as s:
            deposit_q2 = await s.execute(
                select(StripeDeposit).where(StripeDeposit.id == deposit_id)
            )
            deposit_obj2 = deposit_q2.scalar_one_or_none()
            assert deposit_obj2 is not None
            assert deposit_obj2.status == "completed"

            # For auth-only deposits, `apply_credit_card_authorization` => creates a new CC hold
            holds = await s.execute(
                select(Hold).where(Hold.user_id == test_user_id).order_by(Hold.id.asc())
            )
            all_holds = holds.scalars().all()

            found_cc_hold = [h for h in all_holds if h.hold_type == HoldType.CreditCard]
            assert len(found_cc_hold) == 1, (
                f"Expected exactly one credit-card hold after authorization. Found: {found_cc_hold}"
            )
            cc_hold = found_cc_hold[0]
            assert cc_hold.total_amount == amount
            assert cc_hold.used_amount == 0.0
            assert not cc_hold.charged

        print("[test_stripe_authorization_flow] => PASS")
