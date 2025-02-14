# spl/db/server/routes/stripe_routes.py

from datetime import timedelta
from quart import request, jsonify
from .common import create_post_route, AuthMethod
from ..app import app
from ..db_server_instance import db_adapter_server
from ..rate_limiter import rate_limit  # our new rate limiter module

# Limit to 1 request per 10 seconds per user for creating a Stripe credits session.
@app.route('/create_stripe_credits_session', methods=['POST'], endpoint='create_stripe_credits_session_endpoint')
@rate_limit(1, timedelta(seconds=10))
async def create_stripe_credits_session_route():
    route_func = create_post_route(
        db_adapter_server.create_stripe_credits_session,
        ['amount'],
        auth_method=AuthMethod.USER
    )
    return await route_func()

# Limit to 1 request per 10 seconds per user for creating a Stripe authorization session.
@app.route('/create_stripe_auth_session', methods=['POST'], endpoint='create_stripe_auth_session_endpoint')
@rate_limit(1, timedelta(seconds=10))
async def create_stripe_auth_session_route():
    route_func = create_post_route(
        db_adapter_server.create_stripe_authorization_session,
        ['amount'],
        auth_method=AuthMethod.USER
    )
    return await route_func()

# This endpoint is for handling Stripe webhooks, so it is not rate limited.
@app.route("/stripe/webhook", methods=["POST"])
async def stripe_webhook_route():
    payload = await request.get_data()
    sig_header = request.headers.get("Stripe-Signature", "")
    response = await db_adapter_server.handle_stripe_webhook(payload, sig_header)
    if "error" in response:
        return jsonify({"error": response["error"]}), response.get("status_code", 400)
    return jsonify(response), 200
