# spl/db/server/routes/stripe_routes.py

from quart import request, jsonify
from .common import create_post_route, AuthMethod
from ..app import app
from ..db_server_instance import db_adapter_server

@app.route('/create_stripe_credits_session', methods=['POST'], endpoint='create_stripe_credits_session_endpoint')
async def create_stripe_credits_session_route():
    route_func = create_post_route(
        db_adapter_server.create_stripe_credits_session,
        ['amount'],
        auth_method=AuthMethod.USER
    )
    return await route_func()

@app.route('/create_stripe_auth_session', methods=['POST'], endpoint='create_stripe_auth_session_endpoint')
async def create_stripe_auth_session_route():
    route_func = create_post_route(
        db_adapter_server.create_stripe_authorization_session,
        ['amount'],
        auth_method=AuthMethod.USER
    )
    return await route_func()

@app.route("/stripe/webhook", methods=["POST"])
async def stripe_webhook_route():
    payload = await request.get_data()
    sig_header = request.headers.get("Stripe-Signature", "")
    response = await db_adapter_server.handle_stripe_webhook(payload, sig_header)
    if "error" in response:
        return jsonify({"error": response["error"]}), response.get("status_code", 400)
    return jsonify(response), 200
