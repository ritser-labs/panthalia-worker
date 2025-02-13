# spl/db/server/routes/withdrawal_routes.py

from .common import create_get_route, create_post_route_return_id, create_post_route, AuthMethod
from ..app import app
from ..db_server_instance import db_adapter_server

@app.route('/create_withdrawal', methods=['POST'], endpoint='create_withdrawal_endpoint')
async def create_withdrawal_route():
    route_func = create_post_route_return_id(
        db_adapter_server.create_withdrawal_request,
        ['amount','payment_instructions'],
        'withdrawal_id',
        auth_method=AuthMethod.USER
    )
    return await route_func()

@app.route('/complete_withdrawal', methods=['POST'], endpoint='complete_withdrawal_endpoint')
async def complete_withdrawal_route():
    route_func = create_post_route(
        db_adapter_server.complete_withdrawal_flow,
        ['withdrawal_id','payment_record'],
        auth_method=AuthMethod.ADMIN
    )
    return await route_func()

@app.route('/reject_withdrawal', methods=['POST'], endpoint='reject_withdrawal_endpoint')
async def reject_withdrawal_route():
    route_func = create_post_route(
        db_adapter_server.reject_withdrawal_flow,
        ['withdrawal_id','rejection_reason'],
        auth_method=AuthMethod.ADMIN
    )
    return await route_func()

@app.route('/get_pending_withdrawals', methods=['GET'], endpoint='get_pending_withdrawals_endpoint')
async def get_pending_withdrawals_route():
    route_func = create_get_route(
        method=db_adapter_server.get_pending_withdrawals,
        params=['offset', 'limit'],
        auth_method=AuthMethod.ADMIN
    )
    return await route_func()

@app.route('/get_withdrawals_for_user', methods=['GET'], endpoint='get_withdrawals_for_user_endpoint')
async def get_withdrawals_for_user_route():
    route_func = create_get_route(
        method=db_adapter_server.get_withdrawals_for_user,
        params=['offset', 'limit'],
        auth_method=AuthMethod.USER
    )
    return await route_func()
