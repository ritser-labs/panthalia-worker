# spl/db/server/routes/auth_routes.py

from .common import create_post_route, AuthMethod
from ..app import app
from ..db_server_instance import db_adapter_server

@app.route('/admin_deposit_account', methods=['POST'], endpoint='admin_deposit_account_endpoint')
async def admin_deposit_account_route():
    route_func = create_post_route(
        db_adapter_server.admin_deposit_account,
        ['user_id','amount'],
        auth_method=AuthMethod.KEY
    )
    return await route_func()

@app.route('/create_account_key', methods=['POST'], endpoint='create_account_key_endpoint')
async def create_account_key_route():
    route_func = create_post_route(
        db_adapter_server.create_account_key,
        [],
        auth_method=AuthMethod.USER
    )
    return await route_func()

@app.route('/admin_create_account_key', methods=['POST'], endpoint='admin_create_account_key_endpoint')
async def admin_create_account_key_route():
    route_func = create_post_route(
        db_adapter_server.admin_create_account_key,
        ['user_id'],
        auth_method=AuthMethod.KEY
    )
    return await route_func()

@app.route('/delete_account_key', methods=['POST'], endpoint='delete_account_key_endpoint')
async def delete_account_key_route():
    route_func = create_post_route(
        db_adapter_server.delete_account_key,
        ['account_key_id'],
        auth_method=AuthMethod.USER
    )
    return await route_func()

@app.route('/account_key_from_public_key', methods=['GET'], endpoint='account_key_from_public_key_endpoint')
async def account_key_from_public_key_route():
    from .common import create_get_route
    route_func = create_get_route(
        method=db_adapter_server.account_key_from_public_key,
        params=['public_key'],
        auth_method=AuthMethod.KEY
    )
    return await route_func()

@app.route('/user/get_account_keys', methods=['GET'], endpoint='get_account_keys_endpoint')
async def get_account_keys_route():
    from .common import create_get_route
    route_func = create_get_route(
        method=db_adapter_server.get_account_keys,
        params=['offset', 'limit'],
        auth_method=AuthMethod.USER
    )
    return await route_func()

@app.route('/user/balance_and_holds', methods=['GET'], endpoint='get_balance_endpoint')
async def get_balance_route():
    from .common import create_get_route
    route_func = create_get_route(
        method=db_adapter_server.get_balance_details_for_user,
        params=['offset', 'limit'],
        auth_method=AuthMethod.USER
    )
    return await route_func()
