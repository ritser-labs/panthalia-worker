# spl/db/server/routes/subnet_routes.py

from .common import create_get_route, create_post_route_return_id, create_post_route, AuthMethod
from ..app import app
from ..db_server_instance import db_adapter_server

@app.route('/get_subnet', methods=['GET'], endpoint='get_subnet_endpoint')
async def get_subnet_route():
    route_func = create_get_route(
        method=db_adapter_server.get_subnet,
        params=['subnet_id']
    )
    return await route_func()

@app.route('/get_subnet_using_address', methods=['GET'], endpoint='get_subnet_using_address_endpoint')
async def get_subnet_using_address_route():
    route_func = create_get_route(
        method=db_adapter_server.get_subnet_using_address,
        params=['address']
    )
    return await route_func()

@app.route('/create_subnet', methods=['POST'], endpoint='create_subnet_endpoint')
async def create_subnet_route():
    route_func = create_post_route_return_id(
        db_adapter_server.create_subnet,
        ['dispute_period','solve_period','stake_multiplier','target_price', 'description'],
        'subnet_id'
    )
    return await route_func()

@app.route('/set_subnet_target_price', methods=['POST'], endpoint='set_subnet_target_price_endpoint')
async def set_subnet_target_price_route():
    route_func = create_post_route(
        db_adapter_server.set_subnet_target_price,
        ['subnet_id','target_price'],
        auth_method=AuthMethod.KEY
    )
    return await route_func()
