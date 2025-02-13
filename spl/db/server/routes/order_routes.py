# spl/db/server/routes/order_routes.py

from .common import create_get_route, create_post_route_return_id, create_post_route, AuthMethod
from ..app import app
from ..db_server_instance import db_adapter_server

@app.route('/create_order', methods=['POST'], endpoint='create_order_endpoint')
async def create_order_route():
    route_func = create_post_route_return_id(
        db_adapter_server.create_order,
        ['task_id','subnet_id','order_type','price','hold_id'],
        'order_id',
        auth_method=AuthMethod.USER
    )
    return await route_func()

@app.route('/delete_order', methods=['POST'], endpoint='delete_order_endpoint')
async def delete_order_route():
    route_func = create_post_route(
        db_adapter_server.delete_order,
        ['order_id'],
        auth_method=AuthMethod.USER
    )
    return await route_func()

@app.route('/get_num_orders', methods=['GET'], endpoint='get_num_orders_endpoint')
async def get_num_orders_route():
    route_func = create_get_route(
        method=db_adapter_server.get_num_orders,
        params=['subnet_id','order_type','matched'],
        auth_method=AuthMethod.USER
    )
    return await route_func()

@app.route('/get_orders_for_user', methods=['GET'], endpoint='get_orders_for_user_endpoint')
async def get_orders_for_user_route():
    route_func = create_get_route(
        method=db_adapter_server.get_orders_for_user,
        params=['offset', 'limit'],
        auth_method=AuthMethod.USER
    )
    return await route_func()

@app.route('/get_unmatched_orders_for_job', methods=['GET'], endpoint='get_unmatched_orders_for_job_endpoint')
async def get_unmatched_orders_for_job_route():
    route_func = create_get_route(
        method=db_adapter_server.get_unmatched_orders_for_job,
        params=['job_id'],
        auth_method=AuthMethod.KEY
    )
    return await route_func()
