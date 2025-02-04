# spl/db/server/routes/perms_routes.py

from .common import create_get_route, create_post_route_return_id, AuthMethod
from ..app import app
from ..db_server_instance import db_adapter_server

@app.route('/get_perm', methods=['GET'], endpoint='get_perm_endpoint')
async def get_perm_route():
    route_func = create_get_route(
        method=db_adapter_server.get_perm,
        params=['address','perm'],
        auth_method=AuthMethod.NONE
    )
    return await route_func()

@app.route('/create_perm', methods=['POST'], endpoint='create_perm_endpoint')
async def create_perm_route():
    route_func = create_post_route_return_id(
        db_adapter_server.create_perm,
        ['address','perm'],
        'perm_id'
    )
    return await route_func()

@app.route('/create_perm_description', methods=['POST'], endpoint='create_perm_description_endpoint')
async def create_perm_description_route():
    route_func = create_post_route_return_id(
        db_adapter_server.create_perm_description,
        ['perm_type','restricted_sot_id'],
        'perm_description_id'
    )
    return await route_func()
