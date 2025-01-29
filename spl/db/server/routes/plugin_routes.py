# spl/db/server/routes/plugin_routes.py

from .common import create_get_route, create_post_route_return_id, AuthMethod
from ..app import app, db_adapter_server

@app.route('/get_plugin', methods=['GET'], endpoint='get_plugin_endpoint')
async def get_plugin_route():
    route_func = create_get_route(
        method=db_adapter_server.get_plugin,
        params=['plugin_id'],
        auth_method=AuthMethod.NONE
    )
    return await route_func()

@app.route('/create_plugin', methods=['POST'], endpoint='create_plugin_endpoint')
async def create_plugin_route():
    route_func = create_post_route_return_id(
        db_adapter_server.create_plugin,
        ['name','code'],
        'plugin_id',
        auth_method=AuthMethod.USER
    )
    return await route_func()

@app.route('/get_plugins', methods=['GET'], endpoint='get_plugins_endpoint')
async def get_plugins_route():
    route_func = create_get_route(
        method=db_adapter_server.get_plugins,
        params=[],
        auth_method=AuthMethod.NONE
    )
    return await route_func()
