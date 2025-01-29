# spl/db/server/routes/debug_routes.py

from quart import jsonify
from .common import create_get_route, AuthMethod
from ..app import app, db_adapter_server

@app.route('/debug_invariant', methods=['GET'], endpoint='debug_invariant_endpoint')
async def debug_invariant_route():
    route_func = create_get_route(
        method=db_adapter_server.check_invariant,
        params=[],
        auth_method=AuthMethod.NONE
    )
    return await route_func()
