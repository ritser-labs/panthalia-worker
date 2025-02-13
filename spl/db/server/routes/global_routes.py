# file: spl/db/server/routes/global_routes.py

from .common import create_get_route, AuthMethod
from ..app import app
from ..db_server_instance import db_adapter_server

@app.route('/global_stats', methods=['GET'], endpoint='get_global_stats_endpoint')
async def get_global_stats_route():
    """
    GET /get_global_stats
    Returns a JSON object with global statistics.
    """
    route_func = create_get_route(
        method=db_adapter_server.get_global_stats,
        params=[],
        auth_method=AuthMethod.NONE
    )
    return await route_func()
