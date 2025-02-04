# spl/db/server/routes/sot_routes.py

from .common import create_get_route, create_post_route_return_id, create_post_route, AuthMethod
from ..app import app
from ..db_server_instance import db_adapter_server

@app.route('/sot/get_job_state', methods=['GET'], endpoint='sot_get_job_state_endpoint')
async def sot_get_job_state_route():
    route_func = create_get_route(
        method=db_adapter_server.get_sot_job_state,
        params=['job_id'],
        auth_method=AuthMethod.SOT
    )
    return await route_func()


@app.route('/sot/update_job_state', methods=['POST'], endpoint='sot_update_job_state_endpoint')
async def sot_update_job_state_route():
    route_func = create_post_route(
        db_adapter_server.update_sot_job_state,
        ['job_id','new_state'],
        auth_method=AuthMethod.SOT
    )
    return await route_func()


@app.route('/sot/get_job', methods=['GET'], endpoint='sot_get_job_endpoint')
async def sot_get_job_route():
    route_func = create_get_route(
        method=db_adapter_server.get_job,
        params=['job_id'],
        auth_method=AuthMethod.SOT
    )
    return await route_func()


@app.route('/sot/get_sot', methods=['GET'], endpoint='sot_get_sot_endpoint')
async def sot_get_sot_route():
    route_func = create_get_route(
        method=db_adapter_server.get_sot,
        params=['sot_id'],
        auth_method=AuthMethod.SOT
    )
    return await route_func()


@app.route('/create_sot', methods=['POST'], endpoint='create_sot_endpoint')
async def create_sot_route():
    route_func = create_post_route_return_id(
        db_adapter_server.create_sot,
        ['job_id','address','url'],
        'sot_id'
    )
    return await route_func()


@app.route('/update_sot', methods=['POST'], endpoint='update_sot_endpoint')
async def update_sot_route():
    route_func = create_post_route_return_id(
        db_adapter_server.update_sot,
        ['sot_id','url'],
        'success'
    )
    return await route_func()


#
# ------------- MISSING ENDPOINT ADDED: /get_sot_by_job_id -------------
#

@app.route('/get_sot_by_job_id', methods=['GET'], endpoint='get_sot_by_job_id_endpoint')
async def get_sot_by_job_id_route():
    """
    GET /get_sot_by_job_id?job_id=...
    Returns the SOT for that Job, requires KEY auth by default.
    """
    route_func = create_get_route(
        method=db_adapter_server.get_sot_by_job_id,
        params=['job_id'],
        auth_method=AuthMethod.KEY
    )
    return await route_func()
