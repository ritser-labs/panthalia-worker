# spl/db/server/routes/master_state_routes.py

from .common import create_get_route, create_post_route, AuthMethod
from ..app import app, db_adapter_server

@app.route('/get_master_job_state', methods=['GET'], endpoint='get_master_job_state_endpoint')
async def get_master_job_state_route():
    route_func = create_get_route(
        method=db_adapter_server.get_master_job_state,
        params=['job_id'],
        auth_method=AuthMethod.KEY
    )
    return await route_func()

@app.route('/update_master_job_state', methods=['POST'], endpoint='update_master_job_state_endpoint')
async def update_master_job_state_route():
    route_func = create_post_route(
        db_adapter_server.update_master_job_state,
        required_keys=['job_id','new_state'],
        auth_method=AuthMethod.KEY
    )
    return await route_func()

@app.route('/create_state_update', methods=['POST'], endpoint='create_state_update_endpoint')
async def create_state_update_route():
    route_func = create_post_route(
        db_adapter_server.create_state_update,
        required_keys=['job_id','data'],
        auth_method=AuthMethod.KEY
    )
    return await route_func()
