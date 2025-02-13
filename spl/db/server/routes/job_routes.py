# spl/db/server/routes/job_routes.py

from .common import create_get_route, create_post_route, create_post_route_return_id, AuthMethod
from ..app import app
from ..db_server_instance import db_adapter_server

@app.route('/get_job', methods=['GET'], endpoint='get_job_endpoint')
async def get_job_route():
    route_func = create_get_route(
        method=db_adapter_server.get_job,
        params=['job_id'],
        auth_method=AuthMethod.NONE
    )
    return await route_func()


@app.route('/create_job', methods=['POST'], endpoint='create_job_endpoint')
async def create_job_route():
    route_func = create_post_route_return_id(
        db_adapter_server.create_job,
        required_keys=['name','plugin_id','sot_url','iteration','initial_state_url', 'replicate_prob'],
        id_key='job_id',
        auth_method=AuthMethod.USER
    )
    return await route_func()


@app.route('/update_job_iteration', methods=['POST'], endpoint='update_job_iteration_endpoint')
async def update_job_iteration_route():
    route_func = create_post_route_return_id(
        db_adapter_server.update_job_iteration,
        ['job_id','new_iteration'],
        'success',
        auth_method=AuthMethod.KEY
    )
    return await route_func()


@app.route('/mark_job_as_done', methods=['POST'], endpoint='mark_job_as_done_endpoint')
async def mark_job_as_done_route():
    route_func = create_post_route_return_id(
        db_adapter_server.mark_job_as_done,
        ['job_id'],
        'success',
        auth_method=AuthMethod.KEY
    )
    return await route_func()


@app.route('/get_jobs_in_progress', methods=['GET'])
async def get_jobs_in_progress_route():
    route_func = create_get_route(
        method=db_adapter_server.get_jobs_in_progress,
        params=[],
        auth_method=AuthMethod.NONE
    )
    return await route_func()


@app.route('/update_job_queue_status', methods=['POST'], endpoint='update_job_queue_status_endpoint')
async def update_job_queue_status_route():
    route_func = create_post_route(
        db_adapter_server.update_job_queue_status,
        ['job_id','new_queued','assigned_master_id'],
        auth_method=AuthMethod.KEY
    )
    return await route_func()


@app.route('/get_unassigned_queued_jobs', methods=['GET'], endpoint='get_unassigned_queued_jobs_endpoint')
async def get_unassigned_queued_jobs_route():
    route_func = create_get_route(
        method=db_adapter_server.get_unassigned_queued_jobs,
        params=[],
        auth_method=AuthMethod.KEY
    )
    return await route_func()


@app.route('/get_jobs_assigned_to_master', methods=['GET'], endpoint='get_jobs_assigned_to_master_endpoint')
async def get_jobs_assigned_to_master_route():
    route_func = create_get_route(
        method=db_adapter_server.get_jobs_assigned_to_master,
        params=['master_id'],
        auth_method=AuthMethod.KEY
    )
    return await route_func()


@app.route('/get_unassigned_unqueued_active_jobs', methods=['GET'], endpoint='get_unassigned_unqueued_active_jobs_endpoint')
async def get_unassigned_unqueued_active_jobs_route():
    route_func = create_get_route(
        method=db_adapter_server.get_unassigned_unqueued_active_jobs,
        params=[],
        auth_method=AuthMethod.KEY
    )
    return await route_func()


@app.route('/update_job_active', methods=['POST'], endpoint='update_job_active_endpoint')
async def update_job_active_route():
    route_func = create_post_route(
        db_adapter_server.update_job_active,
        ['job_id','active'],
        auth_method=AuthMethod.KEY
    )
    return await route_func()


@app.route('/stop_job', methods=['POST'], endpoint='stop_job_endpoint')
async def stop_job_route():
    route_func = create_post_route(
        db_adapter_server.stop_job,
        ['job_id'],
        auth_method=AuthMethod.USER
    )
    return await route_func()


#
# -------------- MISSING ENDPOINT ADDED: /update_job_sot_url --------------
#

@app.route('/update_job_sot_url', methods=['POST'], endpoint='update_job_sot_url_endpoint')
async def update_job_sot_url_route():
    """
    POST /update_job_sot_url
      JSON: { "job_id": <int>, "new_sot_url": <string> }
    Returns { "success": true } or { "success": false }
    """
    route_func = create_post_route_return_id(
        db_adapter_server.update_job_sot_url,
        ['job_id','new_sot_url'],
        'success',
        auth_method=AuthMethod.KEY
    )
    return await route_func()

@app.route('/get_total_state_updates_for_job', methods=['GET'], endpoint='get_total_state_updates_for_job_endpoint')
async def get_total_state_updates_for_job_route():
    route_func = create_get_route(
        method=db_adapter_server.get_total_state_updates_for_job,
        params=['job_id'],
        auth_method=AuthMethod.NONE
    )
    return await route_func()