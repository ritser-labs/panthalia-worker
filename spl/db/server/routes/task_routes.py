# spl/db/server/routes/task_routes.py

import logging
from quart import request, jsonify
from .common import (
    create_get_route,
    create_post_route_return_id,
    create_post_route,
    AuthMethod
)
from ..app import app
from ..db_server_instance import db_adapter_server

logger = logging.getLogger(__name__)


@app.route('/get_task', methods=['GET'], endpoint='get_task_endpoint')
async def get_task_route():
    route_func = create_get_route(
        method=db_adapter_server.get_task,
        params=['task_id'],
        auth_method=AuthMethod.NONE
    )
    return await route_func()


@app.route('/create_task', methods=['POST'], endpoint='create_task_endpoint')
async def create_task_route():
    route_func = create_post_route_return_id(
        db_adapter_server.create_task,
        ['job_id','job_iteration','status','params'],
        'task_id',
        auth_method=AuthMethod.USER
    )
    return await route_func()


@app.route('/submit_partial_result', methods=['POST'], endpoint='submit_partial_result_endpoint')
async def submit_partial_result():
    route_func = create_post_route(
        db_adapter_server.submit_partial_result,
        ['task_id','partial_result', 'final'],
        auth_method=AuthMethod.USER
    )
    return await route_func()


@app.route('/get_tasks_for_job', methods=['GET'], endpoint='get_tasks_for_job_endpoint')
async def get_tasks_for_job_route():
    route_func = create_get_route(
        method=db_adapter_server.get_tasks_with_pagination_for_job,
        params=['job_id','offset','limit']
    )
    return await route_func()


@app.route('/get_task_count_for_job', methods=['GET'], endpoint='get_task_count_for_job_endpoint')
async def get_task_count_for_job():
    job_id = request.args.get('job_id', type=int)
    count = await db_adapter_server.get_task_count_for_job(job_id)
    return jsonify(count), 200


@app.route('/get_task_count_by_status_for_job', methods=['GET'], endpoint='get_task_count_by_status_for_job_endpoint')
async def get_task_count_by_status_for_job():
    job_id = request.args.get('job_id', type=int)
    raw_list = request.args.getlist('statuses')
    if not raw_list:
        return jsonify({'error': 'No statuses provided'}), 400

    # Flatten comma-separated
    all_status_parts = []
    for item in raw_list:
        parts = [p.strip() for p in item.split(',')]
        all_status_parts.extend(parts)

    from ....models import TaskStatus
    statuses = []
    for st_str in all_status_parts:
        if st_str not in TaskStatus.__members__:
            return jsonify({'error': f'Invalid status: {st_str}'}), 400
        statuses.append(TaskStatus[st_str])

    result = await db_adapter_server.get_task_count_by_status_for_job(job_id, statuses)
    return jsonify(result), 200


@app.route('/get_last_task_with_status', methods=['GET'], endpoint='get_last_task_with_status_endpoint')
async def get_last_task_with_status():
    job_id = request.args.get('job_id', type=int)
    raw_list = request.args.getlist('statuses')
    if not raw_list:
        return jsonify({'error': 'No statuses provided'}), 400

    all_status_parts = []
    for item in raw_list:
        parts = [p.strip() for p in item.split(',')]
        all_status_parts.extend(parts)

    from ....models import TaskStatus
    statuses = []
    for st_str in all_status_parts:
        if st_str not in TaskStatus.__members__:
            return jsonify({'error': f'Invalid status: {st_str}'}), 400
        statuses.append(TaskStatus[st_str])

    task = await db_adapter_server.get_last_task_with_status(job_id, statuses)
    if task:
        return jsonify(task.as_dict()), 200
    return jsonify({'error': 'No task found'}), 404


@app.route('/create_bids_and_tasks', methods=['POST'], endpoint='create_bids_and_tasks_endpoint')
async def create_bids_and_tasks_route():
    route_func = create_post_route(
        db_adapter_server.create_bids_and_tasks,
        ['job_id','num_tasks','price','params','hold_id'],
        auth_method=AuthMethod.KEY
    )
    return await route_func()


#
# ------------- MISSING ENDPOINT: /update_task_status -------------
#
# We replicate the old logic that allowed multiple statuses in one string,
# e.g. "SolutionSubmitted,ResolvedCorrect", and we pick the rightmost recognized one.
#

@app.route('/update_task_status', methods=['POST'], endpoint='update_task_status_endpoint')
async def update_task_status_route():
    route_func = create_post_route(
        db_adapter_server.update_task_status,
        required_keys=['task_id','job_id','status'],
        auth_method=AuthMethod.KEY
    )
    return await route_func()


@app.route('/finalize_sanity_check', methods=['POST'], endpoint='finalize_sanity_check_endpoint')
async def finalize_sanity_check_route():
    route_func = create_post_route(
        db_adapter_server.finalize_sanity_check,
        ['task_id','is_valid'],
        auth_method=AuthMethod.KEY
    )
    return await route_func()


@app.route('/get_assigned_tasks', methods=['GET'], endpoint='get_assigned_tasks_endpoint')
async def get_assigned_tasks_route():
    """
    GET /get_assigned_tasks?subnet_id=...
    Returns tasks assigned to the user for that subnet.
    """
    from .common import create_get_route, AuthMethod
    route_func = create_get_route(
        method=db_adapter_server.get_assigned_tasks,
        params=['subnet_id'],
        auth_method=AuthMethod.USER
    )
    return await route_func()

@app.route('/update_replicated_parent', methods=['POST'])
async def update_replicated_parent_route():
    """
    POST /update_replicated_parent
      JSON: { "child_task_id": <int>, "parent_task_id": <int> }
    Returns { "success": true } on success
    """
    # typical route:
    route_func = create_post_route(
        db_adapter_server.update_replicated_parent,
        required_keys=['child_task_id','parent_task_id'],
        auth_method=AuthMethod.KEY
    )
    return await route_func()