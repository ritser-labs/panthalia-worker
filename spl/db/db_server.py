# db_server.py

import asyncio
from quart import Quart, request, jsonify
import logging
from .db_adapter_server import db_adapter_server
from ..api_auth import requires_authentication
from eth_account.messages import encode_defunct
from ..models import PermType, TaskStatus
from quart_cors import cors
import os

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Quart(__name__)

app = cors(app, allow_origin="http://localhost:3000")

# Define permission constants (should match those in your original PermType)
perm_modify_db = None

# Define file paths and locks (assuming similar to your original setup)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')
state_dir = os.path.join(data_dir, 'state')
db_adapter = None
os.makedirs(state_dir, exist_ok=True)

# Example: Define a permission type for modifying the DB
# This should be consistent with your original PermType enum
# Ensure that 'perm_db_column' corresponds to the permission required for each operation

def get_perm_modify_db():
    # Assuming that PERM_MODIFY_DB is the permission required to modify the DB
    return perm_modify_db

def get_db_adapter():
    return db_adapter_server

def requires_auth(f):
    return requires_authentication(
        get_db_adapter,
        get_perm_modify_db
    )(f)

# Define routes corresponding to each method in db_adapter_server.py

@app.route('/get_job', methods=['GET'])
async def get_job():
    job_id = request.args.get('job_id', type=int)
    if job_id is None:
        return jsonify({'error': 'Missing job_id parameter'}), 400
    job = await db_adapter_server.get_job(job_id)
    if job:
        return jsonify(job.as_dict()), 200
    else:
        return jsonify({'error': 'Job not found'}), 404

@app.route('/update_job_iteration', methods=['POST'])
@requires_auth
async def update_job_iteration():
    data = await request.get_json()
    job_id = data.get('job_id')
    new_iteration = data.get('new_iteration')
    if job_id is None or new_iteration is None:
        return jsonify({'error': 'Missing job_id or new_iteration'}), 400
    await db_adapter_server.update_job_iteration(job_id, new_iteration)
    return jsonify({'status': 'success'}), 200

@app.route('/mark_job_as_done', methods=['POST'])
@requires_auth
async def mark_job_as_done():
    data = await request.get_json()
    job_id = data.get('job_id')
    if job_id is None:
        return jsonify({'error': 'Missing job_id'}), 400
    await db_adapter_server.mark_job_as_done(job_id)
    return jsonify({'status': 'success'}), 200

@app.route('/create_task', methods=['POST'])
@requires_auth
async def create_task():
    data = await request.get_json()
    job_id = data.get('job_id')
    subnet_task_id = data.get('subnet_task_id')
    job_iteration = data.get('job_iteration')
    status = TaskStatus[data.get('status')]
    if None in (job_id, subnet_task_id, job_iteration, status):
        return jsonify({'error': 'Missing parameters'}), 400
    task_id = await db_adapter_server.create_task(job_id, subnet_task_id, job_iteration, status)
    return jsonify({'task_id': task_id}), 200

@app.route('/create_job', methods=['POST'])
@requires_auth
async def create_job():
    data = await request.get_json()
    name = data.get('name')
    plugin_id = data.get('plugin_id')
    subnet_id = data.get('subnet_id')
    sot_url = data.get('sot_url')
    iteration = data.get('iteration')
    if None in (name, plugin_id, subnet_id, sot_url, iteration):
        return jsonify({'error': 'Missing parameters'}), 400
    job_id = await db_adapter_server.create_job(name, plugin_id, subnet_id, sot_url, iteration)
    return jsonify({'job_id': job_id}), 200

@app.route('/create_subnet', methods=['POST'])
@requires_auth
async def create_subnet():
    data = await request.get_json()
    address = data.get('address')
    rpc_url = data.get('rpc_url')
    distributor_address = data.get('distributor_address')
    pool_address = data.get('pool_address')
    if None in (address, rpc_url):
        return jsonify({'error': 'Missing parameters'}), 400
    subnet_id = await db_adapter_server.create_subnet(
        address,
        rpc_url,
        distributor_address,
        pool_address
    )
    return jsonify({'subnet_id': subnet_id}), 200

@app.route('/create_plugin', methods=['POST'])
@requires_auth
async def create_plugin():
    data = await request.get_json()
    name = data.get('name')
    code = data.get('code')
    if None in (name, code):
        return jsonify({'error': 'Missing parameters'}), 400
    plugin_id = await db_adapter_server.create_plugin(name, code)
    return jsonify({'plugin_id': plugin_id}), 200

@app.route('/update_time_solved', methods=['POST'])
@requires_auth
async def update_time_solved():
    data = await request.get_json()
    subnet_task_id = data.get('subnet_task_id')
    job_id = data.get('job_id')
    time_solved = data.get('time_solved')
    if subnet_task_id is None or time_solved is None:
        return jsonify({'error': 'Missing parameters'}), 400
    await db_adapter_server.update_time_solved(subnet_task_id, job_id, time_solved)
    return jsonify({'status': 'success'}), 200

@app.route('/update_time_solver_selected', methods=['POST'])
@requires_auth
async def update_time_solver_selected():
    data = await request.get_json()
    subnet_task_id = data.get('subnet_task_id')
    job_id = data.get('job_id')
    time_solver_selected = data.get('time_solver_selected')
    if subnet_task_id is None or time_solver_selected is None:
        return jsonify({'error': 'Missing parameters'}), 400
    await db_adapter_server.update_time_solver_selected(subnet_task_id, job_id, time_solver_selected)
    return jsonify({'status': 'success'}), 200

@app.route('/update_task_status', methods=['POST'])
@requires_auth
async def update_task_status():
    data = await request.get_json()
    subnet_task_id = data.get('subnet_task_id')
    job_id = data.get('job_id')
    status = TaskStatus[data.get('status')]
    result = data.get('result')
    solver_address = data.get('solver_address')
    if subnet_task_id is None or status is None:
        return jsonify({'error': 'Missing parameters'}), 400
    await db_adapter_server.update_task_status(
        subnet_task_id,
        job_id,
        status,
        result,
        solver_address
    )
    return jsonify({'status': 'success'}), 200

@app.route('/create_state_update', methods=['POST'])
@requires_auth
async def create_state_update():
    data = await request.get_json()
    job_id = data.get('job_id')
    state_data = data.get('data')
    if job_id is None or state_data is None:
        return jsonify({'error': 'Missing parameters'}), 400
    state_update_id = await db_adapter_server.create_state_update(job_id, state_data)
    return jsonify({'state_update_id': state_update_id}), 200

@app.route('/get_plugin', methods=['GET'])
async def get_plugin():
    plugin_id = request.args.get('plugin_id', type=int)
    if plugin_id is None:
        return jsonify({'error': 'Missing plugin_id parameter'}), 400
    plugin = await db_adapter_server.get_plugin(plugin_id)
    if plugin:
        return jsonify(plugin.as_dict()), 200
    else:
        return jsonify({'error': 'Plugin not found'}), 404

@app.route('/get_subnet_using_address', methods=['GET'])
async def get_subnet_using_address():
    address = request.args.get('address')
    if address is None:
        return jsonify({'error': 'Missing address parameter'}), 400
    subnet = await db_adapter_server.get_subnet_using_address(address)
    if subnet:
        return jsonify(subnet.as_dict()), 200
    else:
        return jsonify({'error': 'Subnet not found'}), 404

@app.route('/get_task', methods=['GET'])
async def get_task():
    subnet_task_id = request.args.get('subnet_task_id', type=int)
    subnet_id = request.args.get('subnet_id', type=int)
    if subnet_task_id is None or subnet_id is None:
        return jsonify({'error': 'Missing parameters'}), 400
    task = await db_adapter_server.get_task(subnet_task_id, subnet_id)
    if task:
        return jsonify(task.as_dict()), 200
    else:
        return jsonify({'error': 'Task not found'}), 404

@app.route('/get_tasks_for_job', methods=['GET'])
async def get_tasks_for_job():
    try:
        job_id = request.args.get('job_id', type=int)
        offset = request.args.get('offset', default=0, type=int)
        limit = request.args.get('limit', default=20, type=int)

        if job_id is None:
            return jsonify({'error': 'Missing job_id parameter'}), 400

        tasks = await db_adapter_server.get_tasks_with_pagination_for_job(job_id, offset, limit)
        if tasks:
            return jsonify([task.as_dict() for task in tasks]), 200
        else:
            return jsonify({'error': 'No tasks found for the given job'}), 404
    except Exception as e:
        logger.error(f"Error fetching tasks for job {job_id}: {e}")
        return jsonify({'error': 'Failed to fetch tasks', 'details': str(e)}), 500

@app.route('/get_task_count_for_job', methods=['GET'])
async def get_task_count_for_job():
    """
    Retrieve the total number of tasks for a specific job.
    """
    job_id = request.args.get('job_id', type=int)
    if job_id is None:
        return jsonify({'error': 'Missing job_id parameter'}), 400

    task_count = await db_adapter_server.get_task_count_for_job(job_id)
    return jsonify(task_count), 200

@app.route('/get_task_count_by_status_for_job', methods=['GET'])
async def get_task_count_by_status_for_job():
    """
    Retrieve the number of tasks for a specific job with a given list of statuses.
    """
    job_id = request.args.get('job_id', type=int)
    status_list = request.args.getlist('statuses')

    if job_id is None or not status_list:
        return jsonify({'error': 'Missing job_id or statuses parameter'}), 400

    try:
        statuses = [TaskStatus[status] for status in status_list]
        task_count_by_status = await db_adapter_server.get_task_count_by_status_for_job(job_id, statuses)
        return jsonify(task_count_by_status), 200
    except KeyError as e:
        return jsonify({'error': f'Invalid TaskStatus value: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error fetching task count for job {job_id}: {e}")
        return jsonify({'error': 'Failed to fetch task count', 'details': str(e)}), 500

@app.route('/get_perm', methods=['GET'])
async def get_perm():
    address = request.args.get('address')
    perm = request.args.get('perm', type=int)
    if address is None or perm is None:
        return jsonify({'error': 'Missing parameters'}), 400
    perm = await db_adapter_server.get_perm(address, perm)
    if perm:
        return jsonify({'perm': perm.as_dict()}), 200
    else:
        return jsonify({'error': 'Permission not found'}), 404

@app.route('/set_last_nonce', methods=['POST'])
@requires_auth
async def set_last_nonce():
    data = await request.get_json()
    address = data.get('address')
    perm = data.get('perm')
    last_nonce = data.get('last_nonce')
    if None in (address, perm, last_nonce):
        return jsonify({'error': 'Missing parameters'}), 400
    await db_adapter_server.set_last_nonce(address, perm, last_nonce)
    return jsonify({'status': 'success'}), 200

@app.route('/get_sot', methods=['GET'])
async def get_sot():
    id = request.args.get('id', type=int)
    if id is None:
        return jsonify({'error': 'Missing id parameter'}), 400
    sot = await db_adapter_server.get_sot(id)
    if sot:
        return jsonify(sot.as_dict()), 200
    else:
        return jsonify({'error': 'SOT not found'}), 404

@app.route('/create_perm', methods=['POST'])
@requires_auth
async def create_perm():
    data = await request.get_json()
    address = data.get('address')
    perm = data.get('perm')
    if None in (address, perm):
        return jsonify({'error': 'Missing parameters'}), 400
    perm_id = await db_adapter_server.create_perm(address, perm)
    return jsonify({'perm_id': perm_id}), 200

@app.route('/create_perm_description', methods=['POST'])
@requires_auth
async def create_perm_description():
    data = await request.get_json()
    perm_type = PermType[data.get('perm_type')]

    perm_description_id = await db_adapter_server.create_perm_description(perm_type)
    return jsonify({'perm_description_id': perm_description_id}), 200

@app.route('/create_sot', methods=['POST'])
@requires_auth
async def create_sot():
    data = await request.get_json()
    job_id = data.get('job_id')
    url = data.get('url')
    if None in (job_id, url):
        return jsonify({'error': 'Missing parameters'}), 400
    sot_id = await db_adapter_server.create_sot(job_id, url)
    return jsonify({'sot_id': sot_id}), 200

@app.route('/get_sot_by_job_id', methods=['GET'])
async def get_sot_by_job_id():
    job_id = request.args.get('job_id', type=int)
    if job_id is None:
        return jsonify({'error': 'Missing job_id parameter'}), 400
    sot = await db_adapter_server.get_sot_by_job_id(job_id)
    if sot:
        return jsonify(sot.as_dict()), 200
    else:
        return jsonify({'error': 'SOT not found'}), 404

@app.route('/get_total_state_updates_for_job', methods=['GET'])
async def get_total_state_updates_for_job():
    job_id = request.args.get('job_id', type=int)
    if job_id is None:
        return jsonify({'error': 'Missing job_id parameter'}), 400

    try:
        total_state_updates = await db_adapter_server.get_total_state_updates_for_job(job_id)
        return jsonify(total_state_updates), 200
    except Exception as e:
        logger.error(f"Error fetching total state updates for job {job_id}: {e}")
        return jsonify({'error': 'Failed to fetch state updates', 'details': str(e)}), 500

@app.route('/get_last_task_with_status', methods=['GET'])
async def get_last_task_with_status():
    job_id = request.args.get('job_id', type=int)
    status_list = request.args.getlist('statuses')

    if job_id is None or not status_list:
        return jsonify({'error': 'Missing job_id or statuses parameter'}), 400

    try:
        statuses = [TaskStatus[status] for status in status_list]
        last_task = await db_adapter_server.get_last_task_with_status(job_id, statuses)
        if last_task:
            return jsonify(last_task.as_dict()), 200
        else:
            return jsonify({'error': 'No task found matching the given statuses'}), 404
    except KeyError as e:
        return jsonify({'error': f'Invalid TaskStatus value: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error fetching last task with status for job {job_id}: {e}")
        return jsonify({'error': 'Failed to fetch task', 'details': str(e)}), 500

if __name__ == "__main__":
    import argparse
    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    parser = argparse.ArgumentParser(description="Database Server")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind the server')
    parser.add_argument('--perm', type=int, default=0, help='Permission ID')
    parser.add_argument('--root_wallet', type=str, default='0x0', help='Root wallet address')
    args = parser.parse_args()

    perm_modify_db = args.perm
    
    asyncio.run(db_adapter_server.create_perm(args.root_wallet, perm_modify_db))

    config = Config()
    config.bind = [f'{args.host}:{args.port}']

    logger.info(f"Starting Database Server on {args.host}:{args.port}...")
    logger.info(f"Permission ID: {args.perm}")
    logger.info(f"Root wallet address: {args.root_wallet}")
    asyncio.run(serve(app, config))
