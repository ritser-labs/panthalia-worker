# db_server.py

import asyncio
from quart import Quart, request, jsonify
import logging
from .db_adapter_server import db_adapter_server
from ..api_auth import requires_authentication
from ..models import PermType, TaskStatus, ServiceType
from quart_cors import cors
import os
from functools import wraps

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

def get_perm_modify_db():
    return perm_modify_db

def get_db_adapter():
    return db_adapter_server

def requires_auth(f):
    return requires_authentication(
        get_db_adapter,
        get_perm_modify_db
    )(f)

# Helper decorators to reduce repetitive code
def handle_errors(f):
    @wraps(f)
    async def wrapper(*args, **kwargs):
        try:
            return await f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error: {e}")
            return jsonify({'error': str(e)}), 500
    return wrapper

def require_params(*required_params):
    def decorator(f):
        @wraps(f)
        async def wrapper(*args, **kwargs):
            missing = [p for p in required_params if not request.args.get(p)]
            if missing:
                return jsonify({'error': f'Missing parameters: {", ".join(missing)}'}), 400
            return await f(*args, **kwargs)
        return wrapper
    return decorator

def require_json_keys(*required_keys):
    def decorator(f):
        @wraps(f)
        async def wrapper(*args, **kwargs):
            data = await request.get_json()
            missing = [key for key in required_keys if key not in data]
            if missing:
                return jsonify({'error': f'Missing parameters: {", ".join(missing)}'}), 400
            return await f(*args, **kwargs, data=data)
        return wrapper
    return decorator

# Helper for POST routes returning IDs
def create_post_route_return_id(method, required_keys, id_key):
    @requires_auth
    @require_json_keys(*required_keys)
    @handle_errors
    async def handler(data):
        entity_id = await method(**data)
        return jsonify({id_key: entity_id}), 200
    return handler

# Helper for creating GET routes
def create_get_route(entity_name, method, params):
    async def handler(*args, **kwargs):
        query_params = {p: request.args.get(p) for p in params}
        entity = await method(**query_params)
        
        # Check if the returned entity is a list or a single entity
        if isinstance(entity, list):
            if entity:
                return jsonify([item.as_dict() for item in entity]), 200
            else:
                return jsonify([]), 200
        elif entity:
            return jsonify(entity.as_dict()), 200
        else:
            return jsonify({'error': f'{entity_name} not found'}), 404

    return require_params(*params)(handle_errors(handler))


# Define GET routes
app.route('/get_job', methods=['GET'], endpoint='get_job_endpoint')(create_get_route('Job', db_adapter_server.get_job, ['job_id']))
app.route('/get_plugin', methods=['GET'], endpoint='get_plugin_endpoint')(create_get_route('Plugin', db_adapter_server.get_plugin, ['plugin_id']))
app.route('/get_subnet_using_address', methods=['GET'], endpoint='get_subnet_endpoint')(create_get_route('Subnet', db_adapter_server.get_subnet_using_address, ['address']))
app.route('/get_task', methods=['GET'], endpoint='get_task_endpoint')(create_get_route('Task', db_adapter_server.get_task, ['subnet_task_id', 'subnet_id']))
app.route('/get_perm', methods=['GET'], endpoint='get_perm_endpoint')(create_get_route('Permission', db_adapter_server.get_perm, ['address', 'perm']))
app.route('/get_sot', methods=['GET'], endpoint='get_sot_endpoint')(create_get_route('SOT', db_adapter_server.get_sot, ['id']))
app.route('/get_sot_by_job_id', methods=['GET'], endpoint='get_sot_by_job_id_endpoint')(create_get_route('SOT', db_adapter_server.get_sot_by_job_id, ['job_id']))
app.route('/get_instance_by_service_type', methods=['GET'], endpoint='get_instance_by_service_type_endpoint')(create_get_route('Instance', db_adapter_server.get_instance_by_service_type, ['service_type', 'job_id']))
app.route('/get_instances_by_job', methods=['GET'], endpoint='get_instances_by_job_endpoint')(create_get_route('Instance', db_adapter_server.get_instances_by_job, ['job_id']))
app.route('/get_tasks_for_job', methods=['GET'], endpoint='get_tasks_for_job_endpoint')(create_get_route('Task', db_adapter_server.get_tasks_with_pagination_for_job, ['job_id', 'offset', 'limit']))
app.route('/get_all_instances', methods=['GET'], endpoint='get_all_instances_endpoint')(create_get_route('Instance', db_adapter_server.get_all_instances, []))
app.route('/get_jobs_without_instances', methods=['GET'], endpoint='get_jobs_without_instances_endpoint')(create_get_route('Job', db_adapter_server.get_jobs_without_instances, []))

@app.route('/get_task_count_for_job', methods=['GET'], endpoint='get_task_count_for_job_endpoint')
@require_params('job_id')
@handle_errors
async def get_task_count_for_job():
    job_id = request.args.get('job_id', type=int)
    task_count = await db_adapter_server.get_task_count_for_job(job_id)
    return jsonify(task_count), 200

@app.route('/get_task_count_by_status_for_job', methods=['GET'], endpoint='get_task_count_by_status_for_job_endpoint')
@require_params('job_id')
@handle_errors
async def get_task_count_by_status_for_job():
    job_id = request.args.get('job_id', type=int)
    status_list = request.args.getlist('statuses')
    statuses = [TaskStatus[status] for status in status_list]
    task_count_by_status = await db_adapter_server.get_task_count_by_status_for_job(job_id, statuses)
    return jsonify(task_count_by_status), 200

@app.route('/get_total_state_updates_for_job', methods=['GET'], endpoint='get_total_state_updates_for_job_endpoint')
@require_params('job_id')
@handle_errors
async def get_total_state_updates_for_job():
    job_id = request.args.get('job_id', type=int)
    total_state_updates = await db_adapter_server.get_total_state_updates_for_job(job_id)
    return jsonify(total_state_updates), 200

@app.route('/get_last_task_with_status', methods=['GET'], endpoint='get_last_task_with_status_endpoint')
@require_params('job_id')
@handle_errors
async def get_last_task_with_status():
    job_id = request.args.get('job_id', type=int)
    status_list = request.args.getlist('statuses')
    statuses = [TaskStatus[status] for status in status_list]
    last_task = await db_adapter_server.get_last_task_with_status(job_id, statuses)
    if last_task:
        return jsonify(last_task.as_dict()), 200
    else:
        return jsonify({'error': 'No task found matching the given statuses'}), 404

@app.route('/health', methods=['GET'], endpoint='health_endpoint')
@handle_errors
async def health_check():
    return jsonify({'status': 'healthy'}), 200

# Define POST routes
app.route('/create_job', methods=['POST'], endpoint='create_job_endpoint')(
    create_post_route_return_id(db_adapter_server.create_job, ['name', 'plugin_id', 'subnet_id', 'sot_url', 'iteration'], 'job_id')
)
app.route('/create_plugin', methods=['POST'], endpoint='create_plugin_endpoint')(
    create_post_route_return_id(db_adapter_server.create_plugin, ['name', 'code'], 'plugin_id')
)
app.route('/create_perm', methods=['POST'], endpoint='create_perm_endpoint')(
    create_post_route_return_id(db_adapter_server.create_perm, ['address', 'perm'], 'perm_id')
)
app.route('/create_perm_description', methods=['POST'], endpoint='create_perm_description_endpoint')(
    create_post_route_return_id(db_adapter_server.create_perm_description, ['perm_type'], 'perm_description_id')
)
app.route('/create_task', methods=['POST'], endpoint='create_task_endpoint')(
    create_post_route_return_id(db_adapter_server.create_task, ['job_id', 'subnet_task_id', 'job_iteration', 'status'], 'task_id')
)
app.route('/create_subnet', methods=['POST'], endpoint='create_subnet_endpoint')(
    create_post_route_return_id(db_adapter_server.create_subnet, ['address', 'rpc_url', 'distributor_address', 'pool_address'], 'subnet_id')
)
app.route('/create_state_update', methods=['POST'], endpoint='create_state_update_endpoint')(
    create_post_route_return_id(db_adapter_server.create_state_update, ['job_id', 'data'], 'state_update_id')
)
app.route('/create_instance', methods=['POST'], endpoint='create_instance_endpoint')(
    create_post_route_return_id(db_adapter_server.create_instance, ['name', 'service_type', 'job_id', 'private_key', 'pod_id', 'process_id'], 'instance_id')
)
app.route('/create_sot', methods=['POST'], endpoint='create_sot_endpoint')(
    create_post_route_return_id(db_adapter_server.create_sot, ['job_id', 'url'], 'sot_id')
)

# NEW POST ROUTES FOR MISSED FUNCTIONS
app.route('/set_last_nonce', methods=['POST'], endpoint='set_last_nonce_endpoint')(
    create_post_route_return_id(db_adapter_server.set_last_nonce, ['address', 'perm', 'last_nonce'], 'success')
)
app.route('/update_job_iteration', methods=['POST'], endpoint='update_job_iteration_endpoint')(
    create_post_route_return_id(db_adapter_server.update_job_iteration, ['job_id', 'new_iteration'], 'success')
)
app.route('/mark_job_as_done', methods=['POST'], endpoint='mark_job_as_done_endpoint')(
    create_post_route_return_id(db_adapter_server.mark_job_as_done, ['job_id'], 'success')
)
app.route('/update_time_solved', methods=['POST'], endpoint='update_time_solved_endpoint')(
    create_post_route_return_id(db_adapter_server.update_time_solved, ['subnet_task_id', 'job_id', 'time_solved'], 'success')
)
app.route('/update_time_solver_selected', methods=['POST'], endpoint='update_time_solver_selected_endpoint')(
    create_post_route_return_id(db_adapter_server.update_time_solver_selected, ['subnet_task_id', 'job_id', 'time_solver_selected'], 'success')
)
app.route('/update_task_status', methods=['POST'], endpoint='update_task_status_endpoint')(
    create_post_route_return_id(db_adapter_server.update_task_status, ['subnet_task_id', 'job_id', 'status'], 'success')
)
app.route('/update_instance', methods=['POST'], endpoint='update_instance_endpoint')(
    create_post_route_return_id(db_adapter_server.update_instance, ['instance_id'], 'success')
)

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
