# db_server.py

import asyncio
from quart import Quart, request, jsonify
import logging
from .db_adapter_server import db_adapter_server
from ..auth.api_auth import requires_authentication
from ..models import PermType, TaskStatus, ServiceType
from ..auth.server_auth import requires_user_auth
from quart_cors import cors
import os
from functools import wraps
from enum import Enum
from inspect import signature
from typing import get_type_hints, Dict, Any, Union, get_origin, get_args
from ..util.enums import str_to_enum
import types
import traceback

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

class AuthMethod(Enum):
    NONE = 0
    USER = 1
    KEY = 2


def get_perm_modify_db():
    return perm_modify_db

def get_db_adapter():
    return db_adapter_server

def requires_auth(f):
    return requires_authentication(
        get_db_adapter,
        get_perm_modify_db
    )(f)

def requires_user_auth_with_adapter(f):
    return requires_user_auth(get_db_adapter)(f)


def handle_errors(f):
    @wraps(f)
    async def wrapper(*args, **kwargs):
        try:
            return await f(*args, **kwargs)
        except Exception as e:
            #import traceback
            logger.error(f"Error: {e}")
            #logger.error("".join(traceback.format_exception(type(e), e, e.__traceback__)))
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

def convert_to_type(value, expected_type):
    if value is None:
        # if the value is None and the type is Optional, allow it
        if get_origin(expected_type) is Union or isinstance(expected_type, types.UnionType):
            if type(None) in get_args(expected_type):
                return None
        raise ValueError(f"Cannot convert None to {expected_type}")

    # handle Enums
    if isinstance(expected_type, type) and issubclass(expected_type, Enum):
        return str_to_enum(expected_type, value)

    # handle Lists of Enums or other types
    if get_origin(expected_type) is list:
        inner_type = expected_type.__args__[0]
        return [convert_to_type(v, inner_type) for v in value]

    # handle Optional, Union, or UnionType
    if get_origin(expected_type) is Union or isinstance(expected_type, types.UnionType):
        for sub_type in get_args(expected_type):
            try:
                return convert_to_type(value, sub_type)
            except ValueError:
                continue
        raise ValueError(f"Cannot convert {value} to any of {get_args(expected_type)}")

    # default case: attempt direct conversion
    try:
        return expected_type(value)
    except Exception as e:
        raise ValueError(f"Error converting {value} to {expected_type}: {e}")


def parse_args_with_types(func, args: Dict[str, Any]):
    sig = signature(func)
    hints = get_type_hints(func)
    parsed_args = {}
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue  # Skip 'self' for class methods
        if param_name in args:
            if param_name in hints:
                expected_type = hints[param_name]
                value = args[param_name]
                try:
                    parsed_args[param_name] = convert_to_type(value, expected_type)
                except ValueError as ve:
                    raise ValueError(f"Parameter '{param_name}': {ve}")
            else:
                parsed_args[param_name] = args[param_name]
        else:
            if param.default is not param.empty:
                parsed_args[param_name] = param.default
            else:
                raise ValueError(f"Missing required parameter: '{param_name}'")
    return parsed_args

def create_route(handler_func, method, params=None, required_keys=None, id_key=None, auth_method=AuthMethod.NONE, is_post=False):
    """
    Generalized route creation for GET and POST routes.
    """

    async def handler(*args, **kwargs):
        try:
            if is_post:
                data = await request.get_json()
                if required_keys:
                    missing = [key for key in required_keys if key not in data]
                    if missing:
                        return jsonify({'error': f'Missing parameters: {", ".join(missing)}'}), 400
                parsed_data = parse_args_with_types(method, data)
                result = await method(**parsed_data)
                if id_key:
                    return jsonify({id_key: result}), 200
                return jsonify(result or {'success': True}), 200
            else:
                query_params = {p: request.args.get(p) for p in (params or [])}
                parsed_params = parse_args_with_types(method, query_params)
                result = await method(**parsed_params)
                if isinstance(result, list):
                    return jsonify([item.as_dict() for item in result]), 200 #if result else 404
                if isinstance(result, dict):
                    return jsonify(result), 200
                if result:
                    return jsonify(result.as_dict()), 200
                return jsonify({'error': f'{method.__name__} not found'}), 404
        except ValueError as ve:
            logging.error(f"Validation error in {method.__name__}: {ve}")
            return jsonify({'error': str(ve)}), 400
        except Exception as e:
            tb = traceback.format_exc()
            logging.error(f"Error in {method.__name__}: {e}\n{tb}")
            return jsonify({'error': str(e)}), 500

    if auth_method == AuthMethod.USER:
        handler = requires_user_auth_with_adapter(handler)
    elif auth_method == AuthMethod.KEY:
        handler = requires_auth(handler)

    if is_post:
        if required_keys:
            handler = require_json_keys(*required_keys)(handler)
    else:
        if params:
            handler = require_params(*params)(handler)

    return handle_errors(handler)

def create_get_route(entity_name, method, params, auth_method=AuthMethod.NONE):
    return create_route(None, method, params=params, auth_method=auth_method)

def create_post_route(method, required_keys, auth_method=AuthMethod.KEY):
    return create_route(None, method, required_keys=required_keys, auth_method=auth_method, is_post=True)

def create_post_route_return_id(method, required_keys, id_key, auth_method=AuthMethod.KEY):
    return create_route(None, method, required_keys=required_keys, id_key=id_key, auth_method=auth_method, is_post=True)


# Define GET routes
app.route('/get_job', methods=['GET'], endpoint='get_job_endpoint')(create_get_route('Job', db_adapter_server.get_job, ['job_id']))
app.route('/get_plugin', methods=['GET'], endpoint='get_plugin_endpoint')(create_get_route('Plugin', db_adapter_server.get_plugin, ['plugin_id']))
app.route('/get_subnet_using_address', methods=['GET'], endpoint='get_subnet_using_address_endpoint')(create_get_route('Subnet', db_adapter_server.get_subnet_using_address, ['address']))
app.route('/get_subnet', methods=['GET'], endpoint='get_subnet_endpoint')(create_get_route('Subnet', db_adapter_server.get_subnet, ['subnet_id']))
app.route('/get_task', methods=['GET'], endpoint='get_task_endpoint')(create_get_route('Task', db_adapter_server.get_task, ['task_id']))
app.route('/get_assigned_tasks', methods=['GET'], endpoint='get_assigned_tasks_endpoint')(create_get_route('Task', db_adapter_server.get_assigned_tasks, ['subnet_id'], auth_method=AuthMethod.USER))
app.route('/get_num_orders', methods=['GET'], endpoint='get_num_orders_endpoint')(create_get_route('int', db_adapter_server.get_num_orders, ['subnet_id', 'order_type'], auth_method=AuthMethod.USER))
app.route('/get_perm', methods=['GET'], endpoint='get_perm_endpoint')(create_get_route('Permission', db_adapter_server.get_perm, ['address', 'perm']))
app.route('/get_sot', methods=['GET'], endpoint='get_sot_endpoint')(create_get_route('SOT', db_adapter_server.get_sot, ['id']))
app.route('/get_sot_by_job_id', methods=['GET'], endpoint='get_sot_by_job_id_endpoint')(create_get_route('SOT', db_adapter_server.get_sot_by_job_id, ['job_id']))
app.route('/get_instance_by_service_type', methods=['GET'], endpoint='get_instance_by_service_type_endpoint')(create_get_route('Instance', db_adapter_server.get_instance_by_service_type, ['service_type', 'job_id']))
app.route('/get_instances_by_job', methods=['GET'], endpoint='get_instances_by_job_endpoint')(create_get_route('Instance', db_adapter_server.get_instances_by_job, ['job_id']))
app.route('/get_tasks_for_job', methods=['GET'], endpoint='get_tasks_for_job_endpoint')(create_get_route('Task', db_adapter_server.get_tasks_with_pagination_for_job, ['job_id', 'offset', 'limit']))
app.route('/get_all_instances', methods=['GET'], endpoint='get_all_instances_endpoint')(create_get_route('Instance', db_adapter_server.get_all_instances, []))
app.route('/get_jobs_without_instances', methods=['GET'], endpoint='get_jobs_without_instances_endpoint')(create_get_route('Job', db_adapter_server.get_jobs_without_instances, []))
app.route('/get_plugins', methods=['GET'], endpoint='get_plugins_endpoint')(create_get_route('Plugin', db_adapter_server.get_plugins, []))
app.route('/account_key_from_public_key', methods=['GET'], endpoint='account_key_from_public_key_endpoint')(create_get_route('AccountKey', db_adapter_server.account_key_from_public_key, ['public_key'], auth_method=AuthMethod.KEY))
app.route('/get_account_keys', methods=['GET'], endpoint='get_account_keys_endpoint')(create_get_route('AccountKey', db_adapter_server.get_account_keys, [], auth_method=AuthMethod.USER))

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
    create_post_route_return_id(
        db_adapter_server.create_job,
        ['name', 'plugin_id', 'subnet_id', 'sot_url', 'iteration'],
        'job_id',
        auth_method=AuthMethod.USER
    )
)
app.route('/create_plugin', methods=['POST'], endpoint='create_plugin_endpoint')(
    create_post_route_return_id(
        db_adapter_server.create_plugin,
        ['name', 'code'],
        'plugin_id',
        auth_method=AuthMethod.USER
    )
)
app.route('/create_perm', methods=['POST'], endpoint='create_perm_endpoint')(
    create_post_route_return_id(db_adapter_server.create_perm, ['address', 'perm'], 'perm_id')
)
app.route('/create_perm_description', methods=['POST'], endpoint='create_perm_description_endpoint')(
    create_post_route_return_id(db_adapter_server.create_perm_description, ['perm_type'], 'perm_description_id')
)
# updated create_task endpoint
app.route('/create_task', methods=['POST'], endpoint='create_task_endpoint')(
    create_post_route_return_id(
        db_adapter_server.create_task,
        ['job_id', 'job_iteration', 'status', 'params'],
        'task_id',
        auth_method=AuthMethod.USER
    )
)

app.route('/create_bids_and_tasks', methods=['POST'], endpoint='create_bids_and_tasks_endpoint')(
    create_post_route(
        db_adapter_server.create_bids_and_tasks,
        ['job_id', 'num_tasks', 'price', 'params'],
        auth_method=AuthMethod.KEY
    )
)

# create_order endpoint
app.route('/create_order', methods=['POST'], endpoint='create_order_endpoint')(
    create_post_route_return_id(
        db_adapter_server.create_order,
        ['task_id', 'subnet_id', 'order_type', 'price'],
        'order_id',
        auth_method=AuthMethod.USER
    )
)

# delete_order endpoint
app.route('/delete_order', methods=['POST'], endpoint='delete_order_endpoint')(
    create_post_route(
        db_adapter_server.delete_order,
        ['order_id'],
        auth_method=AuthMethod.USER
    )
)

# deposit_account endpoint
app.route('/deposit_account', methods=['POST'], endpoint='deposit_account_endpoint')(
    create_post_route(
        db_adapter_server.deposit_account,
        ['amount'],
        auth_method=AuthMethod.USER
    )
)

# withdraw_account endpoint
app.route('/withdraw_account', methods=['POST'], endpoint='withdraw_account_endpoint')(
    create_post_route(
        db_adapter_server.withdraw_account,
        ['amount'],
        auth_method=AuthMethod.USER
    )
)
app.route('/create_account_key', methods=['POST'], endpoint='create_account_key_endpoint')(
    create_post_route(db_adapter_server.create_account_key, [], auth_method=AuthMethod.USER)
)
app.route('/admin_create_account_key', methods=['POST'], endpoint='admin_create_account_key_endpoint')(
    create_post_route(db_adapter_server.admin_create_account_key, ['user_id'], auth_method=AuthMethod.KEY)
)
app.route('/delete_account_key', methods=['POST'], endpoint='delete_account_key_endpoint')(
    create_post_route(db_adapter_server.delete_account_key, ['account_key_id'], auth_method=AuthMethod.USER)
)
app.route('/create_subnet', methods=['POST'], endpoint='create_subnet_endpoint')(
    create_post_route_return_id(db_adapter_server.create_subnet, ['dispute_period', 'solve_period', 'stake_multiplier'], 'subnet_id')
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

app.route('/set_last_nonce', methods=['POST'], endpoint='set_last_nonce_endpoint')(
    create_post_route_return_id(db_adapter_server.set_last_nonce, ['address', 'perm', 'last_nonce'], 'success')
)
app.route('/update_job_iteration', methods=['POST'], endpoint='update_job_iteration_endpoint')(
    create_post_route_return_id(db_adapter_server.update_job_iteration, ['job_id', 'new_iteration'], 'success')
)
app.route('/mark_job_as_done', methods=['POST'], endpoint='mark_job_as_done_endpoint')(
    create_post_route_return_id(db_adapter_server.mark_job_as_done, ['job_id'], 'success')
)
app.route('/update_task_status', methods=['POST'], endpoint='update_task_status_endpoint')(
    create_post_route_return_id(db_adapter_server.update_task_status, ['task_id', 'job_id', 'status'], 'success')
)
app.route('/update_instance', methods=['POST'], endpoint='update_instance_endpoint')(
    create_post_route_return_id(db_adapter_server.update_instance, ['instance_id'], 'success')
)
app.route('/update_sot', methods=['POST'], endpoint='update_sot_endpoint')(
    create_post_route_return_id(db_adapter_server.update_sot, ['sot_id', 'url'], 'success')
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
