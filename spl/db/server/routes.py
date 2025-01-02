# spl/db/server/routes.py

import logging
import traceback
from functools import wraps
from quart import request, jsonify
from enum import Enum
from inspect import signature
from typing import get_type_hints, Dict, Any, Union, get_origin, get_args
import types

from ...models import TaskStatus, PermType, ServiceType, WithdrawalStatus
from ...auth.api_auth import requires_authentication
from ...auth.server_auth import requires_user_auth
from ...util.enums import str_to_enum

from .app import app, db_adapter, logger, get_perm_modify_db
from .adapter import db_adapter_server

class AuthMethod(Enum):
    NONE = 0
    USER = 1
    KEY = 2

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
            if data is None:
                return jsonify({'error': 'Missing JSON body'}), 400
            missing = [key for key in required_keys if key not in data]
            if missing:
                return jsonify({'error': f'Missing parameters: {", ".join(missing)}'}), 400
            return await f(*args, **kwargs, data=data)
        return wrapper
    return decorator

def convert_to_type(value, expected_type):
    if value is None:
        # Check if expected_type can accept None, i.e. Union[..., None]
        if get_origin(expected_type) is Union or isinstance(expected_type, types.UnionType):
            if type(None) in get_args(expected_type):
                return None
        raise ValueError(f"Cannot convert None to {expected_type}")

    # If it's an Enum, convert by name
    if isinstance(expected_type, type) and issubclass(expected_type, Enum):
        return str_to_enum(expected_type, value)

    # If it's a list type (List[X]), recurse for each item
    if get_origin(expected_type) is list:
        inner_type = expected_type.__args__[0]
        return [convert_to_type(v, inner_type) for v in value]

    # If it's a Union, try each of the subtypes in turn
    if get_origin(expected_type) is Union or isinstance(expected_type, types.UnionType):
        for sub_type in get_args(expected_type):
            try:
                return convert_to_type(value, sub_type)
            except ValueError:
                continue
        raise ValueError(f"Cannot convert {value} to any of {get_args(expected_type)}")

    # Special check for booleans from string
    if expected_type is bool and isinstance(value, str):
        lower_val = value.lower()
        if lower_val in ['true', '1']:
            return True
        elif lower_val in ['false', '0']:
            return False
        else:
            raise ValueError(f"Cannot convert {value} to bool")

    # Otherwise, just cast the value directly
    try:
        return expected_type(value)
    except Exception as e:
        raise ValueError(f"Error converting {value} to {expected_type}: {e}")

def parse_args_with_types(func, args: Dict[str, Any]):
    """
    Utility that inspects the function signature and type hints, then
    converts query/JSON parameters to the right Python types.
    """
    sig = signature(func)
    hints = get_type_hints(func)
    parsed_args = {}
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue
        if param_name in args:
            if param_name in hints:
                expected_type = hints[param_name]
                value = args[param_name]
                parsed_args[param_name] = convert_to_type(value, expected_type)
            else:
                parsed_args[param_name] = args[param_name]
        else:
            # If the function param has a default, we can skip
            if param.default is not param.empty:
                parsed_args[param_name] = param.default
            else:
                raise ValueError(f"Missing required parameter: '{param_name}'")
    return parsed_args

def create_route(
    handler_func,
    method,
    params=None,
    required_keys=None,
    id_key=None,
    auth_method=AuthMethod.NONE,
    is_post=False
):
    """
    Helper that creates a route handler that parses either query params or JSON,
    calls `method`, and returns JSON. 
    """
    def recursive_as_dict(obj):
        if isinstance(obj, dict):
            return {k: recursive_as_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_as_dict(item) for item in obj]
        elif hasattr(obj, 'as_dict') and callable(obj.as_dict):
            return obj.as_dict()
        else:
            return obj

    async def handler(*args, **kwargs):
        try:
            if is_post:
                data = await request.get_json()
                # Ensure any required JSON keys are present
                if required_keys:
                    if data is None:
                        return jsonify({'error': 'Missing JSON body'}), 400
                    missing = [key for key in required_keys if key not in data]
                    if missing:
                        return jsonify({
                            'error': f'Missing parameters: {", ".join(missing)}'
                        }), 400
                parsed_data = parse_args_with_types(method, data)
                result = await method(**parsed_data)
                if id_key:
                    return jsonify({id_key: result}), 200
                return jsonify(result or {'success': True}), 200
            else:
                # For GET, parse query params
                query_params = {p: request.args.get(p) for p in (params or [])}
                parsed_params = parse_args_with_types(method, query_params)
                result = await method(**parsed_params)
                if isinstance(result, list):
                    return jsonify([recursive_as_dict(item) for item in result]), 200
                if isinstance(result, dict):
                    return jsonify(recursive_as_dict(result)), 200
                if result:
                    return jsonify(recursive_as_dict(result)), 200
                return jsonify({'error': f'{method.__name__} not found'}), 404
        except ValueError as ve:
            logging.error(f"Validation error in {method.__name__}: {ve}")
            return jsonify({'error': str(ve)}), 400
        except Exception as e:
            tb = traceback.format_exc()
            logging.error(f"Error in {method.__name__}: {e}\n{tb}")
            return jsonify({'error': str(e)}), 500

    # Apply any required auth
    if auth_method == AuthMethod.USER:
        handler = requires_user_auth_with_adapter(handler)
    elif auth_method == AuthMethod.KEY:
        handler = requires_auth(handler)

    # If it's a POST route, enforce required JSON keys if given
    if is_post:
        if required_keys:
            handler = require_json_keys(*required_keys)(handler)
    else:
        if params:
            handler = require_params(*params)(handler)

    return handle_errors(handler)


# Shortcut: create routes for GET & POST quickly
def create_get_route(entity_name, method, params, auth_method=AuthMethod.NONE):
    return create_route(
        None, method,
        params=params,
        auth_method=auth_method
    )

def create_post_route(method, required_keys, auth_method=AuthMethod.KEY):
    return create_route(
        None, method,
        required_keys=required_keys,
        auth_method=auth_method,
        is_post=True
    )

def create_post_route_return_id(method, required_keys, id_key, auth_method=AuthMethod.KEY):
    return create_route(
        None, method,
        required_keys=required_keys,
        id_key=id_key,
        auth_method=auth_method,
        is_post=True
    )


################################################################
# Basic GET routes for reading Job, Plugin, etc.
################################################################
app.route('/get_job', methods=['GET'], endpoint='get_job_endpoint')(
    create_get_route('Job', db_adapter_server.get_job, ['job_id'])
)
app.route('/get_plugin', methods=['GET'], endpoint='get_plugin_endpoint')(
    create_get_route('Plugin', db_adapter_server.get_plugin, ['plugin_id'])
)
app.route('/get_subnet_using_address', methods=['GET'], endpoint='get_subnet_using_address_endpoint')(
    create_get_route('Subnet', db_adapter_server.get_subnet_using_address, ['address'])
)
app.route('/get_subnet', methods=['GET'], endpoint='get_subnet_endpoint')(
    create_get_route('Subnet', db_adapter_server.get_subnet, ['subnet_id'])
)
app.route('/get_task', methods=['GET'], endpoint='get_task_endpoint')(
    create_get_route('Task', db_adapter_server.get_task, ['task_id'])
)
app.route('/get_assigned_tasks', methods=['GET'], endpoint='get_assigned_tasks_endpoint')(
    create_get_route('Task', db_adapter_server.get_assigned_tasks, ['subnet_id'], auth_method=AuthMethod.USER)
)
app.route('/get_num_orders', methods=['GET'], endpoint='get_num_orders_endpoint')(
    create_get_route('int', db_adapter_server.get_num_orders, ['subnet_id','order_type','matched'], auth_method=AuthMethod.USER)
)
app.route('/get_perm', methods=['GET'], endpoint='get_perm_endpoint')(
    create_get_route('Permission', db_adapter_server.get_perm, ['address','perm'])
)
app.route('/get_sot', methods=['GET'], endpoint='get_sot_endpoint')(
    create_get_route('SOT', db_adapter_server.get_sot, ['id'])
)
app.route('/get_sot_by_job_id', methods=['GET'], endpoint='get_sot_by_job_id_endpoint')(
    create_get_route('SOT', db_adapter_server.get_sot_by_job_id, ['job_id'])
)
app.route('/get_instance_by_service_type', methods=['GET'], endpoint='get_instance_by_service_type_endpoint')(
    create_get_route('Instance', db_adapter_server.get_instance_by_service_type, ['service_type','job_id'])
)
app.route('/get_instances_by_job', methods=['GET'], endpoint='get_instances_by_job_endpoint')(
    create_get_route('Instance', db_adapter_server.get_instances_by_job, ['job_id'])
)
app.route('/get_tasks_for_job', methods=['GET'], endpoint='get_tasks_for_job_endpoint')(
    create_get_route('Task', db_adapter_server.get_tasks_with_pagination_for_job, ['job_id','offset','limit'])
)
app.route('/get_all_instances', methods=['GET'], endpoint='get_all_instances_endpoint')(
    create_get_route('Instance', db_adapter_server.get_all_instances, [])
)
app.route('/get_jobs_without_instances', methods=['GET'], endpoint='get_jobs_without_instances_endpoint')(
    create_get_route('Job', db_adapter_server.get_jobs_without_instances, [])
)
app.route('/get_plugins', methods=['GET'], endpoint='get_plugins_endpoint')(
    create_get_route('Plugin', db_adapter_server.get_plugins, [])
)
app.route('/account_key_from_public_key', methods=['GET'], endpoint='account_key_from_public_key_endpoint')(
    create_get_route('AccountKey', db_adapter_server.account_key_from_public_key, ['public_key'], auth_method=AuthMethod.KEY)
)
app.route('/get_account_keys', methods=['GET'], endpoint='get_account_keys_endpoint')(
    create_get_route('AccountKey', db_adapter_server.get_account_keys, [], auth_method=AuthMethod.USER)
)
app.route('/get_orders_for_user', methods=['GET'], endpoint='get_orders_for_user_endpoint')(
    create_get_route('Order', db_adapter_server.get_orders_for_user, [], auth_method=AuthMethod.USER)
)

# global stats
app.route('/global_stats', methods=['GET'], endpoint='global_stats_endpoint')(
    create_get_route('GlobalStats', db_adapter_server.get_global_stats, [], auth_method=AuthMethod.USER)
)

################################################################
# Additional GET endpoints with custom logic
################################################################
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
        return jsonify({'error': 'No task found'}), 404

@app.route('/get_jobs_in_progress', methods=['GET'])
async def get_jobs_in_progress():
    """
    Returns JSON list of jobs that are either active or have unresolved tasks.
    """
    try:
        jobs = await db_adapter_server.get_jobs_in_progress()
        return jsonify([job.as_dict() for job in jobs]), 200
    except Exception as e:
        app.logger.error(f"Error in /get_jobs_in_progress: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'], endpoint='health_endpoint')
@handle_errors
async def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/get_job_state', methods=['GET'])
@require_params('job_id')
@handle_errors
async def get_job_state():
    """
    Return job.state_json for the given job_id.
    """
    job_id = int(request.args.get('job_id'))
    state = await db_adapter_server.get_job_state(job_id)
    return jsonify(state), 200

@app.route('/update_job_state', methods=['POST'])
@handle_errors
@require_json_keys('job_id', 'new_state')
async def update_job_state(*args, **kwargs):
    """
    Overwrites job.state_json with the new dictionary
    provided in the JSON 'new_state' key.
    """
    data = kwargs['data']
    job_id = data['job_id']
    new_state = data['new_state']  # should be a dict
    if not isinstance(new_state, dict):
        return jsonify({'error': 'new_state must be a dict'}), 400
    await db_adapter_server.update_job_state(job_id, new_state)
    return jsonify({'success': True}), 200

################################################################
# Withdrawals-related endpoints
################################################################
@app.route('/create_withdrawal', methods=['POST'])
@require_json_keys('user_id', 'amount')
@handle_errors
async def create_withdrawal():
    data = await request.get_json()
    user_id = data['user_id']
    amount = float(data['amount'])
    try:
        withdrawal_id = await db_adapter_server.create_withdrawal_request(user_id, amount)
        return jsonify({'withdrawal_id': withdrawal_id}), 200
    except Exception as e:
        logger.error(f"create_withdrawal error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/update_withdrawal_status', methods=['POST'])
@require_json_keys('withdrawal_id', 'new_status')
@handle_errors
async def update_withdrawal_status():
    data = await request.get_json()
    withdrawal_id = data['withdrawal_id']
    new_status_str = data['new_status']
    # Convert string -> enum
    try:
        new_status = WithdrawalStatus[new_status_str.upper()]
    except KeyError:
        return jsonify({'error': f"Invalid withdrawal status: {new_status_str}"}), 400
    await db_adapter_server.update_withdrawal_status(withdrawal_id, new_status)
    return jsonify({'success': True}), 200

################################################################
# Creating resources endpoints (Job, Plugin, Task, Orders, etc.)
################################################################
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
        ['job_id', 'num_tasks', 'price', 'params', 'hold_id'],
        auth_method=AuthMethod.KEY
    )
)

################################################################
# Submit solutions, orders, keys, etc.
################################################################
app.route('/submit_task_result', methods=['POST'], endpoint='submit_task_result_endpoint')(
    create_post_route(
        db_adapter_server.submit_task_result,
        ['task_id', 'result'],
        auth_method=AuthMethod.USER
    )
)
app.route('/create_order', methods=['POST'], endpoint='create_order_endpoint')(
    create_post_route_return_id(
        db_adapter_server.create_order,
        ['task_id', 'subnet_id', 'order_type', 'price', 'hold_id'],
        'order_id',
        auth_method=AuthMethod.USER
    )
)
app.route('/delete_order', methods=['POST'], endpoint='delete_order_endpoint')(
    create_post_route(
        db_adapter_server.delete_order,
        ['order_id'],
        auth_method=AuthMethod.USER
    )
)
app.route('/admin_deposit_account', methods=['POST'], endpoint='admin_deposit_account_endpoint')(
    create_post_route(
        db_adapter_server.admin_deposit_account,
        ['user_id', 'amount'],
        auth_method=AuthMethod.KEY
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

app.route('/get_balance', methods=['GET'], endpoint='get_balance_endpoint')(
    create_get_route(
        'Balance',
        db_adapter_server.get_balance_details_for_user,
        [],
        auth_method=AuthMethod.USER
    )
)

################################################################
# Updating / last nonce / iteration, etc.
################################################################
app.route('/set_last_nonce', methods=['POST'], endpoint='set_last_nonce_endpoint')(
    create_post_route_return_id(db_adapter_server.set_last_nonce, ['address', 'perm', 'last_nonce'], 'perm')
)
app.route('/update_job_iteration', methods=['POST'], endpoint='update_job_iteration_endpoint')(
    create_post_route_return_id(db_adapter_server.update_job_iteration, ['job_id', 'new_iteration'], 'success')
)
app.route('/update_job_sot_url', methods=['POST'], endpoint='update_job_sot_url_endpoint')(
    create_post_route_return_id(db_adapter_server.update_job_sot_url, ['job_id', 'new_sot_url'], 'success')
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
app.route('/finalize_sanity_check', methods=['POST'], endpoint='finalize_sanity_check_endpoint')(
    create_post_route(db_adapter_server.finalize_sanity_check, ['task_id', 'is_valid'], 'success')
)

@app.route('/update_job_active', methods=['POST'], endpoint='update_job_active_endpoint')
@require_json_keys('job_id', 'new_active')
@handle_errors
async def update_job_active():
    data = await request.get_json()
    job_id = data['job_id']
    new_active = bool(data['new_active'])
    await db_adapter_server.update_job_active(job_id, new_active)
    return jsonify({'success': True}), 200


@app.route("/create_stripe_session", methods=["POST"])
async def create_stripe_session_route():
    data = await request.get_json() or {}
    user_id = data.get("user_id")
    raw_amt = data.get("amount")
    if not user_id or raw_amt is None:
        return jsonify({"error": "Missing user_id or amount"}), 400

    try:
        amount = float(raw_amt)
    except ValueError:
        return jsonify({"error": "Invalid amount"}), 400

    db_adapter = get_db_adapter()
    result = await db_adapter.create_stripe_session(user_id, amount)

    if "error" in result:
        status_code = result.get("status_code", 400)
        return jsonify({"error": result["error"]}), status_code

    return jsonify(result), 200


@app.route("/stripe/webhook", methods=["POST"])
async def stripe_webhook_route():
    payload = await request.get_data()
    sig_header = request.headers.get("Stripe-Signature", "")

    db_adapter = get_db_adapter()
    response = await db_adapter.handle_stripe_webhook(payload, sig_header)
    if "error" in response:
        return jsonify({"error": response["error"]}), response.get("status_code", 400)
    return jsonify(response), 200

app.route('/debug_invariant', methods=['GET'], endpoint='debug_invariant_endpoint')(
    create_get_route(
        entity_name='InvariantResult',
        method=db_adapter_server.check_invariant, 
        params=[],            # no query params needed
        auth_method=AuthMethod.NONE  # or AuthMethod.USER/KEY if you want
    )
)
