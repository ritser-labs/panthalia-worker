# spl/db/server/routes/common.py

import logging
import traceback
from functools import wraps
from quart import request, jsonify
from enum import Enum
import json
import uuid
import time
from eth_account import Account
from eth_account.messages import encode_defunct

from ..app import logger, get_perm_modify_db
from ..db_server_instance import db_adapter_server
from ....auth.key_auth import requires_key_auth
from ....auth.user_auth import requires_user_or_key_auth, requires_sot_auth
from ..parse import parse_args_with_types

class AuthMethod(Enum):
    NONE = 0      # No auth
    USER = 1      # user_auth or key_auth
    KEY = 2       # key_auth only
    ADMIN = 3     # user_auth + admin check
    SOT = 4       # SOT auth
    
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_db_adapter():
    return db_adapter_server

def requires_auth(f):
    return requires_key_auth(get_db_adapter, get_perm_modify_db)(f)

def requires_user_auth_with_adapter(f):
    return requires_user_or_key_auth(get_db_adapter, get_perm_modify_db)(f)

def requires_admin_auth_with_adapter(f):
    return requires_user_or_key_auth(get_db_adapter, get_perm_modify_db, require_admin=True)(f)

def requires_sot_auth_with_adapter(f):
    return requires_sot_auth(get_db_adapter)(f)

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
                return jsonify({
                    'error': f'Missing parameters: {", ".join(missing)}'
                }), 400
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
                return jsonify({
                    'error': f'Missing parameters: {", ".join(missing)}'
                }), 400
            return await f(*args, **kwargs, data=data)
        return wrapper
    return decorator

def create_route(method, params=None, required_keys=None, id_key=None,
                 auth_method=AuthMethod.NONE, is_post=False):
    def recursive_as_dict(obj):
        if isinstance(obj, dict):
            return {k: recursive_as_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_as_dict(item) for item in obj]
        elif hasattr(obj, 'as_dict') and callable(obj.as_dict):
            return obj.as_dict()
        else:
            return obj

    @handle_errors
    async def handler(*args, **kwargs):
        if is_post:
            data = await request.get_json()
            parsed_data = parse_args_with_types(method, data or {})
            result = await method(**parsed_data)

            if isinstance(result, bool):
                return jsonify({'success': result}), 200
            elif id_key and isinstance(result, (int, str)):
                return jsonify({id_key: result}), 200
            elif result is None:
                if id_key:
                    return jsonify({id_key: None}), 200
                else:
                    return jsonify({'success': True}), 200
            elif isinstance(result, dict):
                return jsonify(result), 200
            else:
                return jsonify(result), 200
        else:
            query_params = {p: request.args.get(p) for p in (params or [])}
            parsed_params = parse_args_with_types(method, query_params)
            result = await method(**parsed_params)

            if isinstance(result, list):
                return jsonify([recursive_as_dict(item) for item in result]), 200
            if isinstance(result, dict):
                return jsonify(recursive_as_dict(result)), 200
            if result is not None:
                return jsonify(recursive_as_dict(result)), 200
            return jsonify({'error': f'{method.__name__} not found'}), 404

    if auth_method == AuthMethod.ADMIN:
        handler = requires_admin_auth_with_adapter(handler)
    elif auth_method == AuthMethod.USER:
        handler = requires_user_auth_with_adapter(handler)
    elif auth_method == AuthMethod.KEY:
        handler = requires_auth(handler)
    elif auth_method == AuthMethod.SOT:
        handler = requires_sot_auth_with_adapter(handler)

    if is_post and required_keys:
        handler = require_json_keys(*required_keys)(handler)
    else:
        if params:
            handler = require_params(*params)(handler)

    return handler

def create_get_route(method, params=None, auth_method=AuthMethod.NONE):
    return create_route(
        method=method,
        params=params,
        auth_method=auth_method,
        is_post=False
    )

def create_post_route(method, required_keys=None, auth_method=AuthMethod.KEY):
    return create_route(
        method=method,
        required_keys=required_keys,
        auth_method=auth_method,
        is_post=True
    )

def create_post_route_return_id(method, required_keys, id_key, auth_method=AuthMethod.KEY):
    return create_route(
        method=method,
        required_keys=required_keys,
        id_key=id_key,
        auth_method=auth_method,
        is_post=True
    )
