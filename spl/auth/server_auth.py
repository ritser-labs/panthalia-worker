import traceback
from quart import jsonify, request, g
from .jwt_verification import verify_jwt
from functools import wraps
from ..db.db_adapter_server import db_adapter_server

def requires_user_auth(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        authorization_header = request.headers.get('Authorization')
        if not authorization_header:
            return jsonify({'error': 'Missing Authorization header'}), 401

        parts = authorization_header.split()
        if len(parts) != 2:
            return jsonify({'error': f'Invalid Authorization header format'}), 401
        
        if parts[0].lower() != 'bearer':
            return jsonify({'error': 'Invalid Authorization header type'}), 401
        
        try:
            payload = verify_jwt(parts[1])
            g.user = payload
            await db_adapter_server.get_or_create_account(payload['sub'])
        except Exception as e:
            tb = traceback.format_exc()
            raise Exception(f'Error verifying JWT: {e}\n{tb}')
        
        return await func(*args, **kwargs)
    return wrapper
