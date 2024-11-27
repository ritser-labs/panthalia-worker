from quart import jsonify, request, g
from .jwt_verification import verify_jwt
from functools import wraps


def requires_user_auth(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        authorization_header = request.headers.get('Authorization')
        if not authorization_header:
            return jsonify({'error': 'Missing Authorization header'}), 401

        parts = authorization_header.split()
        if len(parts) != 2:
            return jsonify({'error': 'Invalid Authorization header'}), 401
        
        if parts[0].lower() != 'bearer':
            return jsonify({'error': 'Invalid Authorization header'}), 401
        
        try:
            payload = verify_jwt(parts[1])
            g.user = payload
        except Exception as e:
            raise Exception('Access token verification failed')
        
        return func(*args, **kwargs)
    return wrapper

def get_user_id():
    """
    Retrieves the user's unique identifier (sub) from the JWT payload.

    Returns:
        str: The user ID from the JWT payload, or None if not authenticated.
    """
    if not hasattr(g, 'user') or not g.user:
        return None
    return g.user.get('sub')
