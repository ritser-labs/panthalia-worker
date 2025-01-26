# server_auth.py

import traceback
import logging
from quart import jsonify, request, g, make_response
from functools import wraps
from eth_account import Account
from eth_account.messages import encode_defunct
import json
import time

from .jwt_verification import verify_jwt
from ..models.enums import PermType
from .key_auth import requires_key_auth

logger = logging.getLogger(__name__)

EXPIRY_TIME = 10  # Seconds for signature validity

class AuthError(Exception):
    pass

async def verify_signature(db_adapter, message, signature):
    """
    Verifies the Ethereum signature and retrieves the associated user_id.

    Args:
        db_adapter: The database adapter to interact with the database.
        message (str): The original message that was signed.
        signature (str): The signature to verify.

    Returns:
        user_id (str) if verification is successful, else None.
    """
    # Encode the message using EIP-191
    message_encoded = encode_defunct(text=message)
    # Recover the address from the signature
    recovered_address = Account.recover_message(message_encoded, signature=signature)
    logger.debug(f"Recovered address from signature: {recovered_address}")

    # Retrieve the AccountKey using the recovered address
    account_key = await db_adapter.account_key_from_public_key(recovered_address)
    if not account_key:
        raise AuthError(f"No AccountKey found for address: {recovered_address}")

    # Parse the message to extract nonce and timestamp
    message_data = json.loads(message)
    timestamp = message_data.get('timestamp')

    if not timestamp:
        logger.error("Message missing nonce or timestamp")
        return None

    # Check if the message has expired
    current_time = int(time.time())
    if current_time - timestamp > EXPIRY_TIME:
        logger.error("Message expired")
        return None

    # Return the associated user_id
    return account_key.user_id

def requires_user_auth(get_db_adapter, require_admin=False):
    """
    Decorator to require user authentication via JWT or Ethereum signature.

    Args:
        get_db_adapter (callable): A function that returns the database adapter.

    Returns:
        Decorated function with authentication enforced.
    """
    def decorator(f):
        @wraps(f)
        async def wrapper(*args, **kwargs):
            authorization_header = request.headers.get('Authorization')
            logger.debug(f"Authorization header: {authorization_header}")

            if not authorization_header:
                logger.error("Missing Authorization header")
                return jsonify({'error': 'Missing Authorization header'}), 401

            db_adapter = get_db_adapter()

            if authorization_header.startswith('Bearer '):
                # Handle JWT authentication
                token = authorization_header[len('Bearer '):]
                try:
                    payload = verify_jwt(token)
                    user_id = payload.get('sub')
                    g.user = {
                        'user_id': user_id,
                        'token_payload': payload
                    }
                    g.auth_type = 'user-jwt'
                    logger.debug(f"Authenticated via JWT: {user_id}")
                except Exception as e:
                    logger.error(f"JWT verification failed: {e}")
                    return jsonify({'error': 'Invalid JWT token'}), 401

            else:
                # Handle Ethereum signature authentication
                try:
                    message, signature = authorization_header.rsplit(':', 1)
                    logger.debug(f"Message: {message}")
                    logger.debug(f"Signature: {signature}")
                except ValueError:
                    logger.error("Invalid Authorization header format for signature authentication")
                    return jsonify({'error': 'Invalid Authorization header format'}), 401

                user_id = await verify_signature(db_adapter, message, signature)
                if not user_id:
                    logger.error("Signature verification failed")
                    return jsonify({'error': 'Invalid signature or authentication expired'}), 403

                g.user = {
                    'user_id': user_id
                }
                g.auth_type = 'user-signature'
                    
            # Ensure the user account exists in the database
            user_obj = await db_adapter.get_or_create_account(user_id)
            if require_admin and not user_obj.is_admin:
                logger.error("User is not authorized")
                return jsonify({'error': 'User is not authorized'}), 403
            logger.debug(f"Authenticated via signature: {user_id}")

            return await f(*args, **kwargs)
        return wrapper
    return decorator

def requires_sot_auth(get_db_adapter):
    def decorator(f):
        @wraps(f)
        async def wrapper(*args, **kwargs):
            # 1) Check Authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                return jsonify({'error': 'Missing Authorization header'}), 401

            # 2) Parse out <message>:<signature> from header
            try:
                message, signature = auth_header.rsplit(':', 1)
            except ValueError:
                return jsonify({'error': 'Invalid Authorization header format'}), 401

            # 3) Check JSON in `message` => must have `timestamp` (avoid replay)
            try:
                msg_obj = json.loads(message)
                timestamp = msg_obj.get('timestamp')
                if not timestamp:
                    return jsonify({'error': 'Missing timestamp in message'}), 400
                if time.time() - timestamp > 10:
                    return jsonify({'error': 'Message expired'}), 403
            except:
                return jsonify({'error': 'Invalid message JSON'}), 401

            # 4) Recover address
            defunct_msg = encode_defunct(text=message)
            try:
                recovered_address = Account.recover_message(defunct_msg, signature=signature)
            except Exception:
                return jsonify({'error': 'Signature invalid'}), 403

            db_adapter = get_db_adapter()

            # 5) Attempt to read job_id / sot_id from query OR from JSON body
            job_id_str = request.args.get('job_id')
            sot_id_str = request.args.get('sot_id')

            if request.method == 'POST':
                data = await request.get_json() or {}
                # if the query param was missing, see if user supplied in JSON
                if job_id_str is None and 'job_id' in data:
                    job_id_str = str(data['job_id'])
                if sot_id_str is None and 'sot_id' in data:
                    sot_id_str = str(data['sot_id'])

            # We’ll parse them carefully to ensure 0 or negative => invalid
            job_id = None
            if job_id_str is not None:
                try:
                    parsed = int(job_id_str)
                    if parsed <= 0:
                        return jsonify({'error': 'job_id must be a positive integer'}), 400
                    job_id = parsed
                except ValueError:
                    return jsonify({'error': 'job_id must be an integer'}), 400

            sot_id = None
            if sot_id_str is not None:
                try:
                    parsed = int(sot_id_str)
                    if parsed <= 0:
                        return jsonify({'error': 'sot_id must be a positive integer'}), 400
                    sot_id = parsed
                except ValueError:
                    return jsonify({'error': 'sot_id must be an integer'}), 400

            # 6) Exactly one of {job_id, sot_id} must be specified
            both = (job_id is not None) and (sot_id is not None)
            none = (job_id is None) and (sot_id is None)
            if both or none:
                return jsonify({
                    'error': 'Must supply exactly one: job_id OR sot_id (not both, not neither).'
                }), 400

            # 7) Fetch SoT object using whichever was provided
            if job_id is not None:
                sot_obj = await db_adapter.get_sot_by_job_id(job_id)
                if not sot_obj:
                    return jsonify({'error': 'No SoT found for this job_id'}), 404
            else:
                sot_obj = await db_adapter.get_sot(sot_id)
                if not sot_obj:
                    return jsonify({'error': 'No SoT found with this sot_id'}), 404

            # 8) Check the SoT’s perm_description => must be type=ModifySot, etc.
            perm_desc_id = sot_obj.perm
            perm_desc = await db_adapter.get_perm_description(perm_desc_id)
            if not perm_desc or perm_desc.perm_type != PermType.ModifySot:
                return jsonify({'error': f'Not a ModifySot perm desc'}), 403

            # also verify restricted_sot_id matches
            if perm_desc.restricted_sot_id != sot_obj.id:
                return jsonify({'error': 'Wrong SoT ID restriction'}), 403

            # 9) Now see if recovered_address has a Perm row => (address, perm_desc_id)
            p = await db_adapter.get_perm(recovered_address.lower(), perm_desc_id)
            if not p:
                return jsonify({'error': 'You do not hold the SOT perm'}), 403

            # Optionally store g.sot = sot_obj, g.job_id = job_id, etc.
            return await f(*args, **kwargs)
        return wrapper
    return decorator

def _unify_response(resp):
    """
    If `resp` is a tuple (response, status_code) or (response, status_code, headers),
    unify it into a single Quart Response object with the appropriate status_code.
    """
    if not isinstance(resp, tuple):
        # Already a single response object or None
        return resp
    
    # resp might have up to 3 items: (response_body, status_code, headers)
    length = len(resp)
    
    if length == 2:
        response_body, code = resp
        if hasattr(response_body, 'status_code'):
            # It's already a Response, just update the code
            unified = response_body
            unified.status_code = code
            return unified
        else:
            # It's probably a string or dict, so create a new Response
            unified = make_response(response_body, code)
            return unified
    
    elif length == 3:
        response_body, code, headers = resp
        unified = make_response(response_body, code)
        if isinstance(headers, dict):
            for k, v in headers.items():
                unified.headers[k] = v
        return unified
    
    # Fallback: return as-is if we can't parse
    return resp

def requires_user_or_key_auth(get_db_adapter, get_perm_db, require_admin=False):
    def decorator(f):
        @wraps(f)
        async def wrapper(*args, **kwargs):
            # 1) Attempt user authentication
            try:
                user_auth_func = requires_user_auth(get_db_adapter, require_admin)(f)
                user_resp = await user_auth_func(*args, **kwargs)
                # Normalize tuple => single Response
                user_resp_unified = _unify_response(user_resp)
                logger.debug(f"User auth response => {user_resp_unified} (code={user_resp_unified.status_code})")
                return user_resp_unified
            except AuthError as e:
                key_auth_func = requires_key_auth(get_db_adapter, get_perm_db)(f)
                key_resp = await key_auth_func(*args, **kwargs)
                key_resp_unified = _unify_response(key_resp)
                return key_resp_unified

            # Otherwise user-auth succeeded (or some other code)
        return wrapper
    return decorator
