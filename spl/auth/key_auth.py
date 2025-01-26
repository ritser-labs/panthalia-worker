# spl/auth/api_auth.py

import functools
import json
import logging
from quart import request, jsonify, g
from eth_account import Account
from eth_account.messages import encode_defunct
import time

EXPIRY_TIME = 10

logger = logging.getLogger(__name__)

async def verify_signature(db_adapter, message: str, signature: str, perm_db_column: int):
    message = encode_defunct(text=message)
    try:
        recovered_address = Account.recover_message(message, signature=signature)
    except Exception as e:
        logger.error(f"Error recovering address: {e}")
        return False

    logger.debug(f"Recovered address: {recovered_address}")
    logger.debug(f"Perm DB column: {perm_db_column}")

    return await db_adapter.get_perm(recovered_address, perm_db_column)

def requires_key_auth(get_db_adapter, get_perm_db):
    def decorator(f):
        @functools.wraps(f)
        async def decorated_function(*args, **kwargs):
            auth_header = request.headers.get('Authorization')
            logger.debug(f"Authorization header: {auth_header}")
            if not auth_header:
                logger.error("Authorization header missing")
                return jsonify({'error': 'Authorization header missing'}), 401

            try:
                message, signature = auth_header.rsplit(':', 1)
            except ValueError:
                logger.error("Invalid Authorization header format")
                return jsonify({'error': 'Invalid Authorization header format'}), 401

            perm_db_column = get_perm_db()
            db_adapter = get_db_adapter()
            perm = await verify_signature(
                db_adapter, message, signature, perm_db_column)
            logging.debug(f"Perm: {perm}")
            if not perm:
                logger.error("Invalid signature")
                return jsonify({'error': 'Invalid signature'}), 403

            # Parse the message to extract the nonce and timestamp
            try:
                message_data = json.loads(message)
                nonce = message_data['nonce']
                timestamp = message_data['timestamp']
                logger.debug(f"Message nonce: {nonce}, timestamp: {timestamp}")
            except (KeyError, json.JSONDecodeError):
                logger.error("Invalid message format")
                return jsonify({'error': 'Invalid message format'}), 401

            # Check if the nonce has been used before
            if nonce == perm.last_nonce:
                logger.error("Nonce already used")
                return jsonify({'error': 'Nonce already used'}), 403
            else:
                await db_adapter.set_last_nonce(perm.address, perm_db_column, nonce)

            # Check if the message has expired (validity period of 10 seconds)
            current_time = int(time.time())
            if current_time - timestamp > EXPIRY_TIME:
                logger.error("Message expired")
                return jsonify({'error': 'Message expired'}), 403
            g.auth_type = 'key'
            return await f(*args, **kwargs)
        return decorated_function
    return decorator
