import functools
import requests
import json
import os
import time
import logging
from quart import request, jsonify
from eth_account import Account
from eth_account.messages import encode_defunct
from .db.db_adapter_client import DBAdapterClient
import aiofiles

EXPIRY_TIME = 10

logger = logging.getLogger(__name__)

async def verify_signature(db_adapter, message, signature, perm_db_column):
    message = encode_defunct(text=message)
    recovered_address = Account.recover_message(message, signature=signature)

    logger.debug(f"Recovered address: {recovered_address}")
    logger.debug(f"Perm DB column: {perm_db_column}")

    return await db_adapter.has_perm(recovered_address, perm_db_column)


def requires_authentication(get_db_adapter, get_perm_db):
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

            return await f(*args, **kwargs)
        return decorated_function
    return decorator