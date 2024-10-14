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

async def save_json(file_path, data, file_lock):
    with file_lock:
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(json.dumps(data))

async def load_json(file_path, default, file_lock):
    with file_lock:
        if os.path.exists(file_path):
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                if not content.strip():  # Check if the file is empty
                    logging.error(f"The file {file_path} is empty. Returning default value.")
                    return default
                try:
                    return json.loads(content)  # Try loading the JSON content
                except json.JSONDecodeError as e:
                    logging.error(f"JSONDecodeError in file {file_path}: {e}. Returning default value.")
                    async with aiofiles.open(file_path, 'w') as f:
                        await f.write(json.dumps(default))
                    return default
        else:
            logging.info(f"The file {file_path} does not exist. Saving default value.")
            return default


async def verify_signature(db_adapter, message, signature, perm_db_column):
    message = encode_defunct(text=message)
    recovered_address = Account.recover_message(message, signature=signature)

    logging.debug(f"Recovered address: {recovered_address}")
    logging.debug(f"Perm DB column: {perm_db_column}")

    return await db_adapter.has_perm(recovered_address, perm_db_column)


def requires_authentication(get_db_adapter, get_perm_db):
    def decorator(f):
        @functools.wraps(f)
        async def decorated_function(*args, **kwargs):
            auth_header = request.headers.get('Authorization')
            logging.debug(f"Authorization header: {auth_header}")
            if not auth_header:
                logging.error("Authorization header missing")
                return jsonify({'error': 'Authorization header missing'}), 401

            try:
                message, signature = auth_header.rsplit(':', 1)
            except ValueError:
                logging.error("Invalid Authorization header format")
                return jsonify({'error': 'Invalid Authorization header format'}), 401

            perm_db_column = get_perm_db()
            db_adapter = get_db_adapter()
            perm = await verify_signature(
                db_adapter, message, signature, perm_db_column)
            if not perm:
                logging.error("Invalid signature")
                return jsonify({'error': 'Invalid signature'}), 403

            # Parse the message to extract the nonce and timestamp
            try:
                message_data = json.loads(message)
                nonce = message_data['nonce']
                timestamp = message_data['timestamp']
                logging.debug(f"Message nonce: {nonce}, timestamp: {timestamp}")
            except (KeyError, json.JSONDecodeError):
                logging.error("Invalid message format")
                return jsonify({'error': 'Invalid message format'}), 401

            # Check if the nonce has been used before
            if nonce == perm.last_nonce:
                logging.error("Nonce already used")
                return jsonify({'error': 'Nonce already used'}), 403
            else:
                await db_adapter.set_last_nonce(perm.address, perm_db_column, nonce)
            

            # Check if the message has expired (validity period of 10 seconds)
            current_time = int(time.time())
            if current_time - timestamp > EXPIRY_TIME:
                logging.error("Message expired")
                return jsonify({'error': 'Message expired'}), 403

            return await f(*args, **kwargs)
        return decorated_function
    return decorator