# server_auth.py

import traceback
import logging
from quart import jsonify, request, g
from functools import wraps
from eth_account import Account
from eth_account.messages import encode_defunct
import json
import time

from .jwt_verification import verify_jwt
from ..db.db_adapter_server import DBAdapterServer  # Adjust import as necessary
from ..models import AccountKey, PermType  # Import necessary models

logger = logging.getLogger(__name__)

EXPIRY_TIME = 10  # Seconds for signature validity

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
    try:
        # Encode the message using EIP-191
        message_encoded = encode_defunct(text=message)
        # Recover the address from the signature
        recovered_address = Account.recover_message(message_encoded, signature=signature)
        logger.debug(f"Recovered address from signature: {recovered_address}")

        # Retrieve the AccountKey using the recovered address
        account_key = await db_adapter.account_key_from_public_key(recovered_address)
        if not account_key:
            logger.error(f"No AccountKey found for address: {recovered_address}")
            return None

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

    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"Error during signature verification: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during signature verification: {e}")
        return None

def requires_user_auth(get_db_adapter):
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
                    g.user = {
                        'user_id': payload.get('sub'),
                        'token_payload': payload
                    }
                    logger.debug(f"Authenticated via JWT: {g.user['user_id']}")

                    # Ensure the user account exists in the database
                    await db_adapter.get_or_create_account(payload.get('sub'))

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
                logger.debug(f"Authenticated via signature: {g.user['user_id']}")

            return await f(*args, **kwargs)
        return wrapper
    return decorator
