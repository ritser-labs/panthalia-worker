# spl/worker/db_client.py
import logging
from .config import args
from .logging_config import logger
from ..db.db_adapter_client import DBAdapterClient

db_adapter = DBAdapterClient(args.db_url, args.private_key)

# Global connection state
connected_once = False

def set_connected_once(val: bool):
    global connected_once
    connected_once = val

def have_connected_once():
    return connected_once

async def verify_db_connection_and_auth():
    # Check server health
    health_response = await db_adapter._authenticated_request('GET', '/health')
    if 'error' in health_response:
        logger.error(f"Unable to connect to DB at {db_adapter.base_url}. "
                     f"Check if the URL is correct and the server is running.")
        return False

    # Check authentication by calling a protected endpoint.
    test_tasks = await db_adapter.get_assigned_tasks(args.subnet_id)
    if test_tasks is None:
        logger.error("Authentication failed. Check your private_key or API credentials.")
        return False

    return True
