# spl/worker/main_logic.py
import asyncio
import time
from .config import load_config, args
from .db_client import db_adapter, verify_db_connection_and_auth, set_connected_once, have_connected_once
from .tasks import deposit_stake, handle_task
from .logging_config import logger
from .gui_config import get_subnet_id, get_private_key

from ..plugins.manager import ensure_docker_image

async def main():
    logger.info("Starting main process")

    if args.gui:
        subnet_id = get_subnet_id()
        if subnet_id is None:
            logger.error("No Subnet ID configured in GUI settings. Please set it.")
            return
    else:
        subnet_id = args.subnet_id

    args.subnet_id = subnet_id

    connected_once = False

    # Re-check config after arguments are parsed
    new_config_data = load_config()
    # CHANGED: Use private_key from config if available
    new_db_url = new_config_data.get("db_url", args.db_url)
    new_private_key = new_config_data.get("private_key", args.private_key)

    from .db_client import db_adapter as global_db_adapter
    global_db_adapter.base_url = new_db_url.rstrip('/')
    global_db_adapter.private_key = new_private_key

    while not connected_once:
        if await verify_db_connection_and_auth():
            logger.info("Successfully connected and authenticated to DB.")
            connected_once = True
            set_connected_once(True)
        else:
            logger.error("Failed to connect/auth to DB. Retrying in 3 seconds...")
            await asyncio.sleep(3)

    await ensure_docker_image()
    logger.info("Docker image is ensured upon worker startup.")

    subnet_in_db = await global_db_adapter.get_subnet(subnet_id)

    logger.info("Starting tensor synchronization...")

    processed_tasks = set()
    last_loop_time = time.time()

    while True:
        logger.debug(f'Loop interval: {time.time() - last_loop_time:.2f}s')
        last_loop_time = time.time()

        await deposit_stake()

        assigned_tasks = await global_db_adapter.get_assigned_tasks(subnet_in_db.id)
        for task in assigned_tasks:
            if task.id not in processed_tasks:
                asyncio.create_task(handle_task(task, time.time()))
                processed_tasks.add(task.id)

        await asyncio.sleep(args.poll_interval)
