# spl/worker/main_logic.py
import asyncio
import signal
import time
import os
import threading

from .config import args, load_config
from .db_client import db_adapter, verify_db_connection_and_auth, set_connected_once
from .logging_config import logger
from .gui_config import get_subnet_id, get_private_key
from ..plugins.manager import ensure_docker_image
from .tasks import deposit_stake, handle_task, concurrent_tasks_counter, concurrent_tasks_counter_lock
from .shutdown_flag import is_shutdown_requested, set_shutdown_requested

# Signal handler that sets the shutdown flag
def on_shutdown():
    set_shutdown_requested(True)
    logger.info("Shutdown signal received. Initiating graceful shutdown.")

async def cancel_unmatched_orders():
    """
    Cancel all unmatched orders with proper pagination (limit=25).
    An order is considered unmatched if its 'ask_task_id' is None.
    """
    try:
        limit = 25
        offset = 0
        while True:
            orders = await db_adapter.get_orders_for_user(offset=offset, limit=limit)
            if not orders:
                break

            page_has_unmatched = False
            for order in orders:
                if order.ask_task_id is None:
                    page_has_unmatched = True
                    try:
                        await db_adapter.delete_order(order.id)
                        logger.info(f"Cancelled unmatched order {order.id}")
                    except Exception as e:
                        logger.error(f"Error cancelling order {order.id}: {e}")

            # If any unmatched orders were deleted, restart pagination (offset=0)
            # since deletions may shift orders into the current page.
            if page_has_unmatched:
                offset = 0
            else:
                offset += limit

    except Exception as e:
        logger.error(f"Error in cancel_unmatched_orders: {e}")


async def graceful_shutdown():
    logger.info("Starting graceful shutdown process.")
    await cancel_unmatched_orders()
    # Wait until all in-flight tasks complete
    while True:
        async with concurrent_tasks_counter_lock:
            if concurrent_tasks_counter == 0:
                break
            current = concurrent_tasks_counter
        logger.info(f"Waiting for {current} in-flight tasks to complete...")
        await asyncio.sleep(1)
    logger.info("All in-flight tasks completed. Shutdown complete.")

async def main():
    logger.info("Starting main process")
    
    # Set Docker engine URL from args
    os.environ["DOCKER_ENGINE_URL"] = args.docker_engine_url
    
    # Register signal handlers only if in the main thread.
    loop = asyncio.get_running_loop()
    if threading.current_thread() is threading.main_thread():
        try:
            loop.add_signal_handler(signal.SIGINT, on_shutdown)
            loop.add_signal_handler(signal.SIGTERM, on_shutdown)
        except NotImplementedError:
            logger.warning("Signal handlers are not supported on this platform.")
    else:
        logger.info("Running in non-main thread; signal handlers are registered in the main thread (via signal.signal).")
    
    # Determine subnet_id (from GUI settings if in GUI mode)
    if args.gui:
        subnet_id = get_subnet_id()
        if subnet_id is None:
            logger.error("No Subnet ID configured in GUI settings. Please set it.")
            return
    else:
        subnet_id = args.subnet_id
    args.subnet_id = subnet_id

    connected_once = False

    # Load configuration and update DB adapter settings
    new_config_data = load_config()
    new_db_url = new_config_data.get("db_url", args.db_url)
    new_private_key = new_config_data.get("private_key", args.private_key)
    # Update the DB adapter's base_url and private_key:
    db_adapter.base_url = new_db_url.rstrip('/')
    db_adapter.private_key = new_private_key

    # Try connecting/authenticating with DB until successful or shutdown requested.
    while not connected_once and not (await is_shutdown_requested()):
        if await verify_db_connection_and_auth():
            logger.info("Successfully connected and authenticated to DB.")
            connected_once = True
            set_connected_once(True)
        else:
            logger.error("Failed to connect/auth to DB. Retrying in 3 seconds...")
            await asyncio.sleep(3)

    if await is_shutdown_requested():
        logger.info("Shutdown requested before main loop start. Exiting.")
        return

    # Ensure required Docker image is available.
    await ensure_docker_image()
    logger.info("Docker image ensured upon worker startup.")

    # Retrieve subnet information from the DB
    subnet_in_db = await db_adapter.get_subnet(subnet_id)
    if subnet_in_db is None:
        logger.error(f"Subnet {subnet_id} not found in DB. Exiting.")
        return

    logger.info("Starting tensor synchronization...")
    processed_tasks = set()
    last_loop_time = time.time()

    # Main processing loop: run until a shutdown signal is received.
    while not (await is_shutdown_requested()):
        logger.debug(f"Loop interval: {time.time() - last_loop_time:.2f}s")
        last_loop_time = time.time()

        # Deposit stake only if shutdown has not been requested.
        await deposit_stake()

        assigned_tasks = await db_adapter.get_assigned_tasks(args.subnet_id)
        for task in assigned_tasks:
            if task.id not in processed_tasks:
                asyncio.create_task(handle_task(task, time.time()))
                processed_tasks.add(task.id)

        await asyncio.sleep(args.poll_interval)

    logger.info("Shutdown requested: breaking main loop.")
    await graceful_shutdown()

if __name__ == "__main__":
    asyncio.run(main())
