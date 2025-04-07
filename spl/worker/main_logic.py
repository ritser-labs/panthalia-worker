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
from ..models import OrderType
from .worker_tag import get_worker_tag

def on_shutdown():
    set_shutdown_requested(True)
    logger.info("Shutdown signal received. Initiating graceful shutdown.")

async def graceful_shutdown():
    logger.info("Starting graceful shutdown process.")

    worker_tag = get_worker_tag()
    try:
        orders = await db_adapter.get_orders_for_user(offset=0, limit=1000)
    except Exception as e:
        logger.error(f"Error fetching orders for shutdown: {e}")
        orders = []

    for order in orders:
        if (order.worker_tag == worker_tag 
            and order.order_type == OrderType.Ask.name 
            and not order.ask_task_id):
            try:
                await db_adapter.delete_order(order.id)
                logger.info(f"Cancelled unmatched ask order {order.id}")
            except Exception as e:
                logger.error(f"Error cancelling order {order.id}: {e}")

    while True:
        assigned_tasks = await db_adapter.get_assigned_tasks(
            args.subnet_id,
            worker_tag=worker_tag
        ) or []
        async with concurrent_tasks_counter_lock:
            in_flight = concurrent_tasks_counter

        if not assigned_tasks and in_flight == 0:
            break

        logger.info(
            f"Waiting for {len(assigned_tasks)} tasks and {in_flight} in-flight "
            "tasks to complete before shutdown..."
        )
        await asyncio.sleep(1)

    logger.info("All in-flight tasks completed. Shutdown complete.")

async def main():
    logger.info("Starting main process")
    
    os.environ["DOCKER_ENGINE_URL"] = args.docker_engine_url
    
    loop = asyncio.get_running_loop()
    if threading.current_thread() is threading.main_thread():
        try:
            loop.add_signal_handler(signal.SIGINT, on_shutdown)
            loop.add_signal_handler(signal.SIGTERM, on_shutdown)
        except NotImplementedError:
            logger.warning("Signal handlers are not supported on this platform.")
    else:
        logger.info("Running in non-main thread; signal handlers are registered in the main thread (via signal.signal).")
    
    if args.gui:
        subnet_id = get_subnet_id()
        if subnet_id is None:
            logger.error("No Subnet ID configured in GUI settings. Please set it.")
            return
    else:
        subnet_id = args.subnet_id
    args.subnet_id = subnet_id

    # Ensure DB connectivity and proper configuration before starting main loop.
    while not (await is_shutdown_requested()):
        new_config_data = load_config()
        new_db_url = new_config_data.get("db_url", args.db_url)
        new_private_key = new_config_data.get("private_key", args.private_key)
        db_adapter.base_url = new_db_url.rstrip('/')
        db_adapter.private_key = new_private_key

        if not (await verify_db_connection_and_auth()):
            logger.error("Failed to connect/authenticate to DB. Retrying in 3 seconds...")
            await asyncio.sleep(3)
            continue

        subnet_obj = await db_adapter.get_subnet(subnet_id)
        if subnet_obj is None:
            logger.error(f"Subnet {subnet_id} not found in DB. Retrying in 3 seconds...")
            await asyncio.sleep(3)
            continue

        break

    if await is_shutdown_requested():
        logger.info("Shutdown requested before main loop start. Exiting.")
        return

    set_connected_once(True)
    await ensure_docker_image(subnet_obj.docker_image)
    logger.info("Docker image ensured upon worker startup.")

    logger.info("Starting tensor synchronization...")
    # Use a global set to permanently track tasks that have been picked up.
    picked_tasks = set()
    last_loop_time = time.time()
    my_worker_tag = get_worker_tag()
    logger.info(f"Worker tag: {my_worker_tag}")

    async def handle_and_cleanup(task, time_invoked):
        try:
            await handle_task(task, time_invoked)
        finally:
            # Do not remove task.id from picked_tasks; once picked, it should never be re-processed.
            pass
    num_deposit_fails = 0
    while not (await is_shutdown_requested()):
        logger.debug(f"Loop interval: {time.time() - last_loop_time:.2f}s")
        last_loop_time = time.time()

        try:
            await deposit_stake()
            num_deposit_fails = 0
        except Exception as e:
            logger.error(f"Error depositing stake: {e}")
            num_deposit_fails += 1
            if num_deposit_fails >= 3:
                logger.error("Too many deposit failures. Exiting.")
                break
            

        assigned_tasks = await db_adapter.get_assigned_tasks(args.subnet_id, worker_tag=my_worker_tag)
        if assigned_tasks is None:
            logger.error("Failed to fetch assigned tasks. Retrying in 3 seconds...")
            await asyncio.sleep(3)
            continue
        for task in assigned_tasks:
            # If this task has not been picked up before, add it to the permanent set and process it.
            if task.id not in picked_tasks:
                picked_tasks.add(task.id)
                asyncio.create_task(handle_and_cleanup(task, time.time()))

        await asyncio.sleep(args.poll_interval)

    logger.info("Shutdown requested: breaking main loop.")
    await graceful_shutdown()

if __name__ == "__main__":
    asyncio.run(main())
