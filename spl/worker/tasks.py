# spl/worker/tasks.py
import time
import json
import logging
import asyncio
from dataclasses import dataclass
from collections import defaultdict
from datetime import timezone

from .config import args
from .queue import TaskQueue
from .db_client import db_adapter
from .uploads import upload_tensor
from ..common import download_file, TENSOR_NAME
from ..models import OrderType
from .queued_lock import AsyncQueuedLock
from ..plugins.manager import get_plugin

logger = logging.getLogger(__name__)

MAX_WORKER_TASK_RETRIES = 3

concurrent_tasks_counter = 0
concurrent_tasks_counter_lock = asyncio.Lock()
task_start_times = {}
task_start_times_lock = asyncio.Lock()
last_handle_event_timestamp = None
last_handle_event_timestamp_lock = asyncio.Lock()

task_processing_lock = AsyncQueuedLock()
upload_lock = AsyncQueuedLock()

task_queue = TaskQueue()

async def get_ask_price():
    return 1

async def deposit_stake():
    global concurrent_tasks_counter
    if (await task_queue.queue_length() + concurrent_tasks_counter) > args.max_tasks_handling:
        logger.debug("Too many tasks being processed. Not depositing more stakes.")
        return
    
    num_orders = await db_adapter.get_num_orders(args.subnet_id, OrderType.Ask.name, False)
    logger.info(f"Current number of stakes: {num_orders}")
    
    for _ in range(args.max_stakes - num_orders):
        price = await get_ask_price()
        await db_adapter.create_order(None, args.subnet_id, OrderType.Ask.name, price, None)

async def handle_task(task, time_invoked):
    global last_handle_event_timestamp

    current_time = time.time()
    logger.debug(f'Time since invocation: {current_time - time_invoked:.2f} seconds')
    async with last_handle_event_timestamp_lock:
        if last_handle_event_timestamp is not None:
            time_since_last_event = current_time - last_handle_event_timestamp
            logger.debug(f"Time since last handle_event call: {time_since_last_event:.2f} seconds")
        last_handle_event_timestamp = current_time

    logger.info(f"Received event for task id {task.id}")
    logger.debug(f"Task params: {task.params}")
    task_params = json.loads(task.params)

    logger.debug(f"Adding task to queue with ID: {task.id}")
    job_db = await db_adapter.get_job(task.job_id)

    task_queue_obj = {
        'task_id': task.id,
        'plugin_id': job_db.plugin_id,
        'sot_url': job_db.sot_url,
        'task_params': task_params,
        'time_solver_selected': task.time_solver_selected.timestamp()
    }

    await task_queue.add_task(task_queue_obj)
    await get_plugin(task_queue_obj['plugin_id'], db_adapter)

    time_since_change = time.time() - task_queue_obj['time_solver_selected']
    logger.debug(f"Time since solver selection: {time_since_change} seconds")

    async with task_start_times_lock:
        task_start_times[task.id] = time.time()
    await process_tasks()

async def process_tasks():
    global task_queue, concurrent_tasks_counter
    retry_attempt = 0
    task_success = False

    if await task_queue.queue_length() == 0:
        logger.debug("No tasks in the queue to process.")
        return

    async with concurrent_tasks_counter_lock:
        concurrent_tasks_counter += 1
    try:
        next_task = await task_queue.get_next_task()

        while retry_attempt < MAX_WORKER_TASK_RETRIES and not task_success:
            try:
                if not next_task:
                    logger.debug("No tasks in the queue to process.")
                    return

                task_id = next_task['task_id']
                task_params = next_task['task_params']
                sot_url = next_task['sot_url']
                time_solver_selected = next_task['time_solver_selected']

                logger.debug(f"{task_id}: Processing task with params: {task_params}")

                # First, predownload data required by the adapter
                predownloaded_data = await download_file(task_params['input_url'])
                # Acquire processing lock and run execute_task
                await task_processing_lock.acquire(priority=time_solver_selected)
                try:
                    plugin = await get_plugin(next_task['plugin_id'], db_adapter)
                    result = await plugin.call_submodule(
                        'model_adapter', 'execute_task',
                        TENSOR_NAME,
                        sot_url,
                        task_params,
                        predownloaded_data
                    )
                    version_number, updates, loss = result
                    logger.info(f"{task_id}: Updates tensor size: {updates.element_size() * updates.numel() / (1024*1024):.2f} MB")
                finally:
                    await task_processing_lock.release()

                # Acquire upload lock and upload results
                await upload_lock.acquire(priority=time_solver_selected)
                try:
                    upload_start = time.time()
                    upload_res = await upload_results(version_number, updates, loss, sot_url)
                    upload_end = time.time()
                    logger.info(f"{task_id}: Uploaded results in {upload_end - upload_start:.2f} seconds")
                finally:
                    await upload_lock.release()

                # Submit solution
                await submit_solution(task_id, upload_res)

                async with task_start_times_lock:
                    task_start_time = task_start_times.pop(task_id, None)
                if task_start_time:
                    total_time = time.time() - task_start_time
                    logger.info(f"{task_id}: Total time to process: {total_time:.2f}s")

                end_time = time.time()
                logger.info(f"{task_id}: process_tasks() completed in {end_time - (task_start_time if task_start_time else end_time):.2f}s. Concurrent tasks: {concurrent_tasks_counter}")
                task_success = True

            except Exception as e:
                retry_attempt += 1
                logger.error(f"Error processing task (attempt {retry_attempt}): {e}", exc_info=True)
                if retry_attempt >= MAX_WORKER_TASK_RETRIES:
                    logger.error(f"Max retries reached for task {task_id}.")
                    raise
                logger.info(f"Retrying task (attempt {retry_attempt + 1}/{MAX_WORKER_TASK_RETRIES})...")

    finally:
        async with concurrent_tasks_counter_lock:
            concurrent_tasks_counter -= 1

async def submit_solution(task_id, result):
    try:
        logger.info('Submitting solution')
        result_str = json.dumps(result)
        receipt = await db_adapter.submit_task_result(task_id, result_str)
        logger.info(f"Solution submission receipt: {receipt}")
    except Exception as e:
        logger.error(f"Error submitting solution for task {task_id}: {e}")
        raise

async def upload_results(version_number, updates, loss, sot_url):
    grads_url = await upload_tensor(updates, 'grads', sot_url)
    return {
        'grads_url': grads_url,
        'loss': loss,
        'version_number': version_number
    }
