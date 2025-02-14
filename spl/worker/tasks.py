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
from ..common import download_file, TENSOR_NAME, NoMoreDataException
from ..models import OrderType
from .queued_lock import AsyncQueuedLock
from ..plugins.manager import get_plugin

import torch
import aiohttp

from .shutdown_flag import is_shutdown_requested  # Import the shutdown flag

# NEW: Import order-state functions
from .order_state import mark_order_pending, mark_order_processing, mark_order_completed

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

# -------------------------------------------------------------------------
# Helper to quickly fetch the SOT's *current* version
# -------------------------------------------------------------------------
async def fetch_current_sot_version(sot_url: str) -> int:
    endpoint = f"{sot_url}/current_timestamp"
    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint) as resp:
            if resp.status != 200:
                txt = await resp.text()
                logger.warning(f"fetch_current_sot_version => status={resp.status}, body={txt}")
                return 0
            data = await resp.json()
            return data.get("version_number", 0)

# -------------------------------------------------------------------------
# Helper that loops briefly until SOT version > local_version
# -------------------------------------------------------------------------
async def wait_for_version_advance(local_version: int, sot_url: str,
                                   poll_interval: float = 0.1,
                                   max_attempts: int = 300) -> int:
    for attempt in range(max_attempts):
        current_sot_version = await fetch_current_sot_version(sot_url)
        if current_sot_version > local_version:
            logger.debug(f"SOT version advanced from {local_version} to {current_sot_version}")
            return current_sot_version
        await asyncio.sleep(poll_interval)

    logger.warning(
        f"wait_for_version_advance: SOT did NOT advance beyond {local_version} "
        f"after {poll_interval * max_attempts:.1f} secs. Continuing anyway."
    )
    return local_version

async def get_ask_price():
    if args.limit_price is None:
        raise ValueError("No limit price configured. Please provide a valid limit price.")
    from ..models.schema import DOLLAR_AMOUNT
    # If the value is a float (e.g. provided via CLI), assume it is in dollars
    # and multiply by DOLLAR_AMOUNT. If it's already an int, assume it's scaled.
    if not isinstance(args.limit_price, int):
        price = int(float(args.limit_price) * DOLLAR_AMOUNT)
    else:
        price = args.limit_price
    return price


async def deposit_stake():
    """
    Create ask orders ("stakes") only if we are not shutting down.
    """
    if await is_shutdown_requested():
        logger.info("Shutdown requested; skipping deposit_stake (no new orders will be created).")
        return

    global concurrent_tasks_counter
    current_queue_length = await task_queue.queue_length()
    async with concurrent_tasks_counter_lock:
        total_in_progress = current_queue_length + concurrent_tasks_counter
    if total_in_progress > args.max_tasks_handling:
        logger.debug("Too many tasks being processed. Not depositing more stakes.")
        return

    num_orders = await db_adapter.get_num_orders(args.subnet_id, OrderType.Ask.name, False)
    logger.info(f"Current number of stakes: {num_orders}")

    # Create new ask orders only if needed
    for _ in range(args.max_stakes - num_orders):
        price = await get_ask_price()
        # Capture the created order's ID (assuming create_order returns the new order ID)
        order_id = await db_adapter.create_order(None, args.subnet_id, OrderType.Ask.name, price, None)
        logger.info(f"Created new stake order {order_id} with price {price}")
        # Mark the order as pending in our internal state
        await mark_order_pending(order_id)

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

    # Mark the order as pending in our internal state (if not already marked)
    await mark_order_pending(task.id)

    # Prepare a queue entry
    job_db = await db_adapter.get_job(task.job_id)
    task_queue_obj = {
        'task_id': task.id,
        'plugin_id': job_db.plugin_id,
        'sot_url': job_db.sot_url,
        'task_params': task_params,
        'time_solver_selected': task.time_solver_selected.timestamp()
    }

    logger.debug(f"Adding task to queue with ID: {task.id}")
    await task_queue.add_task(task_queue_obj)

    # Preload the plugin for speed
    await get_plugin(task_queue_obj['plugin_id'], db_adapter)

    time_since_change = time.time() - task_queue_obj['time_solver_selected']
    logger.debug(f"Time since solver selection: {time_since_change} seconds")

    async with task_start_times_lock:
        task_start_times[task.id] = time.time()

    # Attempt to process the queue (the actual processing logic remains unchanged)
    await process_tasks()

# -------------------------------------------------------------------------
# get_diffs_in_range
# -------------------------------------------------------------------------
async def get_diffs_in_range(start_version, end_time, sot_url):
    endpoint = (
        f"{sot_url}/get_diffs_since?from_version={start_version}&end_time={end_time}"
        if end_time is not None
        else f"{sot_url}/get_diffs_since?from_version={start_version}"
    )

    async with aiohttp.ClientSession() as session:
        async with session.get(endpoint) as resp:
            if resp.status != 200:
                txt = await resp.text()
                logger.warning(f"get_diffs_in_range => status={resp.status}, body={txt}")
                return [], start_version

            data = await resp.json()
            diffs_list = data.get("diffs", [])
            used_end_time = data.get("used_end_time", start_version)
            return diffs_list, used_end_time

# -------------------------------------------------------------------------
# submit_solution
# -------------------------------------------------------------------------
async def submit_solution(task_id, result: dict, final: bool):
    """
    Call DB to store partial result => final=bool.
    This triggers SOT's update_state internally (via _update_sot_with_partial).
    """
    try:
        logger.info(f"Submitting result for task {task_id}, final={final}: {result}")
        result_str = json.dumps(result)
        receipt = await db_adapter.submit_partial_result(task_id, result_str, final=final)
        logger.info(f"DB submission receipt: {receipt}")
    except Exception as e:
        logger.error(f"Error in submit_solution for task {task_id}: {e}", exc_info=True)
        raise

# -------------------------------------------------------------------------
# The main processing loop
# -------------------------------------------------------------------------
async def process_tasks():
    """
    Main loop that processes one queued task at a time.
    """
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
        if not next_task:
            logger.debug("No tasks to process after retrieval.")
            return

        task_id = next_task['task_id']
        task_params = next_task['task_params']
        sot_url = next_task['sot_url']
        time_solver_selected = next_task['time_solver_selected']
        
        # Mark the order as processing in our internal state
        await mark_order_processing(task_id)

        logger.info(f"Processing task {task_id} with params: {task_params}")

        while retry_attempt < MAX_WORKER_TASK_RETRIES and not task_success:
            try:
                logger.debug(
                    f"{task_id}: Attempt {retry_attempt+1}/{MAX_WORKER_TASK_RETRIES}, "
                    f"params: {task_params}"
                )

                # 1) Download the input data
                predownloaded_data = await download_file(task_params['input_url'], chunk_timeout=300)
                if predownloaded_data is None:
                    raise Exception("Predownloaded data is None unexpectedly.")

                # 2) Acquire the task-processing lock
                await task_processing_lock.acquire(priority=time_solver_selected)
                try:
                    plugin = await get_plugin(next_task['plugin_id'], db_adapter)

                    replicate_sequence = task_params.get("replicate_sequence", None)
                    original_task_id = task_params.get("original_task_id", None)

                    if replicate_sequence is not None and original_task_id is not None:
                        logger.info(
                            f"[process_tasks] Task={task_id} => REPLICA. We'll do the same 'steps' loop."
                        )

                        await plugin.call_submodule(
                            'model_adapter',
                            'load_input_tensor',
                            predownloaded_data
                        )
                        del predownloaded_data

                        steps = task_params.get("steps", len(replicate_sequence))

                        difference_threshold = 1e-3
                        mismatch_found = False
                        debug_steps_info = {}

                        local_version = 0
                        for step_idx in range(steps):
                            ver = replicate_sequence[step_idx]

                            if step_idx == 0:
                                forced_params = {"force_version_number": ver}
                                local_version_forced = await plugin.call_submodule(
                                    'model_adapter',
                                    'initialize_tensor',
                                    TENSOR_NAME,
                                    sot_url,
                                    forced_params
                                )
                                local_version = ver
                            else:
                                diffs, used_end = await get_diffs_in_range(local_version, ver, sot_url)
                                for diff_url in diffs:
                                    full_diff_url = f"{sot_url}{diff_url}"
                                    diff_data = await download_file(
                                        full_diff_url, download_type='tensor', chunk_timeout=300
                                    )
                                    if not isinstance(diff_data, dict):
                                        logger.warning(f"Downloaded diff is not a dict => skipping. url={diff_url}")
                                        continue
                                    diff_tensor = await plugin.call_submodule(
                                        'model_adapter', 'decode_diff', diff_data
                                    )
                                    await plugin.call_submodule('model_adapter', 'apply_diff', diff_tensor)
                                local_version = ver

                            encoded_grad, loss_val = await plugin.call_submodule(
                                'model_adapter',
                                'execute_step',
                                task_params,
                                step_idx
                            )
                            try:
                                original_grad = await fetch_original_step_grad_from_db(
                                    original_task_id, ver, plugin
                                )
                            except Exception as ex:
                                logger.error(
                                    f"[replica] Could not fetch top solver grad ver={ver}: {ex}",
                                    exc_info=True
                                )
                                mismatch_found = True
                                debug_steps_info[f"ver_{ver}_diff"] = "missing_original"
                                break

                            new_grad_decoded = await plugin.call_submodule(
                                'model_adapter', 'decode_diff', encoded_grad
                            )
                            norm_new = new_grad_decoded.norm().item()
                            denom = max(norm_new, 1e-12)
                            relative_diff = (new_grad_decoded - original_grad).norm().item() / denom
                            debug_steps_info[f"ver_{ver}_diff"] = relative_diff
                            logger.info(f"[replica] step={step_idx}, ver={ver}, difference_norm={relative_diff:.6f}")

                            if relative_diff > difference_threshold:
                                mismatch_found = True
                                break

                        final_is_original_valid = (not mismatch_found)
                        final_result = {
                            "is_original_valid": final_is_original_valid,
                            "debug_info": debug_steps_info
                        }
                        await submit_solution(task_id, final_result, final=True)

                    elif replicate_sequence:
                        logger.warning(
                            "[process_tasks] replicate_sequence set but no original_task_id => skipping comparison."
                        )
                        final_result = {"is_original_valid": False, "error": "No original_task_id provided"}
                        await submit_solution(task_id, final_result, final=True)

                    else:
                        version_number = await plugin.call_submodule(
                            'model_adapter',
                            'initialize_tensor',
                            TENSOR_NAME,
                            sot_url,
                            task_params
                        )
                        local_version = version_number

                        await plugin.call_submodule(
                            'model_adapter',
                            'load_input_tensor',
                            predownloaded_data
                        )
                        del predownloaded_data

                        steps = task_params['steps']
                        for step_idx in range(steps):
                            logger.info(f"{task_id}: Step={step_idx}")
                            encoded_grad, loss_val = await plugin.call_submodule(
                                'model_adapter',
                                'execute_step',
                                task_params,
                                step_idx
                            )
                            grads_url = await upload_tensor(encoded_grad, 'grads', sot_url)

                            version_before_submission = await fetch_current_sot_version(sot_url)

                            partial_result = {
                                "version_number": local_version,
                                "result_url": grads_url,
                                "loss": loss_val
                            }
                            is_final = (step_idx == steps - 1)
                            await submit_solution(task_id, partial_result, final=is_final)
                            await wait_for_version_advance(
                                version_before_submission,
                                sot_url
                            )

                            new_diffs, end_time = await get_diffs_in_range(
                                start_version=local_version,
                                end_time=None,
                                sot_url=sot_url
                            )
                            for diff_url in new_diffs:
                                full_diff_url = f"{sot_url}{diff_url}"
                                diff_data = await download_file(
                                    full_diff_url, download_type='tensor', chunk_timeout=300
                                )
                                if not isinstance(diff_data, dict):
                                    logger.warning(f"Downloaded diff is not a dict => skip {diff_url}")
                                    continue
                                diff_tensor = await plugin.call_submodule(
                                    'model_adapter', 'decode_diff', diff_data
                                )
                                logger.info(
                                    f"{task_id}: Step={step_idx}, fetched diff norm={diff_tensor.norm().item():.6f}"
                                )
                                await plugin.call_submodule(
                                    'model_adapter', 'apply_diff', diff_tensor
                                )
                            local_version = end_time

                finally:
                    await task_processing_lock.release()

                async with task_start_times_lock:
                    task_start_time = task_start_times.pop(task_id, None)
                total_time = time.time() - (task_start_time if task_start_time else time.time())
                logger.info(
                    f"{task_id}: Completed entire task in {total_time:.2f}s. "
                    f"Concurrent tasks: {concurrent_tasks_counter}"
                )
                task_success = True

            except NoMoreDataException:
                logger.info(f"{task_id}: No more data available. Skipping further attempts.")
                break
            except Exception as e:
                retry_attempt += 1
                logger.error(f"Error processing task {task_id} on attempt {retry_attempt}: {e}",
                             exc_info=True)
                if retry_attempt >= MAX_WORKER_TASK_RETRIES:
                    logger.error(f"Max retries reached for task {task_id}. Giving up.")
                else:
                    backoff = 2
                    logger.info(f"Retrying task {task_id} in {backoff * retry_attempt}s...")
                    await asyncio.sleep(backoff * retry_attempt)

    finally:
        async with concurrent_tasks_counter_lock:
            concurrent_tasks_counter -= 1
        # Mark the order as completed in our internal state
        await mark_order_completed(task_id)
