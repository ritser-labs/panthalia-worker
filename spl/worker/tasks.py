# file: spl/worker/tasks.py

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
    """
    Calls POST /current_timestamp on the SOT, which returns { "version_number": <int> }.
    If there's an error, returns 0 or raises an exception.
    """
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
async def wait_for_version_advance(
    local_version: int,
    sot_url: str,
    poll_interval: float = 0.1,
    max_attempts: int = 300
) -> int:
    """
    Loops for up to poll_interval*max_attempts seconds, calling fetch_current_sot_version(sot_url).
    If the SOT's version_number becomes > local_version, we return it immediately.
    If it never advances, we return the old local_version after logging a warning.
    """
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
    subnet_db = await db_adapter.get_subnet(args.subnet_id)
    return subnet_db.target_price

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

    # Mark the start time
    async with task_start_times_lock:
        task_start_times[task.id] = time.time()

    # Attempt to process the queue
    await process_tasks()


# -------------------------------------------------------------------------
# get_diffs_in_range (unchanged)
# -------------------------------------------------------------------------
async def get_diffs_in_range(start_version, end_time, sot_url):
    """
    Fetches diffs from the SOT in a version range [start_version+1 .. end_time].
    If end_time is None, the SOT will interpret that as "up to the current version."

    Returns:
      (diffs_list, used_end_time)
    """
    if end_time is not None:
        endpoint = f"{sot_url}/get_diffs_since?from_version={start_version}&end_time={end_time}"
    else:
        endpoint = f"{sot_url}/get_diffs_since?from_version={start_version}"

    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get(endpoint) as resp:
            if resp.status != 200:
                txt = await resp.text()
                logger.warning(f"get_diffs_in_range => status={resp.status}, body={txt}")
                # Return an empty list and the same start_version on error
                return [], start_version

            data = await resp.json()
            diffs_list = data.get("diffs", [])
            used_end_time = data.get("used_end_time", start_version)
            return diffs_list, used_end_time


# -------------------------------------------------------------------------
# submit_solution (unchanged)
# -------------------------------------------------------------------------
async def submit_solution(task_id, result: dict, final: bool):
    """
    Call DB to store partial result => final=bool
    (This triggers the SOT's update_state internally, so no direct call is needed here.)
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
    global task_queue, concurrent_tasks_counter
    retry_attempt = 0
    task_success = False

    if await task_queue.queue_length() == 0:
        logger.debug("No tasks in the queue to process.")
        return

    # concurrency guard
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

                # 2) Acquire processing lock => run the steps
                await task_processing_lock.acquire(priority=time_solver_selected)
                try:
                    plugin = await get_plugin(next_task['plugin_id'], db_adapter)

                    # *** NEW LOGIC => Check for replicate_sequence ***
                    replicate_sequence = task_params.get("replicate_sequence", [])

                    if replicate_sequence:
                        # --------------------------------------------------
                        # *** ADDED *** REPLICATION LOGIC
                        # if replicate_sequence is present, we do a special multi-step approach:
                        #   for i in range(len(replicate_sequence)):
                        #       if i == 0: initialize_tensor(force_version_number=replicate_sequence[0])
                        #       else: fetch diffs from [replicate_sequence[i-1] ... replicate_sequence[i]]
                        #       forward/backward => partial result
                        # --------------------------------------------------
                        logger.info(f"[process_tasks] Task {task_id} => Replication steps for versions={replicate_sequence}")

                        # Load input
                        await plugin.call_submodule('model_adapter', 'load_input_tensor', predownloaded_data)
                        del predownloaded_data

                        local_version = None

                        for idx, ver in enumerate(replicate_sequence):
                            if idx == 0:
                                # force param to version=ver
                                forced_params = {
                                    "force_version_number": ver
                                }
                                lv = await plugin.call_submodule(
                                    'model_adapter',
                                    'initialize_tensor',
                                    TENSOR_NAME,
                                    sot_url,
                                    forced_params  # *** pass the force_version_number
                                )
                                local_version = lv
                            else:
                                # fetch diffs from replicate_sequence[idx-1] up to ver
                                start_v = replicate_sequence[idx-1]
                                # We'll do get_diffs_in_range( start_v, ver ), apply them in a loop
                                diffs, used_end = await get_diffs_in_range(start_v, ver, sot_url)
                                for diff_url in diffs:
                                    full_diff_url = f"{sot_url}{diff_url}"
                                    diff_data = await download_file(full_diff_url, download_type='tensor', chunk_timeout=300)
                                    if not isinstance(diff_data, dict):
                                        logger.warning(f"Downloaded diff is not a dict => skipping. url={diff_url}")
                                        continue

                                    diff_tensor = await plugin.call_submodule('model_adapter', 'decode_diff', diff_data)
                                    await plugin.call_submodule('model_adapter', 'apply_diff', diff_tensor)

                                local_version = ver

                            # Now do forward/backward => partial result
                            # We'll do 1 step => step_idx=idx
                            encoded_grad, loss_val = await plugin.call_submodule(
                                'model_adapter',
                                'execute_step',
                                task_params,
                                idx
                            )

                            # We store partial result => version_number=ver
                            grads_url = await upload_tensor(encoded_grad, 'grads', sot_url)
                            partial_result = {
                                "version_number": local_version,
                                "result_url": grads_url,
                                "loss": loss_val
                            }
                            is_final = (idx == len(replicate_sequence) - 1)
                            await submit_solution(task_id, partial_result, final=is_final)

                    else:
                        # --------------------------------------------------
                        # The normal (non-replicate) path
                        # --------------------------------------------------
                        # plugin: model_adapter
                        version_number = await plugin.call_submodule(
                            'model_adapter',
                            'initialize_tensor',
                            TENSOR_NAME,
                            sot_url,
                            task_params
                        )
                        local_version = version_number

                        # 2b) Load the input
                        await plugin.call_submodule(
                            'model_adapter',
                            'load_input_tensor',
                            predownloaded_data
                        )
                        del predownloaded_data

                        steps = task_params['steps']

                        for step_idx in range(steps):
                            # i) Compute gradient
                            encoded_grad, loss_val = await plugin.call_submodule(
                                'model_adapter',
                                'execute_step',
                                task_params,
                                step_idx
                            )

                            # Re-check from SOT for concurrency safety
                            local_version = await fetch_current_sot_version(sot_url)

                            # ii) Store gradient => triggers SOT update_state
                            grads_url = await upload_tensor(encoded_grad, 'grads', sot_url)

                            partial_result = {
                                "version_number": local_version,
                                "result_url": grads_url,
                                "loss": loss_val
                            }

                            # Submit partial result => SOT update_state
                            is_final = (step_idx == steps - 1)
                            await submit_solution(task_id, partial_result, final=is_final)

                            # iii) Wait for the SOT to finalize next version
                            new_global_version = await wait_for_version_advance(local_version, sot_url)
                            if new_global_version > local_version:
                                logger.debug(f"Local version advanced from {local_version} to {new_global_version}")
                                # get diffs in range
                                new_diffs, used_end_time = await get_diffs_in_range(
                                    start_version=local_version,
                                    end_time=None,   # means "until now"
                                    sot_url=sot_url
                                )
                                logger.debug(
                                    f"Obtained {len(new_diffs)} diffs in range from "
                                    f"version {local_version} to {used_end_time}"
                                )
                                for diff_url in new_diffs:
                                    full_diff_url = f"{sot_url}{diff_url}"
                                    logger.debug(f"Downloading diff: {full_diff_url}")
                                    diff_data = await download_file(
                                        full_diff_url,
                                        download_type='tensor',
                                        chunk_timeout=300
                                    )
                                    if not isinstance(diff_data, dict):
                                        logger.warning(
                                            f"Downloaded diff data from {full_diff_url} is not a dict. Skipping..."
                                        )
                                        continue

                                    diff_tensor = await plugin.call_submodule(
                                        'model_adapter',
                                        'decode_diff',
                                        diff_data
                                    )

                                    logger.info(
                                        f"{task_id}: Step={step_idx} "
                                        f"Fetched diff norm={diff_tensor.norm().item():.6f}, "
                                        f"max_abs={diff_tensor.abs().max().item():.6f}"
                                    )
                                    await plugin.call_submodule(
                                        'model_adapter',
                                        'apply_diff',
                                        diff_tensor
                                    )
                                    local_version += 1

                                # Just to be safe
                                local_version = max(local_version, used_end_time)

                finally:
                    await task_processing_lock.release()

                # Mark success
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
                logger.error(
                    f"Error processing task {task_id} on attempt {retry_attempt}: {e}",
                    exc_info=True
                )
                if retry_attempt >= MAX_WORKER_TASK_RETRIES:
                    logger.error(f"Max retries reached for task {task_id}. Giving up.")
                else:
                    backoff = 2
                    logger.info(f"Retrying task {task_id} in {backoff * retry_attempt}s...")
                    await asyncio.sleep(backoff * retry_attempt)

    finally:
        async with concurrent_tasks_counter_lock:
            concurrent_tasks_counter -= 1
