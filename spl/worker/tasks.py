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

    async with task_start_times_lock:
        task_start_times[task.id] = time.time()

    # Attempt to process the queue
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
    Call DB to store partial result => final=bool
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

                    replicate_sequence = task_params.get("replicate_sequence", [])
                    original_task_id = task_params.get("original_task_id", None)

                    if replicate_sequence and original_task_id is not None:
                        # ------------------------------------------------------------------
                        # REPLICATION PATH: 
                        # replicate ALWAYS references the top solver (original_task_id),
                        # but aggregator versions might be new => we "back-fill" partials
                        # in the top solver's row for each version in replicate_sequence.
                        # ------------------------------------------------------------------
                        logger.info(f"[process_tasks] Task {task_id} => REPLICA for versions={replicate_sequence}")
                        
                        await plugin.call_submodule(
                            'model_adapter',
                            'load_input_tensor',
                            predownloaded_data
                        )
                        del predownloaded_data

                        difference_threshold = 1e-3
                        mismatch_found = False
                        debug_steps_info = {}

                        for idx, ver in enumerate(replicate_sequence):
                            # (A) Download aggregator diffs from the previous version to `ver`.
                            if idx == 0:
                                # Use "force_version_number" param to fetch exactly ver from SOT
                                forced_params = {"force_version_number": ver}
                                local_version = await plugin.call_submodule(
                                    'model_adapter',
                                    'initialize_tensor',
                                    TENSOR_NAME,
                                    sot_url,
                                    forced_params
                                )
                            else:
                                start_v = replicate_sequence[idx-1]
                                diffs, used_end = await get_diffs_in_range(start_v, ver, sot_url)
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

                            # (B) Now compute gradient
                            encoded_grad, loss_val = await plugin.call_submodule(
                                'model_adapter',
                                'execute_step',
                                task_params,
                                idx
                            )

                            # (C) "Back-fill" that aggregator version in the top solver's row
                            #     so nested replicas won't get "missing_original".
                            grads_url = await upload_tensor(encoded_grad, 'grads', sot_url)
                            partial_for_top_solver = {
                                "version_number": ver,
                                "result_url": grads_url,
                                "loss": loss_val
                            }
                            # "final" only on last iteration
                            final_for_parent = (idx == len(replicate_sequence) - 1)
                            await submit_solution(original_task_id, partial_for_top_solver, final_for_parent)

                            # (D) Compare with top solver's partial for version=ver
                            try:
                                original_grad = await fetch_original_step_grad_from_db(
                                    original_task_id, ver, plugin
                                )
                            except Exception as ex:
                                logger.error(f"[replica] Could not fetch top solver grad ver={ver}: {ex}",
                                             exc_info=True)
                                mismatch_found = True
                                debug_steps_info[f"ver_{ver}_diff"] = "missing_original"
                                break

                            new_grad_decoded = await plugin.call_submodule(
                                'model_adapter', 'decode_diff', encoded_grad
                            )
                            diff_norm = (new_grad_decoded - original_grad).norm().item()
                            debug_steps_info[f"ver_{ver}_diff"] = diff_norm
                            logger.info(f"[replica] ver={ver}, difference_norm={diff_norm:.6f}")

                            if diff_norm > difference_threshold:
                                mismatch_found = True
                                break

                        final_replica_status = (not mismatch_found)
                        final_result = {
                            "replica_status": final_replica_status,
                            "debug_info": debug_steps_info
                        }
                        # store final replicate status in *this replicate's* row
                        await submit_solution(task_id, final_result, final=True)

                    elif replicate_sequence:
                        # replicate_sequence set but no original_task_id => fallback
                        logger.warning("[process_tasks] replicate_sequence set but no original_task_id => skipping comparison.")
                        final_result = {"replica_status": False, "error": "No original_task_id provided"}
                        await submit_solution(task_id, final_result, final=True)

                    else:
                        # -------------------------
                        # NORMAL SOLVER PATH
                        # -------------------------
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
                            old_local_version = await fetch_current_sot_version(sot_url)
                            # compute gradient
                            encoded_grad, loss_val = await plugin.call_submodule(
                                'model_adapter',
                                'execute_step',
                                task_params,
                                step_idx
                            )
                            grads_url = await upload_tensor(encoded_grad, 'grads', sot_url)

                            # aggregator => next version
                            new_global_version = await wait_for_version_advance(old_local_version, sot_url)

                            partial_result = {
                                "version_number": new_global_version,
                                "result_url": grads_url,
                                "loss": loss_val
                            }
                            is_final = (step_idx == steps - 1)
                            await submit_solution(task_id, partial_result, final=is_final)

                            # now fetch diffs from old_local_version..new_global_version
                            if new_global_version > old_local_version:
                                new_diffs, used_end_time = await get_diffs_in_range(
                                    start_version=old_local_version,
                                    end_time=new_global_version,
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
                                local_version = max(new_global_version, used_end_time)

                finally:
                    await task_processing_lock.release()

                # Mark success
                async with task_start_times_lock:
                    task_start_time = task_start_times.pop(task_id, None)
                total_time = time.time() - (task_start_time if task_start_time else time.time())
                logger.info(f"{task_id}: Completed entire task in {total_time:.2f}s. "
                            f"Concurrent tasks: {concurrent_tasks_counter}")
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

# -------------------------------------------------------------------------
# fetch_original_step_grad_from_db
# -------------------------------------------------------------------------
async def fetch_original_step_grad_from_db(original_task_id: int,
                                           ver: int,
                                           plugin) -> torch.Tensor:
    """
    1) Load the "original" solver Task from DB (the top solver).
    2) Check task.result JSON for partial_data["version_number"] == `ver`.
    3) Download partial_data["result_url"], decode to a Tensor, and return it.
    Raises Exception if not found or on any error.
    """
    orig_task = await db_adapter.get_task(original_task_id)
    if not orig_task:
        raise RuntimeError(f"Original solver Task {original_task_id} not found in DB.")

    if not orig_task.result:
        raise RuntimeError(f"Original solver Task {original_task_id} has no stored results.")

    partial_dict = orig_task.result
    matched_url = None
    for timestamp_key, partial_data in partial_dict.items():
        if isinstance(partial_data, dict):
            if partial_data.get("version_number") == ver:
                matched_url = partial_data.get("result_url", None)
                break

    if not matched_url:
        raise RuntimeError(
            f"No partial result with version_number={ver} found in top solver task={original_task_id}."
        )

    raw_data = await download_file(matched_url, download_type='tensor', chunk_timeout=300)
    if not isinstance(raw_data, dict):
        raise RuntimeError(f"Original gradient partial not a dict => {matched_url}")

    decoded = await plugin.call_submodule('model_adapter', 'decode_diff', raw_data)
    return decoded
