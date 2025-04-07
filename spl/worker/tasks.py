# spl/worker/tasks.py

import time
import json
import logging
import asyncio
from collections import defaultdict

from spl.worker.worker_tag import get_worker_tag

from .config import args
from .queue import TaskQueue
from .db_client import db_adapter
from .logging_config import logger
from .gui_config import get_subnet_id, get_private_key
from ..plugins.manager import get_plugin
from .uploads import upload_tensor
from ..common import download_file, TENSOR_NAME, NoMoreDataException
from ..models import OrderType
from .queued_lock import AsyncQueuedLock
from ..util import demo

import torch
import aiohttp

from .shutdown_flag import is_shutdown_requested
from .order_state import mark_order_pending, mark_order_processing, mark_order_completed

logger = logging.getLogger(__name__)

MAX_WORKER_TASK_RETRIES = 3  # Not currently in use for direct task retries, but left for reference.

# We track how many tasks are actively being processed.
concurrent_tasks_counter = 0
concurrent_tasks_counter_lock = asyncio.Lock()

# Track when tasks started (for final reporting).
task_start_times = {}
task_start_times_lock = asyncio.Lock()

# Track the last time we handled a task (for diagnosing event-lag).
last_handle_event_timestamp = None
last_handle_event_timestamp_lock = asyncio.Lock()

# A global map: task_id -> {stage_name -> duration_in_seconds}
# Each call to run_with_timeout() logs times to this map.
task_step_times_map = defaultdict(dict)

# Global concurrency lock
task_processing_lock = AsyncQueuedLock()
task_queue = TaskQueue()

CLOUD_TIMEOUT_BONUS = 160.0  # Extra time for "get_plugin" stage if running in cloud mode.


# --------------------------------------------------------------------------------
# A custom exception used when an individual stage times out.
# --------------------------------------------------------------------------------
class StageTimeoutError(Exception):
    def __init__(self, stage_name, elapsed, partial_times):
        super().__init__()
        self.stage_name = stage_name
        self.elapsed = elapsed
        self.partial_times = partial_times

    def __str__(self):
        return (
            f"Stage '{self.stage_name}' timed out after {self.elapsed:.2f} seconds. "
            f"Partial times so far: {json.dumps(self.partial_times)}"
        )


# --------------------------------------------------------------------------------
# Helper that runs any coroutine with a stage-level timeout, storing partial times
# in `task_step_times_map`. If it times out, raises `StageTimeoutError` containing
# partial times so far.
# --------------------------------------------------------------------------------
async def run_with_timeout(coro, stage_name: str, timeout: float, task_id=None):
    start_t = time.time()
    try:
        logger.debug(f"Running stage '{stage_name}' with timeout {timeout:.2f} seconds.")
        return await asyncio.wait_for(coro, timeout=timeout)

    except asyncio.TimeoutError:
        duration = time.time() - start_t
        logger.error(f"Stage '{stage_name}' timed out after {timeout:.2f} seconds.")
        if task_id is not None:
            task_step_times_map[task_id][stage_name] = duration
            partial_times_copy = dict(task_step_times_map[task_id])
        else:
            partial_times_copy = {}
        raise StageTimeoutError(stage_name, duration, partial_times_copy)

    finally:
        # Even if it didn't time out, record the final stage duration.
        duration = time.time() - start_t
        logger.debug(f"Stage '{stage_name}' completed in {duration:.2f} seconds.")
        if task_id is not None:
            task_step_times_map[task_id][stage_name] = duration


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


async def wait_for_version_advance(
    local_version: int,
    sot_url: str,
    poll_interval: float = 0.1,
    max_attempts: int = 300
) -> int:
    for _ in range(max_attempts):
        current_sot_version = await fetch_current_sot_version(sot_url)
        if current_sot_version > local_version:
            logger.debug(f"SOT version advanced from {local_version} to {current_sot_version}")
            return current_sot_version
        await asyncio.sleep(poll_interval)
    logger.warning(
        f"wait_for_version_advance: SOT did NOT advance beyond {local_version} "
        f"after {poll_interval * max_attempts:.1f}s. Continuing anyway."
    )
    return local_version


async def fetch_original_step_grad_from_db(original_task_id: int, ver: int, plugin) -> torch.Tensor:
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
            f"No partial result with version_number={ver} found in original solver task={original_task_id}."
        )
    raw_data = await download_file(matched_url, download_type='tensor', chunk_timeout=300)
    if not isinstance(raw_data, dict):
        raise RuntimeError(f"Original gradient partial not a dict => {matched_url}")

    return await demo.decode_diff(raw_data)


async def submit_solution(task_id, result: dict, final: bool):
    try:
        logger.info(f"Submitting result for task {task_id}, final={final}: {result}")
        result_str = json.dumps(result)
        receipt = await db_adapter.submit_partial_result(task_id, result_str, final=final)
        logger.info(f"DB submission receipt: {receipt}")
    except Exception as e:
        logger.error(f"Error in submit_solution for task {task_id}: {e}", exc_info=True)
        raise


async def get_diffs_in_range(start_version, end_time, sot_url):
    """
    Download all diffs from start_version up to the given end_time (if specified).
    """
    if end_time is not None:
        endpoint = f"{sot_url}/get_diffs_since?from_version={start_version}&end_time={end_time}"
    else:
        endpoint = f"{sot_url}/get_diffs_since?from_version={start_version}"

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


async def deposit_stake():
    """
    Create enough Ask orders (aka "stakes") to keep us at 'args.max_stakes' total.
    We skip depositing if we are already at capacity or if shutdown is requested.
    """
    if await is_shutdown_requested():
        logger.info("Shutdown requested; skipping deposit_stake.")
        return
    global concurrent_tasks_counter
    current_queue_length = await task_queue.queue_length()
    async with concurrent_tasks_counter_lock:
        total_in_progress = current_queue_length + concurrent_tasks_counter
    if total_in_progress >= args.max_tasks_handling:
        logger.debug("Too many tasks being processed. Not depositing more stakes.")
        return

    worker_tag = get_worker_tag()
    num_orders = await db_adapter.get_num_orders(args.subnet_id, OrderType.Ask.name, False, worker_tag)
    logger.info(f"Current number of stakes: {num_orders}")

    from ..models.schema import DOLLAR_AMOUNT

    def ensure_price_is_int(p):
        if not isinstance(p, int):
            return int(float(p) * DOLLAR_AMOUNT)
        return p

    for _ in range(args.max_stakes - num_orders):
        if args.limit_price is None:
            raise ValueError("No limit price configured. Please provide a valid limit price.")
        limit_price_int = ensure_price_is_int(args.limit_price)
        order_id = await db_adapter.create_order(
            None,
            args.subnet_id,
            OrderType.Ask.name,
            limit_price_int,
            None,
            worker_tag
        )
        logger.info(f"Created new stake order {order_id} with price {limit_price_int}")
        await mark_order_pending(order_id)


async def handle_task(task, time_invoked):
    """
    Called when the DB says we've been assigned a new task. We:
      1. Mark the task as pending,
      2. Add it to our local queue,
      3. Trigger process_tasks() to pick it up if possible.
    """
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
    await mark_order_pending(task.id)

    job_db = await db_adapter.get_job(task.job_id)
    subnet_obj = await db_adapter.get_subnet(args.subnet_id)
    task_queue_obj = {
        'job_id': task.job_id,
        'task_id': task.id,
        'plugin_id': job_db.plugin_id,
        'sot_url': job_db.sot_url,
        'task_params': task_params,
        'time_solver_selected': task.time_solver_selected.timestamp(),
        'task_time': subnet_obj.task_time
    }
    logger.debug(f"Adding task to queue with ID: {task.id}")
    await task_queue.add_task(task_queue_obj)

    # Track when this specific task actually started (for total-time reporting).
    async with task_start_times_lock:
        task_start_times[task.id] = time.time()

    await process_tasks()


async def _process_task_once(task_item):
    """
    Core function that processes a single task from start to finish (already holding the concurrency lock).
    We do NOT handle lock acquisition or retry here. If it fails, we immediately finalize the task.
    """
    import uuid

    task_id = task_item['task_id']
    task_params = task_item['task_params']
    sot_url = task_item['sot_url']

    await mark_order_processing(task_id)
    logger.info(f"Processing task {task_id} with params: {task_params}")

    expected_time = float(task_item.get("task_time", 60))
    # If we're in cloud mode, add some extra time for plugin loading.
    get_plugin_extra = CLOUD_TIMEOUT_BONUS if args.cloud else 0.0

    # We'll treat each stage's time factor here:
    multipliers = {
        "download": 1.0,
        "get_plugin": 1.0,
        "initialize_tensor": 1.0,
        "execute_step": 1.0,
        "upload": 1.0,
        "wait_version": 1.0,
        "get_diffs": 1.0,
        "apply_diff": 1.0,
    }

    # ---------------- Stage: Download Input Data ----------------
    download_timeout = expected_time * multipliers["download"]
    predownloaded_data = await run_with_timeout(
        download_file(task_params['input_url'], chunk_timeout=download_timeout),
        "Download Input Data",
        download_timeout,
        task_id=task_id
    )
    if predownloaded_data is None:
        raise RuntimeError("Predownloaded data is None unexpectedly.")

    # ---------------- Stage: Get Plugin ----------------
    get_plugin_timeout = expected_time * multipliers["get_plugin"] + get_plugin_extra
    import_uuid = str(uuid.uuid4())
    plugin = await run_with_timeout(
        get_plugin(
            job_id=task_item['job_id'],
            db_adapter=db_adapter,
            cloud=args.cloud,
            instance_salt=import_uuid
        ),
        "Get Plugin",
        get_plugin_timeout,
        task_id=task_id
    )

    replicate_sequence = task_params.get("replicate_sequence")
    original_task_id = task_params.get("original_task_id")

    # --------------------------------------------------------------------------
    #                           REPLICATION BRANCH
    # --------------------------------------------------------------------------
    if replicate_sequence is not None and original_task_id is not None:
        orig_task = await db_adapter.get_task(original_task_id)
        if not orig_task or not orig_task.result:
            final_result = {
                "is_original_valid": False,
                "error": "Original task missing or has no result",
                "success": False
            }
            await submit_solution(task_id, final_result, final=True)
            return

        # If the original solver ended with success=False, skip mismatch checks.
        for _, partial_data in orig_task.result.items():
            if isinstance(partial_data, dict) and partial_data.get("final") is True:
                if partial_data.get("success") is False:
                    skip_result = {
                        "is_original_valid": True,
                        "explain": "Original had success=False => skipping replicate mismatch check",
                        "success": True
                    }
                    await submit_solution(task_id, skip_result, final=True)
                    return
                break

        init_timeout = expected_time * multipliers["initialize_tensor"]
        await run_with_timeout(
            plugin.call_submodule('model_adapter', 'load_input_tensor', predownloaded_data),
            "Load Input Tensor (Replication)",
            init_timeout,
            task_id=task_id
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
                await run_with_timeout(
                    plugin.call_submodule(
                        'model_adapter',
                        'initialize_tensor',
                        TENSOR_NAME,
                        sot_url,
                        forced_params
                    ),
                    f"Initialize Tensor for version {ver}",
                    init_timeout,
                    task_id=task_id
                )
                local_version = ver
            else:
                get_diffs_timeout = expected_time * multipliers["get_diffs"]
                diffs, used_end = await run_with_timeout(
                    get_diffs_in_range(local_version, ver, sot_url),
                    f"Get Diffs from {local_version} to {ver}",
                    get_diffs_timeout,
                    task_id=task_id
                )
                for diff_url in diffs:
                    full_diff_url = f"{sot_url}{diff_url}"
                    diff_dl_timeout = expected_time * multipliers["download"]
                    diff_data = await run_with_timeout(
                        download_file(full_diff_url, download_type='tensor', chunk_timeout=diff_dl_timeout),
                        f"Download Diff {diff_url}",
                        diff_dl_timeout,
                        task_id=task_id
                    )
                    if not isinstance(diff_data, dict):
                        logger.warning(f"Downloaded diff is not a dict; skipping: {diff_url}")
                        continue
                    apply_diff_timeout = expected_time * multipliers["apply_diff"]
                    await run_with_timeout(
                        plugin.call_submodule('model_adapter', 'decode_and_apply_diff', diff_data),
                        f"Apply Diff {diff_url}",
                        apply_diff_timeout,
                        task_id=task_id
                    )
                local_version = ver

            exec_timeout = expected_time * multipliers["execute_step"]
            encoded_grad, loss_val = await run_with_timeout(
                plugin.call_submodule('model_adapter', 'execute_step', task_params, step_idx),
                f"Execute Step {step_idx}",
                exec_timeout,
                task_id=task_id
            )

            try:
                original_grad = await fetch_original_step_grad_from_db(original_task_id, ver, plugin)
            except Exception as ex:
                logger.error(f"[replica] Could not fetch original grad for version {ver}: {ex}", exc_info=True)
                mismatch_found = True
                debug_steps_info[f"ver_{ver}_diff"] = "missing_original"
                break

            new_grad_decoded = await demo.decode_diff(encoded_grad)
            norm_new = new_grad_decoded.norm().item()
            denom = max(norm_new, 1e-12)
            relative_diff = (new_grad_decoded - original_grad).norm().item() / denom
            debug_steps_info[f"ver_{ver}_diff"] = relative_diff
            logger.info(f"[replica] step={step_idx}, ver={ver}, diff_norm={relative_diff:.6f}")
            if relative_diff > difference_threshold:
                mismatch_found = True
                break

        final_result = {
            "is_original_valid": not mismatch_found,
            "debug_info": debug_steps_info,
            "success": True
        }
        await submit_solution(task_id, final_result, final=True)

    # --------------------------------------------------------------------------
    #                             NORMAL BRANCH
    # --------------------------------------------------------------------------
    elif replicate_sequence:
        logger.warning("[process_tasks] replicate_sequence set but no original_task_id.")
        final_result = {
            "is_original_valid": False,
            "error": "No original_task_id provided",
            "success": False
        }
        await submit_solution(task_id, final_result, final=True)

    else:
        init_timeout = expected_time * multipliers["initialize_tensor"]
        version_number = await run_with_timeout(
            plugin.call_submodule('model_adapter', 'initialize_tensor', TENSOR_NAME, sot_url, task_params),
            "Initialize Tensor (Normal Task)",
            init_timeout,
            task_id=task_id
        )
        local_version = version_number

        load_timeout = expected_time * multipliers["download"]
        await run_with_timeout(
            plugin.call_submodule('model_adapter', 'load_input_tensor', predownloaded_data),
            "Load Input Tensor (Normal Task)",
            load_timeout,
            task_id=task_id
        )
        del predownloaded_data

        steps = task_params['steps']
        for step_idx in range(steps):
            logger.info(f"Task {task_id}: Executing step {step_idx}")
            exec_timeout = expected_time * multipliers["execute_step"]
            encoded_grad, loss_val = await run_with_timeout(
                plugin.call_submodule('model_adapter', 'execute_step', task_params, step_idx),
                f"Execute Step {step_idx}",
                exec_timeout,
                task_id=task_id
            )

            upload_timeout = expected_time * multipliers["upload"]
            grads_url = await run_with_timeout(
                upload_tensor(encoded_grad, 'grads', sot_url),
                f"Upload Gradients at step {step_idx}",
                upload_timeout,
                task_id=task_id
            )

            version_before_submission = await fetch_current_sot_version(sot_url)
            partial_result = {
                "version_number": local_version,
                "result_url": grads_url,
                "loss": loss_val
            }
            is_final = (step_idx == steps - 1)
            if is_final:
                partial_result["success"] = True
            await submit_solution(task_id, partial_result, final=is_final)

            wait_timeout = expected_time * multipliers["wait_version"]
            await run_with_timeout(
                wait_for_version_advance(version_before_submission, sot_url),
                f"Wait for Version Advance at step {step_idx}",
                wait_timeout,
                task_id=task_id
            )

            get_diffs_timeout = expected_time * multipliers["get_diffs"]
            new_diffs, end_time = await run_with_timeout(
                get_diffs_in_range(local_version, None, sot_url),
                f"Get Diffs at step {step_idx}",
                get_diffs_timeout,
                task_id=task_id
            )
            for diff_url in new_diffs:
                full_diff_url = f"{sot_url}{diff_url}"
                diff_dl_timeout = expected_time * multipliers["download"]
                diff_data = await run_with_timeout(
                    download_file(full_diff_url, download_type='tensor', chunk_timeout=diff_dl_timeout),
                    f"Download Diff {diff_url} at step {step_idx}",
                    diff_dl_timeout,
                    task_id=task_id
                )
                if not isinstance(diff_data, dict):
                    logger.warning(f"Downloaded diff is not a dict; skipping: {diff_url}")
                    continue
                apply_diff_timeout = expected_time * multipliers["apply_diff"]
                await run_with_timeout(
                    plugin.call_submodule('model_adapter', 'decode_and_apply_diff', diff_data),
                    f"Apply Diff {diff_url} at step {step_idx}",
                    apply_diff_timeout,
                    task_id=task_id
                )
            local_version = end_time


async def submit_failure_result(task_id, error_message):
    """
    Utility to finalize a task with success=False and log the error for DB.
    """
    final_result = {
        "success": False,
        "error": error_message
    }
    try:
        await submit_solution(task_id, final_result, final=True)
    except Exception as e:
        logger.error(f"Error submitting failure result for task {task_id}: {e}", exc_info=True)


async def process_tasks():
    """
    If there is a queued task, pop it off the queue and process it with:
      - Indefinite wait for concurrency lock (no time-based acquisition limit).
      - An overall timeout that starts only after the lock is acquired.
      - If something fails, we submit a failure result right away.
    """
    global task_queue, concurrent_tasks_counter

    if await task_queue.queue_length() == 0:
        logger.debug("No tasks in the queue to process.")
        return

    async with concurrent_tasks_counter_lock:
        concurrent_tasks_counter += 1

    task_item = None
    try:
        task_item = await task_queue.get_next_task()
        if not task_item:
            logger.debug("No tasks to process after retrieval.")
            return

        task_id = task_item['task_id']

        # ------------------------------------------------------
        # Wait indefinitely for the concurrency lock (no timeout).
        # We measure the wait time but do NOT finalize on any lock
        # acquisition timeout. The worker just waits as long as needed.
        # ------------------------------------------------------
        start_lock_wait = time.time()
        await task_processing_lock.acquire(priority=task_item['time_solver_selected'])
        end_lock_wait = time.time()

        lock_wait_duration = end_lock_wait - start_lock_wait
        logger.debug(
            f"Acquired concurrency lock for task {task_id} after waiting {lock_wait_duration:.2f}s."
        )

        # Now that we hold the lock, start the main `_process_task_once` with an overall timeout.
        overall_multiplier = 3.0
        expected_time = float(task_item.get("task_time", 60))
        overall_timeout = expected_time * overall_multiplier
        if args.cloud:
            overall_timeout += CLOUD_TIMEOUT_BONUS

        worker_task = asyncio.create_task(_process_task_once(task_item))
        try:
            await asyncio.wait_for(worker_task, timeout=overall_timeout)
        except asyncio.TimeoutError:
            partial_times = dict(task_step_times_map.get(task_id, {}))
            logger.error(
                f"Overall task {task_id} processing timed out after {overall_timeout:.2f} seconds. "
                f"Partial step times up to cancellation: {json.dumps(partial_times, indent=2)}"
            )
            error_message = (
                f"Overall processing timeout after {overall_timeout:.2f} seconds. "
                f"Partial step times: {partial_times}"
            )
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
            await submit_failure_result(task_id, error_message)

        except NoMoreDataException:
            logger.info(f"Task {task_id}: No more data available; skipping further attempts.")
            await submit_failure_result(task_id, "No more data available")

        except Exception as e:
            msg = str(e) or "Encountered unknown error."
            logger.error(f"Task {task_id} processing failed: {msg}", exc_info=True)
            await submit_failure_result(task_id, msg)

        finally:
            # Release the concurrency lock (we acquired it earlier).
            await task_processing_lock.release()

        # If we reach here (no exceptions or timeouts), the task finished successfully.
        async with task_start_times_lock:
            task_start_time = task_start_times.pop(task_id, None)
        total_time = time.time() - (task_start_time if task_start_time else time.time())
        logger.info(
            f"Task {task_id}: Completed in {total_time:.2f}s. "
            f"Concurrent tasks: {concurrent_tasks_counter}"
        )

    finally:
        # Decrement the concurrency counter
        async with concurrent_tasks_counter_lock:
            concurrent_tasks_counter -= 1
        # Mark the task as completed if we actually pulled one
        if task_item is not None:
            await mark_order_completed(task_item['task_id'])
