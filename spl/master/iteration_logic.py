# file: spl/master/iteration_logic.py

import json
import time
import asyncio
import logging
import aiohttp

from ..models import TaskStatus
from .config import args
from ..db.db_adapter_client import DBAdapterClient

logger = logging.getLogger(__name__)

###############################################################################
# Constants controlling re-creation if tasks are forcibly deleted
###############################################################################
TASK_MISSING_THRESHOLD = 10
MAX_RECREATES_PER_ITERATION = 3

###############################################################################
# Core iteration-state getters/setters
###############################################################################
async def get_iteration_state(db_adapter, job_id: int, state_key: str, iteration_number: int) -> dict:
    """
    Return job.master_state_json[state_key][str(iteration_number)],
    creating a stub if not found.
    """
    state_data = await db_adapter.get_master_state_for_job(job_id)
    iteration_states = state_data.get(state_key, {})

    key_str = str(iteration_number)
    if key_str not in iteration_states:
        iteration_states[key_str] = {"stage": "pending_get_input"}
        state_data[state_key] = iteration_states
        await db_adapter.update_master_state_for_job(job_id, state_data)

    return iteration_states[key_str]

async def save_iteration_state(db_adapter, job_id: int, state_key: str, iteration_number: int, subdict: dict):
    """
    Overwrite the sub-dict for iteration_number in job.master_state_json[state_key].
    """
    state_data = await db_adapter.get_master_state_for_job(job_id)
    iteration_states = state_data.get(state_key, {})
    iteration_states[str(iteration_number)] = subdict
    state_data[state_key] = iteration_states
    await db_adapter.update_master_state_for_job(job_id, state_data)

async def remove_iteration_entry(db_adapter, job_id: int, state_key: str, iteration_number: int):
    """
    Remove iteration_number from job.master_state_json[state_key] once it's fully done.
    """
    state_data = await db_adapter.get_master_state_for_job(job_id)
    iteration_states = state_data.get(state_key, {})
    key_str = str(iteration_number)
    if key_str in iteration_states:
        del iteration_states[key_str]
        state_data[state_key] = iteration_states
        await db_adapter.update_master_state_for_job(job_id, state_data)
        logger.info(f"[remove_iteration_entry] iteration={iteration_number} removed.")


###############################################################################
# Wait for the final result, re-creating forcibly deleted tasks if needed
###############################################################################
async def wait_for_result(db_adapter, plugin, sot_url, job_id: int, task_id: int) -> dict:
    """
    Keep polling DB for final result. If forcibly deleted for >TASK_MISSING_THRESHOLD
    times in a row, re-create a new Task ID (assuming job is still active).
    """
    missing_count = 0

    while True:
        task = await db_adapter.get_task(task_id)
        if not task:
            missing_count += 1

            # ─────────────────────────────────────────────────────────
            # NEW CHECK: Ensure the job is still active before re-creating.
            # ─────────────────────────────────────────────────────────
            job_obj = await db_adapter.get_job(job_id)
            if not job_obj or not job_obj.active:
                logger.info(
                    f"[wait_for_result] Job {job_id} inactive => skipping re-creation of deleted task {task_id}."
                )
                return {}

            if missing_count >= TASK_MISSING_THRESHOLD:
                logger.warning(f"[wait_for_result] Task {task_id} missing => re-create.")
                new_task_id = await _handle_recreate_missing_task(db_adapter, plugin, sot_url, job_id, task_id)
                if not new_task_id:
                    # If re-creation fails or job is inactive => no result
                    return {}
                task_id = new_task_id
                missing_count = 0
            else:
                await asyncio.sleep(0.5)
            continue
        else:
            missing_count = 0

        # If we found the Task, check status
        if task.status == TaskStatus.SanityCheckPending.name:
            if task.result:
                is_valid = await plugin.call_submodule("model_adapter", "run_sanity_check", task.result)
                await _finalize_sanity_check(db_adapter, job_id, task.id, is_valid)

        if task.status in [TaskStatus.ResolvedCorrect.name, TaskStatus.ResolvedIncorrect.name]:
            return task.result

        await asyncio.sleep(0.5)


###############################################################################
# If a Task was forcibly deleted from DB, re-create it if the job is still active
###############################################################################
async def _handle_recreate_missing_task(db_adapter, plugin, sot_url, job_id: int, old_task_id: int):
    """
    1) Check the job is still active
    2) Find iteration referencing old_task_id
    3) revert from 'pending_wait_for_result' => 'pending_submit_task'
    4) create a new DB Task + Bid
    5) store new task_id in iteration_state
    6) if we exceed MAX_RECREATES_PER_ITERATION => set job inactive
    """
    # 1) Confirm the job is still active
    job_obj = await db_adapter.get_job(job_id)
    if not job_obj or not job_obj.active:
        logger.warning(
            f"[_handle_recreate_missing_task] job {job_id} is inactive => "
            f"skip re-creation of missing task {old_task_id}."
        )
        return 0

    # 2) Find which iteration was referencing old_task_id
    iteration_num, iteration_state = await _find_iteration_for_task(db_adapter, job_id, old_task_id)
    if iteration_num is None:
        logger.error(f"[_handle_recreate_missing_task] cannot find iteration for old_task_id={old_task_id}")
        return 0

    # 3) check how many times we re-created for that iteration
    rekey = f"_recreate_count_{iteration_num}"
    st_data = await db_adapter.get_master_state_for_job(job_id)
    iteration_map = st_data.get("iteration_recreates", {})
    cur_count = iteration_map.get(rekey, 0)

    if cur_count >= MAX_RECREATES_PER_ITERATION:
        logger.error(
            f"[_handle_recreate_missing_task] iteration={iteration_num} used up re-creates => job inactive."
        )
        await db_adapter.update_job_active(job_id, False)
        return 0

    iteration_map[rekey] = cur_count + 1
    st_data["iteration_recreates"] = iteration_map
    await db_adapter.update_master_state_for_job(job_id, st_data)

    # 4) Rewind iteration from "pending_wait_for_result" to "pending_submit_task"
    iteration_state["stage"] = "pending_submit_task"
    iteration_state.pop("task_id_info", None)
    iteration_state.pop("result", None)
    await save_iteration_state(db_adapter, job_id, "master_iteration_state", iteration_num, iteration_state)

    # 5) If no learning_params, fetch from plugin
    learning_params = iteration_state.get("learning_params", {})
    if not learning_params:
        logger.warning(f"[recreate_missing_task] iteration={iteration_num} no learning_params => reload from plugin.")
        learning_params = await plugin.get_master_learning_hyperparameters()
        iteration_state["learning_params"] = learning_params
        await save_iteration_state(db_adapter, job_id, "master_iteration_state", iteration_num, iteration_state)

    input_url = iteration_state.get("input_url")
    if not input_url:
        logger.error(
            f"[_handle_recreate_missing_task] iteration={iteration_num} missing input_url => cannot re-create task."
        )
        return 0

    # 6) Actually create a new DB Task + Bid
    params_json = json.dumps({
        "input_url": input_url,
        **learning_params
    })
    new_task_info = await submit_task_with_persist(db_adapter, job_id, iteration_num, params_json)
    if not new_task_info:
        logger.error("[_handle_recreate_missing_task] failed creating new tasks => returning 0.")
        return 0

    # store it in iteration_state
    iteration_state["task_id_info"] = new_task_info
    iteration_state["stage"] = "pending_wait_for_result"
    await save_iteration_state(db_adapter, job_id, "master_iteration_state", iteration_num, iteration_state)

    # Return the new task_id
    return new_task_info[0]["task_id"]


async def _find_iteration_for_task(db_adapter, job_id: int, old_task_id: int):
    """
    Scan master_iteration_state to see which iteration uses old_task_id
    in 'task_id_info'. Return (iteration_number, iteration_subdict).
    """
    st_data = await db_adapter.get_master_state_for_job(job_id)
    iteration_states = st_data.get("master_iteration_state", {})
    for it_str, subdict in iteration_states.items():
        arr = subdict.get("task_id_info", [])
        if isinstance(arr, list):
            for item in arr:
                if item.get("task_id") == old_task_id:
                    try:
                        it_num = int(it_str)
                        return it_num, subdict
                    except:
                        pass
    return None, None


###############################################################################
# finalize_sanity_check
###############################################################################
async def _finalize_sanity_check(db_adapter: DBAdapterClient, job_id: int, task_id: int, is_valid: bool):
    success = await db_adapter.finalize_sanity_check(task_id, is_valid)
    if not success:
        logger.error("[_finalize_sanity_check] finalize_sanity_check call failed")



###############################################################################
# create_bids_and_tasks wrapper used in re-creation
###############################################################################
async def submit_task_with_persist(db_adapter, job_id: int, iteration_number: int, params_str: str):
    """
    Create a DB task + bid via create_bids_and_tasks. On success, store
    `last_task_creation_time` in job.master_state_json so we can measure inactivity.
    """
    import datetime

    start_time = datetime.datetime.now()
    max_duration = datetime.timedelta(seconds=300)  # 300s => 5 minutes
    retry_delay = 1
    attempt = 0

    while True:
        attempt += 1
        elapsed = datetime.datetime.now() - start_time
        if elapsed >= max_duration:
            raise RuntimeError(f"[submit_task_with_persist] no solver assigned within {max_duration}.")

        try:
            # Basic example: pass a fixed price=1
            created_items = await db_adapter.create_bids_and_tasks(
                job_id, 1, 1, params_str, None
            )
            if created_items and len(created_items) > 0:
                state_data = await db_adapter.get_master_state_for_job(job_id)
                state_data["last_task_creation_time"] = time.time()
                await db_adapter.update_master_state_for_job(job_id, state_data)
                return created_items
            else:
                logger.error("[submit_task_with_persist] create_bids_and_tasks returned empty.")
                return None

        except Exception as e:
            logger.error(f"[submit_task_with_persist] attempt #{attempt}, error => {e}")
            await asyncio.sleep(retry_delay)
            retry_delay = min(2 * retry_delay, 60)


###############################################################################
# Get input_url from SOT with retries
###############################################################################
async def get_input_url_with_persist(sot_url: str, db_adapter, iteration_number: int) -> str:
    import uuid
    import time
    import json
    from eth_account import Account
    from eth_account.messages import encode_defunct

    url = f"{sot_url}/get_batch"
    attempts = 0
    max_attempts = 400
    retry_delay = 1

    while attempts < max_attempts:
        attempts += 1
        try:
            msg = json.dumps({
                "endpoint": "get_batch",
                "nonce": str(uuid.uuid4()),
                "timestamp": int(time.time())
            }, sort_keys=True)
            sig = _sign_with_master_key(msg)

            headers = {"Authorization": f"{msg}:{sig}"}
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if "input_url" in data:
                            return sot_url + data["input_url"]
                        else:
                            logger.error("[get_input_url_with_persist] no input_url in JSON.")
                    else:
                        logger.error(f"[get_input_url_with_persist] got HTTP {resp.status}")
        except Exception as e:
            logger.error(f"[get_input_url_with_persist] exception => {e}")

        await asyncio.sleep(retry_delay)

    raise RuntimeError("[get_input_url_with_persist] Max attempts exceeded.")


###############################################################################
# Helper for signing with the Master’s private key
###############################################################################
def _sign_with_master_key(message: str) -> str:
    from eth_account import Account
    from eth_account.messages import encode_defunct
    msg_def = encode_defunct(text=message)
    acct = Account.from_key(args.private_key)
    signed = acct.sign_message(msg_def)
    return signed.signature.hex()
