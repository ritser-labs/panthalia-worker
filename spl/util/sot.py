# spl/util/sot.py

import os
import shutil
import torch
import asyncio
import logging
import tracemalloc

from ..device import device
from ..common import (
    get_current_version_number,
    get_future_version_number,
    TENSOR_NAME
)
from .nag import nag_update  # <-- Was used before, you can remove 'nag.py' if not needed
from .json import load_json, save_json

# NEW: import your AdamW function
from .adam import adamw_update  # <-- CHANGED: we will use this instead
from safetensors.torch import save_file as safetensors_save_file
from safetensors.torch import load_file as safetensors_load_file

def ensure_file_locks(file_locks: dict) -> dict:
    required_keys = [
        "block_timestamps",
        "last_future_version_number",
        "num_updates",
        "iteration_number",
    ]
    for key in required_keys:
        if key not in file_locks:
            file_locks[key] = asyncio.Lock()
    return file_locks

def get_local_file_path(url: str, request, state_dir: str) -> str | None:
    expected_prefix = f"http://{request.host}/data/state/"
    if not url.startswith(expected_prefix):
        logging.error(f"[get_local_file_path] Invalid or unexpected URL: {url}")
        return None
    relative_path = url.replace(expected_prefix, '')
    return os.path.join(state_dir, relative_path)

def version_number_exists(version_number: int, tensor_name: str, state_dir: str) -> bool:
    path = os.path.join(state_dir, f'{tensor_name}_{version_number}.pt')
    return os.path.exists(path)

def log_memory_usage(note='', enabled=False):
    if not enabled:
        return
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    logging.debug(
        f"Memory usage ({note}): RSS={mem_info.rss / 1024**2:.2f} MB, "
        f"VMS={mem_info.vms / 1024**2:.2f} MB"
    )

def log_memory_diff(snapshot1, snapshot2, note='', enabled=False):
    if not enabled:
        return
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    logging.debug(f"Memory usage differences ({note}):")
    for stat in top_stats[:10]:
        logging.debug(stat)

async def initialize_tensor(
    name: str,
    state_dir: str,
    tensor_version_interval: int,
    init_tensor_func,
    zero_init=False,
    memory_logging=False,
    file_locks=None
):
    logging.info(f"[initialize_tensor] Checking existence of tensor {name}")
    log_memory_usage('Before init', enabled=memory_logging)

    if file_locks is None:
        file_locks = {}
    ensure_file_locks(file_locks)

    snapshot_before = tracemalloc.take_snapshot() if memory_logging else None

    block_timestamps_file = os.path.join(state_dir, 'block_timestamps.json')
    last_future_version_file = os.path.join(state_dir, 'last_future_version_number.json')

    block_timestamps = await load_json(block_timestamps_file, {}, file_locks['block_timestamps'])
    last_future_version_number = await load_json(last_future_version_file, {}, file_locks['last_future_version_number'])

    sync_version_number = block_timestamps.get(name, 0)
    file_path = os.path.join(state_dir, f'{name}_{sync_version_number}.pt')

    if not os.path.exists(file_path):
        logging.info(f"{file_path} is missing; creating from init_tensor(...)")
        tensor = init_tensor_func(zero_init)
        safetensors_save_file({"tensor": tensor}, file_path)

        block_timestamps[name] = sync_version_number
        await save_json(block_timestamps_file, block_timestamps, file_locks['block_timestamps'])

        last_future_version_number[name] = sync_version_number
        await save_json(last_future_version_file, last_future_version_number, file_locks['last_future_version_number'])

        logging.info(f"Tensor {name} created at version {sync_version_number}")
    else:
        logging.info(f"{file_path} exists; no need to re-initialize {name}.")

    if memory_logging:
        snapshot_after = tracemalloc.take_snapshot()
        log_memory_diff(snapshot_before, snapshot_after, note='After init', enabled=memory_logging)
    log_memory_usage('After init', enabled=memory_logging)

async def initialize_all_tensors(
    state_dir: str,
    tensor_version_interval: int,
    init_tensor_func,
    memory_logging=False,
    file_locks=None
):
    logging.info("Initializing all relevant tensors if they do not exist.")
    if file_locks is None:
        file_locks = {}
    ensure_file_locks(file_locks)

    await initialize_tensor(
        TENSOR_NAME,
        state_dir,
        tensor_version_interval,
        init_tensor_func,
        zero_init=False,
        memory_logging=memory_logging,
        file_locks=file_locks
    )

    # If you want separate param for Adam's momentum, keep it zero-initialized
    await initialize_tensor(
        f'{TENSOR_NAME}_adam_m',
        state_dir,
        tensor_version_interval,
        init_tensor_func,
        zero_init=True,
        memory_logging=memory_logging,
        file_locks=file_locks
    )
    # If you want separate param for Adam's variance
    await initialize_tensor(
        f'{TENSOR_NAME}_adam_v',
        state_dir,
        tensor_version_interval,
        init_tensor_func,
        zero_init=True,
        memory_logging=memory_logging,
        file_locks=file_locks
    )

# --------------------------------------------------------------------------
# CHANGED: Replace NAG with AdamW in apply_optimizer(...)
# --------------------------------------------------------------------------
async def apply_optimizer(
    version_number: int,
    tensor_name: str,
    grads_flat: torch.Tensor,
    learning_rate: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    t: float,
    state_dir: str
):
    """
    Applies AdamW to param_vector using first and second moment states in
    f'{tensor_name}_adam_m_{version_number}.pt' and f'{tensor_name}_adam_v_{version_number}.pt'.

    Returns:
      new_params, new_m, new_v
    """
    old_path = os.path.join(state_dir, f'{tensor_name}_{version_number}.pt')
    if not os.path.exists(old_path):
        raise FileNotFoundError(f"Tensor file not found at {old_path}")

    param_vector = safetensors_load_file(param_vector, device=device)['tensor']
    if torch.isnan(grads_flat).any() or torch.isinf(grads_flat).any():
        raise ValueError(f"NaN/Inf in grads for {tensor_name} -- aborting update.")

    # Load or init the old m, v
    old_m_path = os.path.join(state_dir, f'{tensor_name}_adam_m_{version_number}.pt')
    old_v_path = os.path.join(state_dir, f'{tensor_name}_adam_v_{version_number}.pt')

    if os.path.exists(old_m_path):
        m_vector = safetensors_load_file(m_vector, device=device)['tensor']
    else:
        logging.info(f"No momentum file found for {tensor_name} v{version_number}, using zeros.")
        m_vector = torch.zeros_like(param_vector, device=device)

    if os.path.exists(old_v_path):
        v_vector = safetensors_load_file(v_vector, device=device)['tensor']
    else:
        logging.info(f"No variance file found for {tensor_name} v{version_number}, using zeros.")
        v_vector = torch.zeros_like(param_vector, device=device)

    # CHANGED: Use AdamW now
    # step => int(t+1) if t is iteration count
    new_params, new_m, new_v = adamw_update(
        param_vector,
        grads_flat,
        m_vector,
        v_vector,
        lr=learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        step=int(t + 1),
    )

    return new_params.view(-1), new_m.view(-1), new_v.view(-1)

# --------------------------------------------------------------------------

async def update_block_timestamps(
    tensor_name: str,
    block_timestamps: dict,
    num_updates: dict,
    iteration_number: dict,
    last_future_version_number: dict,
    state_dir: str,
    interval: int,
    db_adapter,
    job_id: int
):
    old_ts = block_timestamps.get(tensor_name, 0)
    future_ts = get_future_version_number(interval)
    if old_ts < future_ts:
        src = os.path.join(state_dir, f'{tensor_name}_{old_ts}.pt')
        dst = os.path.join(state_dir, f'{tensor_name}_{future_ts}.pt')
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            logging.info(f"[update_block_timestamps] Copied {src} -> {dst}")

            src_m = os.path.join(state_dir, f'{tensor_name}_adam_m_{old_ts}.pt')
            dst_m = os.path.join(state_dir, f'{tensor_name}_adam_m_{future_ts}.pt')
            if os.path.exists(src_m) and not os.path.exists(dst_m):
                shutil.copy(src_m, dst_m)
                logging.info(f"[update_block_timestamps] Copied {src_m} -> {dst_m}")

            src_v = os.path.join(state_dir, f'{tensor_name}_adam_v_{old_ts}.pt')
            dst_v = os.path.join(state_dir, f'{tensor_name}_adam_v_{future_ts}.pt')
            if os.path.exists(src_v) and not os.path.exists(dst_v):
                shutil.copy(src_v, dst_v)
                logging.info(f"[update_block_timestamps] Copied {src_v} -> {dst_v}")

        block_timestamps[tensor_name] = future_ts
        num_updates[tensor_name] = 0
        iteration_val = iteration_number.get(tensor_name, 0) + 1
        iteration_number[tensor_name] = iteration_val
        last_future_version_number[tensor_name] = future_ts

        state_data = await db_adapter.get_sot_state_for_job(job_id)
        state_data["block_timestamps"] = block_timestamps
        state_data["num_updates"] = num_updates
        state_data["iteration_number"] = iteration_number
        state_data["last_future_version_number"] = last_future_version_number
        await db_adapter.update_sot_state_for_job(job_id, state_data)
        logging.info(
            f"[update_block_timestamps] Finalized old v={old_ts} => new v={future_ts}"
        )

async def cleanup_old_timestamp(
    tensor_name: str,
    old_block_timestamp: int,
    block_timestamps: dict,
    state_dir: str
):
    new_block_ts = block_timestamps.get(tensor_name, 0)
    if old_block_timestamp == new_block_ts:
        logging.debug("[cleanup_old_timestamp] No timestamp change; skipping old cleanup.")
        return

    new_file = os.path.join(state_dir, f'{tensor_name}_{new_block_ts}.pt')
    if not os.path.exists(new_file):
        logging.warning(f"[cleanup_old_timestamp] Skipping removal of old block {old_block_timestamp} because {new_file} doesn't exist.")
        return

    old_file = os.path.join(state_dir, f'{tensor_name}_{old_block_timestamp}.pt')
    if os.path.exists(old_file):
        logging.info(f"[cleanup_old_timestamp] Removing old file {old_file}")
        os.remove(old_file)

    old_mom = os.path.join(state_dir, f'{tensor_name}_adam_m_{old_block_timestamp}.pt')
    if os.path.exists(old_mom):
        logging.info(f"[cleanup_old_timestamp] Removing old momentum {old_mom}")
        os.remove(old_mom)

    old_var = os.path.join(state_dir, f'{tensor_name}_adam_v_{old_block_timestamp}.pt')
    if os.path.exists(old_var):
        logging.info(f"[cleanup_old_timestamp] Removing old variance {old_var}")
        os.remove(old_var)

async def update_cleanup_timestamps(
    tensor_name: str,
    block_timestamps: dict,
    num_updates: dict,
    iteration_number: dict,
    last_future_version_number: dict,
    state_dir: str,
    db_adapter,
    job_id: int,
    interval: int,
    update_timestamp_lock: asyncio.Lock,
    file_locks: dict
):
    ensure_file_locks(file_locks)
    async with update_timestamp_lock:
        old_ts = block_timestamps.get(tensor_name, 0)
        current_version = get_current_version_number(interval)

        logging.debug(
            f"[update_cleanup_timestamps] Called for {tensor_name}. old_ts={old_ts}, current_version={current_version}"
        )

        if old_ts < current_version:
            src = os.path.join(state_dir, f'{tensor_name}_{old_ts}.pt')
            dst = os.path.join(state_dir, f'{tensor_name}_{current_version}.pt')
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy(src, dst)
                logging.info(f"[update_cleanup_timestamps] Copied {src} -> {dst}")

                src_m = os.path.join(state_dir, f'{tensor_name}_adam_m_{old_ts}.pt')
                dst_m = os.path.join(state_dir, f'{tensor_name}_adam_m_{current_version}.pt')
                if os.path.exists(src_m) and not os.path.exists(dst_m):
                    shutil.copy(src_m, dst_m)
                    logging.info(f"[update_cleanup_timestamps] Copied {src_m} -> {dst_m}")

                src_v = os.path.join(state_dir, f'{tensor_name}_adam_v_{old_ts}.pt')
                dst_v = os.path.join(state_dir, f'{tensor_name}_adam_v_{current_version}.pt')
                if os.path.exists(src_v) and not os.path.exists(dst_v):
                    shutil.copy(src_v, dst_v)
                    logging.info(f"[update_cleanup_timestamps] Copied {src_v} -> {dst_v}")

            block_timestamps[tensor_name] = current_version
            num_updates[tensor_name] = 0
            iteration_val = iteration_number.get(tensor_name, 0) + 1
            iteration_number[tensor_name] = iteration_val
            last_future_version_number[tensor_name] = current_version

            state_data = await db_adapter.get_sot_state_for_job(job_id)
            state_data["block_timestamps"] = block_timestamps
            state_data["num_updates"] = num_updates
            state_data["iteration_number"] = iteration_number
            state_data["last_future_version_number"] = last_future_version_number

            logging.debug(
                f"[update_cleanup_timestamps] Saving block_timestamps[{tensor_name}]={current_version} to DB..."
            )

            try:
                await db_adapter.update_sot_state_for_job(job_id, state_data)
                logging.debug("[update_cleanup_timestamps] Successfully updated job state in DB.")
            except Exception as e:
                logging.error("[update_cleanup_timestamps] update_sot_state_for_job crashed: ", exc_info=True)
                return

            # Now remove the old version if new_file definitely exists
            await cleanup_old_timestamp(tensor_name, old_ts, block_timestamps, state_dir)
        else:
            logging.debug(
                f"[update_cleanup_timestamps] old_ts >= current_version => no changes. "
                f"(old_ts={old_ts}, current_version={current_version})"
            )
