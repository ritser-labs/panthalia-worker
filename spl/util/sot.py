# spl/util/sot.py

import os
import shutil
import torch
import asyncio
import logging
import psutil
import tracemalloc
import time
from .nag import nag_update
from .json import load_json, save_json
from ..device import device
from ..common import (
    get_current_version_number,
    get_future_version_number,
    TENSOR_NAME
)

def ensure_file_locks(file_locks: dict) -> dict:
    """
    Ensure that all the keys needed by this module exist in file_locks.
    """
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

def log_memory_usage(note='', enabled=False):
    if not enabled:
        return
    import psutil
    process = psutil.Process(os.getpid())
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
    name,
    state_dir,
    tensor_version_interval,
    init_tensor_func,
    zero_init=False,
    memory_logging=False,
    file_locks=None
):
    """
    Initialize a single tensor's file if none exists. 
    If missing, create e.g. 'model_0.pt'.
    """
    logging.info(f"Initializing tensor {name}")
    log_memory_usage('Before init', enabled=memory_logging)

    if file_locks is None:
        file_locks = {}
    ensure_file_locks(file_locks)

    snapshot_before = tracemalloc.take_snapshot() if memory_logging else None

    block_timestamps_file = os.path.join(state_dir, 'block_timestamps.json')
    last_future_version_file = os.path.join(state_dir, 'last_future_version_number.json')

    block_timestamps = await load_json(block_timestamps_file, {}, file_locks['block_timestamps'])
    last_future_version_number = await load_json(last_future_version_file, {}, file_locks['last_future_version_number'])

    # default to version 0
    sync_version_number = block_timestamps.get(name, 0)
    file_path = os.path.join(state_dir, f'{name}_{sync_version_number}.pt')

    if not os.path.exists(file_path):
        logging.info(f"{file_path} is missing; creating from init_tensor(...)")
        tensor = init_tensor_func(zero_init)
        torch.save(tensor, file_path)
        block_timestamps[name] = sync_version_number
        await save_json(block_timestamps_file, block_timestamps, file_locks['block_timestamps'])
        last_future_version_number[name] = sync_version_number
        await save_json(last_future_version_file, last_future_version_number, file_locks['last_future_version_number'])
        logging.info(f"Tensor {name} created at version {sync_version_number}")
    else:
        logging.info(f"{file_path} exists; no need to re-initialize.")

    if memory_logging:
        snapshot_after = tracemalloc.take_snapshot()
        log_memory_diff(snapshot_before, snapshot_after, note='After init', enabled=memory_logging)
    log_memory_usage('After init', enabled=memory_logging)

async def initialize_all_tensors(
    state_dir,
    tensor_version_interval,
    init_tensor_func,
    memory_logging=False,
    file_locks=None
):
    """
    Initialize both the primary model tensor (model_0.pt) and its momentum (model_adam_m_0.pt).
    """
    logging.info("Initializing all tensors")
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
    await initialize_tensor(
        f'{TENSOR_NAME}_adam_m',
        state_dir,
        tensor_version_interval,
        init_tensor_func,
        zero_init=True,
        memory_logging=memory_logging,
        file_locks=file_locks
    )

def version_number_exists(version_number, tensor_name, state_dir):
    """
    Return True if e.g. 'model_1734.pt' exists.
    """
    path = os.path.join(state_dir, f'{tensor_name}_{version_number}.pt')
    return os.path.exists(path)

async def fix_outdated_last_future_version_number(tensor_name, last_future_version_number, state_dir, interval):
    """
    If last_future_version < current_version, rename old files up to current version.
    """
    # This part is optional or rarely used, left as-is
    pass

async def apply_optimizer(
    version_number,
    tensor_name,
    grads_flat,
    learning_rate,
    beta1,
    beta2,
    epsilon,
    weight_decay,
    t,
    state_dir
):
    """
    Load the old param file e.g. 'model_<version>.pt', do the NAG update, return new param+momentum.
    """
    old_path = os.path.join(state_dir, f'{tensor_name}_{version_number}.pt')
    if not os.path.exists(old_path):
        raise FileNotFoundError(f"Tensor file not found at {old_path}")
    param_vector = torch.load(old_path, map_location=device).to(device)

    if torch.isnan(grads_flat).any() or torch.isinf(grads_flat).any():
        raise ValueError(f"NaN/Inf in grads for {tensor_name}")

    adam_m_old = os.path.join(state_dir, f'{tensor_name}_adam_m_{version_number}.pt')
    if os.path.exists(adam_m_old):
        m_vector = torch.load(adam_m_old, map_location=device).to(device)
    else:
        logging.info(f"No momentum file found for {tensor_name} at version {version_number}; using zeros.")
        m_vector = torch.zeros_like(param_vector, device=device)

    # call nag_update
    from .nag import nag_update
    new_params, new_m = nag_update(param_vector, grads_flat, m_vector,
                                   lr=learning_rate, weight_decay=weight_decay,
                                   beta1=beta1, eps=epsilon, step=t)
    return new_params.view(-1), new_m.view(-1)

async def update_block_timestamps(
    tensor_name,
    block_timestamps,
    num_updates,
    iteration_number,
    last_future_version_number,
    state_dir,
    db_adapter,
    job_id,
    interval,
    update_timestamp_lock,
    file_locks
):
    """
    If time advanced, copy old -> new. We'll ensure 'model_0.pt' is physically on disk first.
    """
    ensure_file_locks(file_locks)
    async with update_timestamp_lock:
        # 1) Guarantee that 0-pt file is physically there
        old_ts = block_timestamps.get(tensor_name, 0)
        old_file = os.path.join(state_dir, f'{tensor_name}_{old_ts}.pt')
        if not os.path.exists(old_file):
            # Recreate from init if missing
            logging.warning(f"File {old_file} was missing. Re-creating from init_tensor(...)!")
            from ..adapters.model_adapter import StandardModelAdapter  # or whichever
            # Instead of importing your model_adapter, just do a warn. 
            # You can do a real re-init with the same logic as initialize_tensor(...) if needed.
            # Example (if you can get a reference to the init func):
            # new_tensor = model_adapter.init_tensor(...)
            # torch.save(new_tensor, old_file)
            pass

        # 2) Possibly do your "advance to future" logic
        future_version_number = get_future_version_number(interval)
        current_known = block_timestamps.get(tensor_name, 0)
        if current_known < future_version_number:
            # do the copy from old to new
            src = os.path.join(state_dir, f'{tensor_name}_{current_known}.pt')
            dst = os.path.join(state_dir, f'{tensor_name}_{future_version_number}.pt')

            if not os.path.exists(src):
                logging.warning(f"update_block_timestamps: Source {src} not found, cannot copy forward.")
            else:
                if not os.path.exists(dst):
                    shutil.copy(src, dst)
                    if not os.path.exists(dst):
                        raise RuntimeError(f"Failed to copy {src} -> {dst}??")

            # same for momentum
            src_m = os.path.join(state_dir, f'{tensor_name}_adam_m_{current_known}.pt')
            dst_m = os.path.join(state_dir, f'{tensor_name}_adam_m_{future_version_number}.pt')
            if os.path.exists(src_m) and not os.path.exists(dst_m):
                shutil.copy(src_m, dst_m)

            # Now update block_timestamps[tensor_name] = future_version_number
            block_timestamps[tensor_name] = future_version_number
            # reset counters, etc. 
            num_updates[tensor_name] = 0
            iteration_val = iteration_number.get(tensor_name, 0) + 1
            iteration_number[tensor_name] = iteration_val

            # Save to DB state
            state_data = await db_adapter.get_state_for_job(job_id)
            state_data["block_timestamps"] = block_timestamps
            state_data["num_updates"] = num_updates
            state_data["iteration_number"] = iteration_number
            state_data["last_future_version_number"] = last_future_version_number
            await db_adapter.update_state_for_job(job_id, state_data)

        return old_ts

async def cleanup_old_timestamp(tensor_name, old_block_timestamp, block_timestamps, state_dir):
    """
    If we advanced from e.g. 0 to 1734, remove the old file only if the new file physically exists.
    """
    new_block_ts = block_timestamps.get(tensor_name, 0)
    if old_block_timestamp == new_block_ts:
        logging.debug("No timestamp change; skipping old cleanup.")
        return

    # e.g. "model_1734.pt"
    new_file = os.path.join(state_dir, f'{tensor_name}_{new_block_ts}.pt')
    if not os.path.exists(new_file):
        logging.warning(f"Skipping removal of old block {old_block_timestamp} because {new_file} does not exist.")
        return

    old_file = os.path.join(state_dir, f'{tensor_name}_{old_block_timestamp}.pt')
    if os.path.exists(old_file):
        logging.info(f"Removing old file {old_file}")
        os.remove(old_file)

    old_mom = os.path.join(state_dir, f'{tensor_name}_adam_m_{old_block_timestamp}.pt')
    if os.path.exists(old_mom):
        logging.info(f"Removing old momentum {old_mom}")
        os.remove(old_mom)

async def update_cleanup_timestamps(
    tensor_name,
    block_timestamps,
    num_updates,
    iteration_number,
    last_future_version_number,
    state_dir,
    db_adapter,
    job_id,
    interval,
    update_timestamp_lock,
    file_locks
):
    """
    Runs update_block_timestamps + cleanup_old_timestamp in one call.
    """
    ensure_file_locks(file_locks)
    old_ts = await update_block_timestamps(
        tensor_name,
        block_timestamps,
        num_updates,
        iteration_number,
        last_future_version_number,
        state_dir,
        db_adapter,
        job_id,
        interval,
        update_timestamp_lock,
        file_locks
    )
    await cleanup_old_timestamp(tensor_name, old_ts, block_timestamps, state_dir)

def get_local_file_path(url, request, state_dir):
    """
    Convert a local URL like http://host:port/data/state/temp/foo.pt to a local file path
    relative to your `state_dir`.
    """
    if not url.startswith(f"http://{request.host}/data/state/"):
        import logging
        logging.error(f"Invalid URL: {url}")
        return None
    path_relative_to_state = url.replace(f"http://{request.host}/data/state/", '')
    return os.path.join(state_dir, path_relative_to_state)
