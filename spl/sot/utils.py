# spl/sot/utils.py
import os
import shutil
import torch
import asyncio
import logging
import psutil
import tracemalloc
import time
import random
from .utils_nag import nag_update
from ..util.json import load_json, save_json
from ..device import device
from ..common import (
    get_current_version_number,
    get_future_version_number,
    CHUNK_SIZE,
    TENSOR_NAME
)

def log_memory_usage(note='', enabled=False):
    if enabled:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        logging.debug(
            f"Memory usage ({note}): RSS={mem_info.rss / 1024**2:.2f} MB, VMS={mem_info.vms / 1024**2:.2f} MB"
        )

def log_memory_diff(snapshot1, snapshot2, note='', enabled=False):
    if enabled:
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        logging.debug(f"Memory usage differences ({note}):")
        for stat in top_stats[:10]:
            logging.debug(stat)

async def initialize_tensor(name, state_dir, plugin, zero_init=False, memory_logging=False, file_locks=None):
    logging.info(f"Initializing tensor {name}")
    log_memory_usage('Before initializing tensor', enabled=memory_logging)

    snapshot_before = tracemalloc.take_snapshot() if memory_logging else None

    block_timestamps_file = os.path.join(state_dir, 'block_timestamps.json')
    last_future_version_file = os.path.join(state_dir, 'last_future_version_number.json')

    # Load the JSON files with locks
    block_timestamps = await load_json(block_timestamps_file, {}, file_locks['block_timestamps'])
    last_future_version_number = await load_json(last_future_version_file, {}, file_locks['last_future_version_number'])

    sync_version_number = block_timestamps.get(
        name, get_current_version_number(await plugin.get('tensor_version_interval')))

    file_path = os.path.join(state_dir, f'{name}_{sync_version_number}.pt')
    if os.path.exists(file_path):
        logging.info(f"Tensor {name} already exists at version {sync_version_number}")
        return

    tensor = await plugin.call_submodule('model_adapter', 'init_tensor', zero_init)
    torch.save(tensor, file_path)
    block_timestamps[name] = sync_version_number
    await save_json(block_timestamps_file, block_timestamps, file_locks['block_timestamps'])
    last_future_version_number[name] = sync_version_number
    await save_json(last_future_version_file, last_future_version_number, file_locks['last_future_version_number'])

    if memory_logging:
        snapshot_after = tracemalloc.take_snapshot()
        log_memory_diff(snapshot_before, snapshot_after, note='After initializing tensor', enabled=memory_logging)

    log_memory_usage('After initializing tensor', enabled=memory_logging)
    logging.info(f'Tensor {name} initialized at version {sync_version_number}')

async def initialize_all_tensors(state_dir, plugin, memory_logging=False, file_locks=None):
    logging.info("Initializing all tensors")
    # Pass file_locks to initialize_tensor
    await initialize_tensor(TENSOR_NAME, state_dir, plugin, zero_init=False, memory_logging=memory_logging, file_locks=file_locks)
    await initialize_tensor(f'{TENSOR_NAME}_adam_m', state_dir, plugin, zero_init=True, memory_logging=memory_logging, file_locks=file_locks)

async def fix_outdated_last_future_version_number(tensor_name, last_future_version_number, state_dir, plugin):
    value = last_future_version_number.get(tensor_name, 0)
    current_version_number = get_current_version_number(await plugin.get('tensor_version_interval'))
    if value < current_version_number:
        if os.path.exists(os.path.join(state_dir, f'{tensor_name}_{value}.pt')):
            for f in [f'{tensor_name}', f'{tensor_name}_adam_m']:
                old_path = os.path.join(state_dir, f'{f}_{value}.pt')
                new_path = os.path.join(state_dir, f'{f}_{current_version_number}.pt')
                if os.path.exists(old_path):
                    await asyncio.to_thread(os.rename, old_path, new_path)
        last_future_version_number[tensor_name] = current_version_number

def version_number_exists(version_number, tensor_name, state_dir):
    return os.path.exists(os.path.join(state_dir, f'{tensor_name}_{version_number}.pt'))

def get_local_file_path(url, request, state_dir):
    if not url.startswith(f"http://{request.host}/data/state/"):
        logging.error(f"Invalid URL: {url}")
        return None
    path_relative_to_state = url.replace(f"http://{request.host}/data/state/", '')
    return os.path.join(state_dir, path_relative_to_state)

async def apply_optimizer(version_number, tensor_name, grads_flat, learning_rate, beta1, beta2, epsilon, weight_decay, t, state_dir):
    tensor_path = os.path.join(state_dir, f'{tensor_name}_{version_number}.pt')
    if not os.path.exists(tensor_path):
        raise FileNotFoundError(f"Tensor file for {tensor_name} not found at {tensor_path}")

    tensor = await asyncio.to_thread(torch.load, tensor_path, map_location=device)
    if tensor is None:
        raise ValueError(f"Failed to load tensor for {tensor_name}")

    tensor = tensor.to(device)

    if torch.isnan(grads_flat).any() or torch.isinf(grads_flat).any():
        logging.error(f"NaNs or Infs detected in gradients before optimizer update for {tensor_name}")
        raise ValueError(f"NaNs or Infs detected in gradients for {tensor_name}")

    tensor_adam_m_path = os.path.join(state_dir, f'{tensor_name}_adam_m_{version_number}.pt')
    if os.path.exists(tensor_adam_m_path):
        adam_m = await asyncio.to_thread(torch.load, tensor_adam_m_path, map_location=device)
        adam_m = adam_m.to(device)
    else:
        logging.debug(f'adam_m not found for {tensor_name}, initializing to zeros')
        adam_m = torch.zeros_like(tensor, device=device)

    param_update, m_update = await asyncio.to_thread(
        nag_update,
        tensor, grads_flat, adam_m, learning_rate, weight_decay, beta1, epsilon, t
    )

    return param_update.view(-1), m_update.view(-1)

async def update_block_timestamps(tensor_name, block_timestamps, num_updates, iteration_number, last_future_version_number, state_dir, db_adapter, job_id, plugin, update_timestamp_lock, file_locks):
    async with update_timestamp_lock:
        future_version_number = get_future_version_number(await plugin.get('tensor_version_interval'))
        await fix_outdated_last_future_version_number(tensor_name, last_future_version_number, state_dir, plugin)

        new_block_timestamp = last_future_version_number.get(tensor_name, 0)
        block_timestamps_file = os.path.join(state_dir, 'block_timestamps.json')
        num_updates_file = os.path.join(state_dir, 'num_updates.json')
        iteration_number_file = os.path.join(state_dir, 'iteration_number.json')
        last_future_version_file = os.path.join(state_dir, 'last_future_version_number.json')

        old_block_timestamp = None
        if new_block_timestamp < future_version_number:
            old_block_timestamp = block_timestamps.get(tensor_name, 0)
            logging.info(f"Updating block timestamps for {tensor_name} from {old_block_timestamp} to {new_block_timestamp}")
            for name in [f'{tensor_name}', f'{tensor_name}_adam_m']:
                src = os.path.join(state_dir, f'{name}_{old_block_timestamp}.pt')
                dst = os.path.join(state_dir, f'{name}_{new_block_timestamp}.pt')
                if not os.path.exists(dst):
                    await asyncio.to_thread(shutil.copy, src, dst)

            block_timestamps[tensor_name] = new_block_timestamp
            await save_json(block_timestamps_file, block_timestamps, file_locks['block_timestamps'])
            saved_num_updates = num_updates.get(tensor_name, 0)

            num_updates[tensor_name] = 0
            await save_json(num_updates_file, num_updates, file_locks['num_updates'])
            saved_iteration_number = iteration_number.get(tensor_name, 0)
            iteration_number[tensor_name] = saved_iteration_number + 1
            await save_json(iteration_number_file, iteration_number, file_locks['iteration_number'])

            if last_future_version_number.get(tensor_name, 0) < future_version_number:
                last_future_version_number[tensor_name] = future_version_number
                await save_json(last_future_version_file, last_future_version_number, file_locks['last_future_version_number'])

            if saved_num_updates > 0:
                state_update_data = {
                    'num_updates': saved_num_updates,
                    'iteration_number': saved_iteration_number,
                }
                await db_adapter.create_state_update(job_id, state_update_data)
        return old_block_timestamp

async def cleanup_old_timestamp(tensor_name, old_block_timestamp, block_timestamps, state_dir):
    new_block_timestamp = block_timestamps.get(tensor_name, 0)
    if old_block_timestamp == new_block_timestamp:
        return
    if old_block_timestamp is not None:
        file_paths = [
            os.path.join(state_dir, f'{tensor_name}_{old_block_timestamp}.pt'),
            os.path.join(state_dir, f'{tensor_name}_adam_m_{old_block_timestamp}.pt')
        ]
        for file_path in file_paths:
            if os.path.exists(file_path):
                await asyncio.to_thread(os.remove, file_path)
            else:
                logging.warning(f"File not found: {file_path}")

async def update_cleanup_timestamps(tensor_name, block_timestamps, num_updates, iteration_number, last_future_version_number, state_dir, db_adapter, job_id, plugin, update_timestamp_lock, file_locks):
    old_block_timestamp = await update_block_timestamps(
        tensor_name, block_timestamps, num_updates, iteration_number,
        last_future_version_number, state_dir, db_adapter, job_id, plugin, update_timestamp_lock, file_locks
    )
    await cleanup_old_timestamp(tensor_name, old_block_timestamp, block_timestamps, state_dir)
