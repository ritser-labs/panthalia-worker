# spl/sot/utils.py
import os
import asyncio
import logging
import psutil
from ..device import device
from ..common import (
    get_current_version_number,
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

def get_local_file_path(url, request, state_dir):
    if not url.startswith(f"http://{request.host}/data/state/"):
        logging.error(f"Invalid URL: {url}")
        return None
    path_relative_to_state = url.replace(f"http://{request.host}/data/state/", '')
    return os.path.join(state_dir, path_relative_to_state)


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

