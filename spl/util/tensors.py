import os
import logging
import torch
from ..util.json import load_json, save_json
from ..common import TENSOR_NAME, device

async def initialize_tensor(name, state_dir, model_adapter, zero_init=False, memory_logging=False, file_locks=None):
    """
    Initialize a single tensor's state file if it does not already exist.
    This sets version numbers and saves the initial tensor parameters.
    """

    block_timestamps_file = os.path.join(state_dir, 'block_timestamps.json')
    last_future_version_file = os.path.join(state_dir, 'last_future_version_number.json')

    # Load JSON files safely with locks
    block_timestamps = await load_json(block_timestamps_file, {}, file_locks['block_timestamps'])
    last_future_version_number = await load_json(last_future_version_file, {}, file_locks['last_future_version_number'])

    # Determine the initial sync_version_number
    # If the tensor doesn't exist, default to version 0
    sync_version_number = block_timestamps.get(name, 0)

    file_path = os.path.join(state_dir, f'{name}_{sync_version_number}.pt')
    if os.path.exists(file_path):
        logging.info(f"Tensor {name} already exists at version {sync_version_number}")
        return

    # Initialize the tensor using the model_adapter
    tensor = model_adapter.init_tensor(zero_init=zero_init)
    torch.save(tensor, file_path)

    # Update block_timestamps and last_future_version_number
    block_timestamps[name] = sync_version_number
    await save_json(block_timestamps_file, block_timestamps, file_locks['block_timestamps'])

    last_future_version_number[name] = sync_version_number
    await save_json(last_future_version_file, last_future_version_number, file_locks['last_future_version_number'])

    logging.info(f"Tensor {name} initialized at version {sync_version_number}")

async def initialize_all_tensors(state_dir, model_adapter, memory_logging=False, file_locks=None):
    """
    Initialize all required tensors at startup.
    This typically includes the main model tensor and its optimizer states (e.g., adam_m).
    """

    logging.info("Initializing all tensors")

    # Initialize the main parameter tensor
    await initialize_tensor(TENSOR_NAME, state_dir, model_adapter, zero_init=False, memory_logging=memory_logging, file_locks=file_locks)

    # Initialize the ADAM momentum tensor
    await initialize_tensor(f'{TENSOR_NAME}_adam_m', state_dir, model_adapter, zero_init=True, memory_logging=memory_logging, file_locks=file_locks)

    logging.info("All tensors initialized successfully")
