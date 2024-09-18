import asyncio
from collections import defaultdict
import os
import json
import logging
from quart import Quart, request, jsonify, send_file, Response, after_this_request
import torch
from io import BytesIO
import aiohttp
import time
import random
from werkzeug.utils import secure_filename
from tqdm import tqdm
from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3
import functools
from filelock import FileLock
import psutil
import tracemalloc
import aiofiles
import shutil

# Import your custom modules
from .common import (
    get_sot_learning_hyperparameters,
    model_adapter,
    batch_size,
    get_current_version_number,
    TENSOR_NAME,
    dataset,
    SOT_PRIVATE_PORT,
    get_future_version_number
)
from .device import device

# Constants for batch preloading
PRELOAD_BATCH_COUNT = 3  # Number of batches to preload
preloaded_batches_queue = asyncio.Queue(maxsize=PRELOAD_BATCH_COUNT)
dataset_iterator = None
dataset_lock = asyncio.Lock()

# File locks to prevent race conditions when reading/writing files
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, 'data')
state_dir = os.path.join(data_dir, 'state')
temp_dir = os.path.join(state_dir, 'temp')
os.makedirs(temp_dir, exist_ok=True)

block_timestamps_file = os.path.join(state_dir, 'block_timestamps.json')
num_updates_file = os.path.join(state_dir, 'num_updates.json')
last_future_version_file = os.path.join(state_dir, 'last_future_version_number.json')
iteration_number_file = os.path.join(state_dir, 'iteration_number.json')

block_timestamps_file_lock = FileLock(os.path.join(state_dir, 'block_timestamps.json.lock'))
num_updates_file_lock = FileLock(os.path.join(state_dir, 'num_updates.json.lock'))
last_future_version_file_lock = FileLock(os.path.join(state_dir, 'last_future_version_number.json.lock'))
iteration_number_file_lock = FileLock(os.path.join(state_dir, 'iteration_number.json.lock'))
used_nonces_file_lock = FileLock(os.path.join(state_dir, 'used_nonces.json.lock'))

# Initialize locks for thread safety
latest_loss_lock = asyncio.Lock()

SOT_FETCH_TIMEOUT = 300  # Timeout for fetching data from the SOT service

# Initialize logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logging.getLogger(__name__).setLevel(logging.INFO)

# Add a global variable to control memory logging
MEMORY_LOGGING_ENABLED = False


def log_memory_usage(note=''):
    """Log the current memory usage of the process if enabled."""
    if MEMORY_LOGGING_ENABLED:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        logging.debug(f"Memory usage ({note}): RSS={mem_info.rss / 1024 ** 2:.2f} MB, VMS={mem_info.vms / 1024 ** 2:.2f} MB")


def log_memory_diff(snapshot1, snapshot2, note=''):
    """Log the difference in memory usage between two snapshots if enabled."""
    if MEMORY_LOGGING_ENABLED:
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')

        logging.debug(f"Memory usage differences ({note}):")
        for stat in top_stats[:10]:  # Log top 10 memory differences
            logging.debug(stat)


def nag_update(params, grads, m, lr=0.002, weight_decay=0.2, beta1=0.9, eps=1e-6, step=1):
    """
    Performs a Nesterov Accelerated Gradient (NAG) update.

    Args:
    params (torch.Tensor): The current parameters.
    grads (torch.Tensor): The gradients of the parameters.
    m (torch.Tensor): The momentum from the previous step.
    lr (float): Learning rate.
    weight_decay (float): Weight decay (L2 regularization).
    beta1 (float): Momentum term coefficient.
    eps (float): A small epsilon value to avoid division by zero.
    step (int): Current step of the optimization.

    Returns:
    new_params (torch.Tensor): The updated parameters.
    new_m (torch.Tensor): The updated momentum.
    """

    # Apply weight decay to the gradients
    grads = grads + weight_decay * params

    # Nesterov lookahead step: params - beta1 * m
    lookahead_params = params - beta1 * m

    # Compute the updated momentum: m = beta1 * m + (1 - beta1) * grad
    new_m = beta1 * m + (1 - beta1) * grads

    # Update parameters using the velocity (new momentum)
    new_params = lookahead_params - lr * new_m

    return new_params, new_m


def create_app(public_keys_file, enable_memory_logging=False):
    """Create and configure the app."""
    global MEMORY_LOGGING_ENABLED
    MEMORY_LOGGING_ENABLED = enable_memory_logging

    if MEMORY_LOGGING_ENABLED:
        tracemalloc.start()  # Start tracing memory allocations

    app = Quart(__name__)

    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 * 1024  # 100 GB

    log_memory_usage('Before initializing or loading initial state')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    logging.info("Initializing or loading initial state...")
    os.makedirs(state_dir, exist_ok=True)

    # Initialize variables
    update_timestamp_lock = asyncio.Lock()
    master_public_keys = None

    async def save_json(file_path, data, file_lock):
        with file_lock:
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(data))

    async def load_json(file_path, default, file_lock):
        with file_lock:
            if os.path.exists(file_path):
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                    if not content.strip():  # Check if the file is empty
                        logging.error(f"The file {file_path} is empty. Returning default value.")
                        return default
                    try:
                        return json.loads(content)  # Try loading the JSON content
                    except json.JSONDecodeError as e:
                        logging.error(f"JSONDecodeError in file {file_path}: {e}. Returning default value.")
                        return default
            else:
                logging.info(f"The file {file_path} does not exist. Saving default value.")
                return default

    def set_dict_and_adam(dict_obj, tensor_name, value):
        dict_obj[tensor_name] = value
        dict_obj[f'{tensor_name}_adam_m'] = value

    async def initialize_tensor(name, sync_version_number=None, zero_init=False):
        logging.info(f"Initializing tensor {name}")
        log_memory_usage('Before initializing tensor')

        snapshot_before = tracemalloc.take_snapshot() if MEMORY_LOGGING_ENABLED else None  # Take snapshot before the operation

        block_timestamps = await load_json(block_timestamps_file, {}, block_timestamps_file_lock)
        last_future_version_number = await load_json(last_future_version_file, {}, last_future_version_file_lock)

        if sync_version_number is None:
            sync_version_number = block_timestamps.get(
                name, get_current_version_number())

        file_path = os.path.join(state_dir, f'{name}_{sync_version_number}.pt')
        if os.path.exists(file_path):
            logging.info(f"Tensor {name} already exists at version {sync_version_number}")
            return

        if TENSOR_NAME not in name:
            raise ValueError(f"Unsupported tensor name: {name}")

        tensor = model_adapter.init_tensor(zero_init)

        torch.save(tensor, file_path)
        block_timestamps[name] = sync_version_number
        await save_json(block_timestamps_file, block_timestamps, block_timestamps_file_lock)
        last_future_version_number[name] = sync_version_number
        await save_json(last_future_version_file, last_future_version_number, last_future_version_file_lock)

        if MEMORY_LOGGING_ENABLED:
            snapshot_after = tracemalloc.take_snapshot()  # Take snapshot after the operation
            log_memory_diff(snapshot_before, snapshot_after, note='After initializing tensor')  # Log memory differences

        log_memory_usage('After initializing tensor')

    async def initialize_all_tensors():
        logging.info("Initializing all tensors")
        await initialize_tensor(TENSOR_NAME, zero_init=False)
        await initialize_tensor(f'{TENSOR_NAME}_adam_m', zero_init=True)

    async def load_next_batch():
        """Load the next batch from the dataset."""
        batch = []
        targets = []
        global dataset_iterator

        async with dataset_lock:
            if dataset_iterator is None:
                dataset_iterator = dataset.__aiter__()  # Initialize the iterator

            try:
                for _ in range(batch_size):
                    inputs, target_tokens = await dataset_iterator.__anext__()
                    if isinstance(inputs, list):
                        inputs = torch.tensor(inputs)
                    if isinstance(target_tokens, list):
                        target_tokens = torch.tensor(target_tokens)
                    batch.append(inputs)
                    targets.append(target_tokens)
            except StopAsyncIteration:
                logging.info("Dataset iterator exhausted.")
                dataset_iterator = None  # Reset the iterator

        if not batch:
            return None, None  # No more data

        batch_tensor = torch.stack(batch)
        targets_tensor = torch.stack(targets)
        timestamp = int(time.time())
        random_suffix = random.randint(1000, 9999)
        batch_filename = f'batch_{timestamp}_{random_suffix}.pt'
        targets_filename = f'targets_{timestamp}_{random_suffix}.pt'
        torch.save(batch_tensor, os.path.join(temp_dir, batch_filename))
        torch.save(targets_tensor, os.path.join(temp_dir, targets_filename))
        return batch_filename, targets_filename

    async def batch_preloader():
        """Background task to preload batches and enqueue them."""
        while True:
            try:
                if preloaded_batches_queue.full():
                    await asyncio.sleep(0.1)  # Wait until space is available
                    continue
                batch_filename, targets_filename = await load_next_batch()
                if batch_filename is None:
                    logging.info("No more batches to preload.")
                    break
                await preloaded_batches_queue.put((batch_filename, targets_filename))
                logging.debug(f"Enqueued batch: {batch_filename}, {targets_filename}")
                log_memory_usage('Batch preloaded')
            except Exception as e:
                logging.error(f"Error preloading batch: {e}", exc_info=True)
                await asyncio.sleep(1)  # Wait before retrying

    async def initialize_service():
        nonlocal master_public_keys
        logging.info("Initializing distributed environment and tensors")

        async with aiofiles.open(public_keys_file, 'r') as f:
            master_public_keys = json.loads(await f.read())
        model_adapter.initialize_environment('gloo')
        await initialize_all_tensors()
        logging.info(f'Loading initial batches for service')
        # Start the background preloader task
        asyncio.create_task(batch_preloader())
        log_memory_usage('After initializing service')
        logging.info("Service initialized")

    @app.route('/health', methods=['GET'])
    async def health_check():
        log_memory_usage('Health check endpoint')
        return jsonify({'status': 'healthy'}), 200

    async def fetch(session, url):
        log_memory_usage('Before fetching URL')
        try:
            async with session.get(url, timeout=SOT_FETCH_TIMEOUT) as response:
                response.raise_for_status()
                return await response.read()
        except aiohttp.ClientError as e:
            logging.error(f"Error fetching {url}: {e}")
            raise
        finally:
            log_memory_usage('After fetching URL')

    def verify_signature(message, signature):
        nonlocal master_public_keys
        snapshot_before = tracemalloc.take_snapshot() if MEMORY_LOGGING_ENABLED else None  # Take snapshot before the operation

        message = encode_defunct(text=message)
        recovered_address = Account.recover_message(message, signature=signature)

        if MEMORY_LOGGING_ENABLED:
            snapshot_after = tracemalloc.take_snapshot()  # Take snapshot after the operation
            log_memory_diff(snapshot_before, snapshot_after, note='After verifying signature')  # Log memory differences

        logging.debug(f"Recovered address: {recovered_address}, Expected addresses: {master_public_keys}")
        return recovered_address.lower() in [key.lower() for key in master_public_keys]

    def requires_authentication(f):
        @functools.wraps(f)
        async def decorated_function(*args, **kwargs):
            auth_header = request.headers.get('Authorization')
            logging.debug(f"Authorization header: {auth_header}")
            if not auth_header:
                logging.error("Authorization header missing")
                return jsonify({'error': 'Authorization header missing'}), 401

            try:
                message, signature = auth_header.rsplit(':', 1)
            except ValueError:
                logging.error("Invalid Authorization header format")
                return jsonify({'error': 'Invalid Authorization header format'}), 401

            if not verify_signature(message, signature):
                logging.error("Invalid signature")
                return jsonify({'error': 'Invalid signature'}), 403

            # Parse the message to extract the nonce and timestamp
            try:
                message_data = json.loads(message)
                nonce = message_data['nonce']
                timestamp = message_data['timestamp']
                logging.debug(f"Message nonce: {nonce}, timestamp: {timestamp}")
            except (KeyError, json.JSONDecodeError):
                logging.error("Invalid message format")
                return jsonify({'error': 'Invalid message format'}), 401

            # Load the used nonces from file
            used_nonces = await load_json(os.path.join(state_dir, 'used_nonces.json'), {}, used_nonces_file_lock)

            # Check if the nonce has been used before
            if nonce in used_nonces:
                logging.error("Nonce already used")
                return jsonify({'error': 'Nonce already used'}), 403

            # Check if the message has expired (validity period of 5 minutes)
            current_time = int(time.time())
            if current_time - timestamp > 300:
                logging.error("Message expired")
                return jsonify({'error': 'Message expired'}), 403

            # Store the nonce to prevent reuse
            used_nonces[nonce] = True
            await save_json(os.path.join(state_dir, 'used_nonces.json'), used_nonces, used_nonces_file_lock)

            return await f(*args, **kwargs)
        return decorated_function

    synced_workers = 0

    @app.route('/report_sync', methods=['POST'])
    async def report_sync():
        nonlocal synced_workers
        synced_workers += 1
        return jsonify({'status': 'ok'})

    @app.route('/get_num_synced', methods=['GET'])
    async def get_num_synced():
        nonlocal synced_workers
        return jsonify(synced_workers)

    @app.route('/get_batch', methods=['POST'])
    @requires_authentication
    async def get_batch():
        logging.info("Accessing /get_batch endpoint")
        try:
            batch_filename, targets_filename = await preloaded_batches_queue.get()
            if batch_filename is None and targets_filename is None:
                # No more batches
                return jsonify({'error': 'No more batches available'}), 404
            return jsonify({
                'batch_url': f'/data/state/temp/{batch_filename}',
                'targets_url': f'/data/state/temp/{targets_filename}'
            })
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logging.error(f"Error in /get_batch: {e}", exc_info=True)
            return jsonify({'error': 'Could not get batch'}), 500

    async def apply_adamw(version_number, tensor_name, grads_flat, learning_rate, beta1, beta2, epsilon, weight_decay, t, clip_grad=1.0):
        tensor_path = os.path.join(state_dir, f'{tensor_name}_{version_number}.pt')
        if not os.path.exists(tensor_path):
            raise FileNotFoundError(f"Tensor file for {tensor_name} not found at {tensor_path}")

        tensor = torch.load(tensor_path, map_location=device)

        if tensor is None:
            raise ValueError(f"Failed to load tensor for {tensor_name}")

        # Ensure tensor is on the correct device and convert to flat tensor if necessary
        tensor = tensor.to(device)

        logging.debug(f"Tensor before AdamW: {tensor}")
        logging.debug(f"Flattened gradients: {grads_flat}")

        if torch.isnan(grads_flat).any() or torch.isinf(grads_flat).any():
            logging.error(f"NaNs or Infs detected in gradients before optimizer update for {tensor_name}")
            raise ValueError(f"NaNs or Infs detected in gradients for {tensor_name}")

        tensor_adam_m_path = os.path.join(state_dir, f'{tensor_name}_adam_m_{version_number}.pt')

        adam_m = None

        if os.path.exists(tensor_adam_m_path):
            adam_m = torch.load(tensor_adam_m_path, map_location=device).to(device)

        if adam_m is None:
            logging.debug(f'adam_m not found for {tensor_name}, initializing to zeros')
            adam_m = torch.zeros_like(tensor, device=device)

        logging.debug(f"m before optimizer: {adam_m}")

        clip_threshold = 1.0

        # Get the StableAdamW updates
        param_update, m_update = nag_update(
            tensor,
            grads_flat,
            adam_m,
            learning_rate,
            weight_decay,
            beta1,
            epsilon,
            t
        )

        logging.debug(f"Updates after applying optimizer: {param_update}")
        logging.debug(f"m after optimizer: {m_update}")

        return param_update.view(-1), m_update.view(-1)

    async def fix_outdated_last_future_version_number(tensor_name, last_future_version_number):
        value = last_future_version_number.get(tensor_name, 0)
        current_version_number = get_current_version_number()
        if value < current_version_number:
            if os.path.exists(os.path.join(state_dir, f'{tensor_name}_{value}.pt')):
                for f in f'{tensor_name}', f'{tensor_name}_adam_m':
                    await asyncio.to_thread(shutil.copy,
                        os.path.join(state_dir, f'{f}_{value}.pt'),
                        os.path.join(state_dir, f'{f}_{current_version_number}.pt')
                    )

            last_future_version_number[tensor_name] = get_current_version_number()
            await save_json(last_future_version_file, last_future_version_number, last_future_version_file_lock)

    async def update_block_timestamps(tensor_name, block_timestamps, num_updates, iteration_number, last_future_version_number):
        future_version_number = get_future_version_number()
        old_block_timestamp = None

        await fix_outdated_last_future_version_number(tensor_name, last_future_version_number)

        new_block_timestamp = last_future_version_number.get(tensor_name, 0)
        # the cause might be that last_future_version_number is not updated

        if new_block_timestamp < future_version_number and not update_timestamp_lock.locked():
            if new_block_timestamp > block_timestamps.get(tensor_name, 0):
                await update_timestamp_lock.acquire()
                old_block_timestamp = block_timestamps.get(tensor_name, 0)
                set_dict_and_adam(block_timestamps, tensor_name, new_block_timestamp)
                await save_json(block_timestamps_file, block_timestamps, block_timestamps_file_lock)

                for name in f'{tensor_name}', f'{tensor_name}_adam_m':
                    if not os.path.exists(os.path.join(state_dir, f'{name}_{new_block_timestamp}.pt')):
                        await asyncio.to_thread(shutil.copy,
                            os.path.join(state_dir, f'{name}_{old_block_timestamp}.pt'), 
                            os.path.join(state_dir, f'{name}_{new_block_timestamp}.pt')
                        )

            set_dict_and_adam(num_updates, tensor_name, 0)
            await save_json(num_updates_file, num_updates, num_updates_file_lock)

            set_dict_and_adam(iteration_number, tensor_name, iteration_number.get(tensor_name, 0) + 1)
            await save_json(iteration_number_file, iteration_number, iteration_number_file_lock)
        return old_block_timestamp

    async def cleanup_old_timestamp(tensor_name, old_block_timestamp, last_future_version_number):
        future_version_number = get_future_version_number()
        if last_future_version_number.get(tensor_name, 0) < future_version_number:
            set_dict_and_adam(last_future_version_number, tensor_name, future_version_number)
            await save_json(last_future_version_file, last_future_version_number, last_future_version_file_lock)

        if old_block_timestamp is not None:
            # Define the file paths
            file_paths = [
                os.path.join(state_dir, f'{tensor_name}_{old_block_timestamp}.pt'),
                os.path.join(state_dir, f'{tensor_name}_adam_m_{old_block_timestamp}.pt')
            ]

            # Remove each file if it exists
            for file_path in file_paths:
                if os.path.exists(file_path):
                    await asyncio.to_thread(os.remove, file_path)
                else:
                    logging.warning(f"File not found: {file_path}")
        if update_timestamp_lock.locked():
            update_timestamp_lock.release()

    async def update_cleanup_timestamps(tensor_name, block_timestamps, num_updates, iteration_number, last_future_version_number):
        old_block_timestamp = await update_block_timestamps(tensor_name, block_timestamps, num_updates, iteration_number, last_future_version_number)
        await cleanup_old_timestamp(tensor_name, old_block_timestamp, last_future_version_number)

    def get_local_file_path(url, request):
        if not url.startswith(f"http://{request.host}/data/state/"):
            logging.error(f"Invalid URL: {url}")
            return None
        path_relative_to_state = url.replace(f"http://{request.host}/data/state/", '')
        return os.path.join(state_dir, path_relative_to_state)

    @app.route('/update_state', methods=['POST'])
    @requires_authentication
    async def update_state():
        logging.info("Accessing /update_state endpoint")
        data = await request.get_json()
        tensor_name = data.get('tensor_name')
        result_url = data.get('result_url')

        logging.debug(f"Received tensor_name: {tensor_name}, version: {data['version_number']}, result_url: {result_url}")

        if not tensor_name or not result_url:
            logging.error("Missing tensor_name or result_url in /update_state request")
            return jsonify({'error': 'Missing tensor_name or result_url'}), 400

        # Load state from files
        block_timestamps = await load_json(block_timestamps_file, {}, block_timestamps_file_lock)
        num_updates = await load_json(num_updates_file, {}, num_updates_file_lock)
        last_future_version_number = await load_json(last_future_version_file, {}, last_future_version_file_lock)
        iteration_number = await load_json(iteration_number_file, {}, iteration_number_file_lock)

        future_version_number = get_future_version_number()

        if data['version_number'] != block_timestamps.get(tensor_name, 0):
            delta = block_timestamps.get(tensor_name, 0) - data['version_number']
            logging.info(f'Delta of {delta} recorded with version number {data["version_number"]}')
            return jsonify({'error': 'Version number mismatch'}), 409
        old_block_timestamp = await update_block_timestamps(tensor_name, block_timestamps, num_updates, iteration_number, last_future_version_number)

        logging.info(f"Future version number for {tensor_name}: {future_version_number}")

        try:
            local_file_path = get_local_file_path(result_url, request)
            batch_url = data.get('batch_url')
            targets_url = data.get('targets_url')

            local_batch_file_path = get_local_file_path(batch_url, request)
            local_targets_file_path = get_local_file_path(targets_url, request)
            logging.debug(f'Deleting batch and targets: {local_batch_file_path}, {local_targets_file_path}')
            await asyncio.to_thread(os.remove, local_batch_file_path)
            await asyncio.to_thread(os.remove, local_targets_file_path)
            logging.debug(f"Deleted batch and targets: {local_batch_file_path}, {local_targets_file_path}")

            if not os.path.exists(local_file_path):
                logging.error(f"File not found at {local_file_path}")
                return jsonify({'error': 'File not found'}), 404

            # Load the tensor from the local file path
            tensor = torch.load(local_file_path, map_location=device)

            # Paths for accumulated grads and future tensor
            accumulated_grads_path = os.path.join(state_dir, f'accumulated_grads_{tensor_name}_{future_version_number}.pt')
            future_tensor_path = os.path.join(state_dir, f'{tensor_name}_{future_version_number}.pt')
            unversioned_tensor_path = os.path.join(state_dir, f'{tensor_name}.pt')
            future_tensor_adam_m_path = os.path.join(state_dir, f'{tensor_name}_adam_m_{future_version_number}.pt')

            # Load or initialize the accumulated_grads tensor
            if os.path.exists(accumulated_grads_path):
                accumulated_grads = torch.load(accumulated_grads_path, map_location=device).to(device)
            else:
                accumulated_grads = torch.zeros_like(tensor, device=device)

            # Update the accumulated_grads tensor
            accumulated_grads += tensor.to(device)
            torch.save(accumulated_grads, accumulated_grads_path)

            # Calculate the future tensor
            current_version_number = block_timestamps.get(tensor_name, 0)
            logging.info(f'Updating state for {tensor_name}, future version number: {future_version_number}, current version number: {current_version_number}')

            num_of_updates = num_updates[tensor_name] + 1
            set_dict_and_adam(num_updates, tensor_name, num_of_updates)
            await save_json(num_updates_file, num_updates, num_updates_file_lock)

            averaged_grads = (accumulated_grads / num_of_updates).to(device)
            learning_params = get_sot_learning_hyperparameters(iteration_number[tensor_name])
            future_tensor, m_update = await apply_adamw(
                current_version_number,
                tensor_name,
                averaged_grads,
                learning_params['learning_rate'],
                learning_params['beta1'],
                learning_params['beta2'],
                learning_params['epsilon'],
                learning_params['weight_decay'],
                learning_params['t']
            )

            torch.save(future_tensor, future_tensor_path)
            torch.save(future_tensor, unversioned_tensor_path)
            torch.save(m_update, future_tensor_adam_m_path)

            await cleanup_old_timestamp(tensor_name, old_block_timestamp, last_future_version_number)
            # Cleanup old accumulated grads tensors
            for filename in os.listdir(state_dir):
                if filename.startswith(f'accumulated_grads_{tensor_name}_') and not filename.endswith(f'{future_version_number}.pt'):
                    await asyncio.to_thread(os.remove, os.path.join(state_dir, filename))

            logging.debug(f"Updated state for {tensor_name} version {future_version_number} with {num_of_updates} updates")

            # Delete the file corresponding to result_url after processing
            if os.path.exists(local_file_path):
                await asyncio.to_thread(os.remove, local_file_path)
                logging.info(f"Deleted file: {local_file_path}")
            else:
                logging.warning(f"File not found for deletion: {local_file_path}")

            return jsonify({'status': 'success', 'version_number': future_version_number})
        except aiohttp.ClientError as e:
            logging.error(f"Failed to update tensor {tensor_name} due to request exception: {e}", exc_info=True)  # Add exc_info=True
        except Exception as e:
            logging.error(f"Failed to update tensor {tensor_name} due to error: {e}", exc_info=True)  # Add exc_info=True
        return jsonify({'error': 'Could not update state'}), 500

    def version_number_exists(version_number, tensor_name):
        return os.path.exists(os.path.join(state_dir, f'{tensor_name}_{version_number}.pt'))

    @app.route('/latest_state', methods=['GET'])
    async def latest_state():
        logging.info("Accessing /latest_state endpoint")
        tensor_name = request.args.get('tensor_name')
        if not tensor_name:
            return jsonify({'error': 'Missing tensor_name parameter'}), 400

        latest_version_number = request.args.get('version_number')
        if latest_version_number is None or not version_number_exists(latest_version_number, tensor_name):
            block_timestamps = await load_json(block_timestamps_file, {}, block_timestamps_file_lock)
            latest_version_number = block_timestamps.get(tensor_name, 0)
        else:
            latest_version_number = int(latest_version_number)

        state_file_path = os.path.join(state_dir, f'{tensor_name}_{latest_version_number}.pt')

        if not os.path.exists(state_file_path):
            return jsonify({'error': 'Tensor not found'}), 404

        try:
            @after_this_request
            def add_header(response):
                response.headers['version_number'] = str(latest_version_number)
                return response

            return await send_file(state_file_path, mimetype='application/octet-stream')

        except Exception as e:
            logging.error(f"Error in /latest_state: {e}", exc_info=True)
            return jsonify({'error': 'Could not retrieve latest state'}), 500

    @app.route('/current_timestamp', methods=['POST'])
    async def current_timestamp():
        logging.info("Accessing /current_timestamp endpoint")
        tensor_name = request.args.get('tensor_name')
        if not tensor_name:
            return jsonify({'error': 'Missing tensor_name parameter'}), 400

        block_timestamps = await load_json(block_timestamps_file, {}, block_timestamps_file_lock)
        num_updates = await load_json(num_updates_file, {}, num_updates_file_lock)
        iteration_number = await load_json(iteration_number_file, {}, iteration_number_file_lock)
        last_future_version_number = await load_json(last_future_version_file, {}, last_future_version_file_lock)
        await update_cleanup_timestamps(tensor_name, block_timestamps, num_updates, iteration_number, last_future_version_number)

        latest_version_number = block_timestamps.get(tensor_name, 0)
        return jsonify({'version_number': latest_version_number})

    @app.route('/tensor_size', methods=['GET'])
    async def get_tensor_size():
        logging.info("Accessing /tensor_size endpoint")
        tensor_name = request.args.get('tensor_name')
        if not tensor_name:
            return jsonify({'error': 'Missing tensor_name parameter'}), 400

        state_file_path = os.path.join(state_dir, f'{tensor_name}.pt')
        if not os.path.exists(state_file_path):
            return jsonify({'error': 'Tensor not found'}), 404

        tensor = torch.load(state_file_path, map_location=device)
        size = tensor.numel()
        return jsonify({'size': size})

    @app.route('/data/state/<path:filename>', methods=['GET'])
    async def get_data_file(filename):
        logging.info(f"Accessing file: {filename}")
        file_path = os.path.join(state_dir, filename)

        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404

        try:
            # Use send_file to handle file streaming and headers
            return await send_file(
                file_path,
                mimetype='application/octet-stream',
                as_attachment=True
            )
        except Exception as e:
            logging.error(f"Error accessing file {filename}: {e}", exc_info=True)
            return jsonify({'error': 'File not found or could not be read'}), 404

    @app.route('/upload_tensor', methods=['POST'])
    async def upload_tensor():
        request_files = await request.files
        if 'tensor' not in request_files:
            return jsonify({'error': 'No tensor file provided'}), 400

        request_form = await request.form
        if 'label' not in request_form:
            return jsonify({'error': 'No label provided'}), 400

        tensor_file = request_files['tensor']
        label = request_form['label']
        update_version_number = int(time.time())
        random_suffix = random.randint(1000, 9999)
        filename = secure_filename(f'{label}_{update_version_number}_{random_suffix}.pt')
        local_file_path = os.path.join(temp_dir, filename)

        logging.debug("Receiving tensor upload...")
        total_size = request.content_length
        chunk_size = 1024 * 1024  # 1MB

        async with aiofiles.open(local_file_path, 'wb') as f:
            while True:
                chunk = tensor_file.read(chunk_size)
                if not chunk:
                    break
                await f.write(chunk)
        logging.debug("Tensor upload completed.")

        tensor_name = filename.split('.')[0]

        # Save the tensor state
        tensor_state = torch.load(local_file_path, map_location=device)
        torch.save(tensor_state, os.path.join(temp_dir, filename))  # Use filename directly

        logging.debug(f"Tensor {tensor_name} uploaded and saved with version_number {update_version_number}")

        return jsonify({'message': 'Tensor uploaded successfully', 'tensor_url': f'/data/state/temp/{filename}'}), 200

    latest_loss = None

    @app.route('/update_loss', methods=['POST'])
    async def update_loss():
        nonlocal latest_loss

        # Get the loss value from the request data
        data = await request.get_json()
        if not data or 'loss' not in data:
            return jsonify({'error': 'Missing loss value'}), 400

        loss = data['loss']

        # Update the loss value with thread-safety
        async with latest_loss_lock:
            latest_loss = await load_json(os.path.join(state_dir, 'latest_loss.json'), {'value': None}, block_timestamps_file_lock)
            latest_loss['value'] = loss
            await save_json(os.path.join(state_dir, 'latest_loss.json'), latest_loss, block_timestamps_file_lock)

        logging.info(f"Updated latest loss for version {data['version_number']}: {loss}")
        return jsonify({'status': 'success'}), 200

    @app.route('/get_loss', methods=['GET'])
    async def get_loss():
        logging.info("Accessing /get_loss endpoint")
        latest_loss = await load_json(os.path.join(state_dir, 'latest_loss.json'), {'value': None}, block_timestamps_file_lock)
        logging.info(f"Retrieved loss: {latest_loss.get('value')}")
        loss = latest_loss.get('value')
        return jsonify({'loss': loss}), 200

    def initialize():
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop is available, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(initialize_service())

    initialize()
    return app

if __name__ == "__main__":
    import argparse
    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    def main():
        parser = argparse.ArgumentParser(description="Source of Truth (SOT) Service")
        parser.add_argument('--public_keys_file', type=str, required=True, help="Path to the file containing public keys of the master for verifying requests")
        parser.add_argument('--enable_memory_logging', action='store_true', help="Enable memory logging")

        args = parser.parse_args()

        # Create the app with the memory logging flag
        app = create_app(args.public_keys_file, enable_memory_logging=args.enable_memory_logging)

        logging.info("Starting SOT service...")

        config = Config()
        config.bind = [f'0.0.0.0:{SOT_PRIVATE_PORT}']

        # Correctly pass the app callable to serve
        asyncio.run(serve(app, config))

    main()
