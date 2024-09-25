import argparse
import json
import logging
import aiohttp
import torch
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass
from web3 import AsyncWeb3
from web3.exceptions import ContractCustomError
from web3.middleware import async_geth_poa_middleware
from collections import defaultdict
from .device import device
from .common import (
    Task, TaskStatus, model_adapter, load_abi, upload_tensor,
    get_current_version_number, async_transact_with_contract_function,
    expected_worker_time, TENSOR_NAME, PoolState, approve_token_once,
    deposit_stake_without_approval, get_future_version_number,
    CHUNK_SIZE
)
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from typing import Optional, Tuple
from io import BytesIO
import time
import os
from tqdm import tqdm
import asyncio
import torch._dynamo
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor
import heapq
import itertools

# Removed threading imports
# from concurrent.futures import ThreadPoolExecutor
# import threading
from .util.queued_lock import AsyncQueuedLock

# Define the maximum number of retries for task processing
MAX_WORKER_TASK_RETRIES = 3

class SuppressTracebackFilter(logging.Filter):
    def filter(self, record):
        if 'ConnectionRefusedError' in record.getMessage() or 'MaxRetryError' in record.getMessage():
            return False
        return True

# Set up logging with timestamps
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.addFilter(SuppressTracebackFilter())

# Initialize asyncio Locks
concurrent_tasks_counter = 0
concurrent_tasks_counter_lock = asyncio.Lock()  # Lock for concurrent_tasks_counter

# Global dictionary to store start times for task IDs
task_start_times = {}
task_start_times_lock = asyncio.Lock()          # Lock for task_start_times

# Global variable to store the last handle_event timestamp
last_handle_event_timestamp = None
last_handle_event_timestamp_lock = asyncio.Lock()  # Lock for last_handle_event_timestamp

# Initialize AsyncQueuedLock instances
task_processing_lock = AsyncQueuedLock()
upload_lock = AsyncQueuedLock()

# Initialize a lock to prioritize tensor downloads
tensor_download_lock = asyncio.Lock()

# Initialize an event to signal tensor download in progress
tensor_download_event = asyncio.Event()

def parse_args():
    parser = argparse.ArgumentParser(description="Worker for processing tasks based on smart contract events")
    parser.add_argument('--task_types', type=str, required=True, help="Types of tasks to process, separated by '+' if multiple")
    parser.add_argument('--subnet_addresses', type=str, required=True, help="Subnet contract addresses")
    parser.add_argument('--private_keys', type=str, required=True, help="Private keys of the worker's Ethereum accounts")
    parser.add_argument('--rpc_url', type=str, default='http://localhost:8545', help="URL of the Ethereum RPC node")
    parser.add_argument('--sot_url', type=str, required=True, help="Source of Truth URL for streaming gradient updates")
    parser.add_argument('--pool_address', type=str, required=True, help="Pool contract address")
    parser.add_argument('--group', type=int, required=True, help="Group for depositing stake")
    parser.add_argument('--backend', type=str, default='nccl', help="Distributed backend to use (default: nccl, use 'gloo' for macOS)")
    parser.add_argument('--layer_idx', type=int, help="Layer index for forward and backward tasks", required=False)
    parser.add_argument('--detailed_logs', action='store_true', help="Enable detailed logging for loss task")
    parser.add_argument('--max_stakes', type=int, default=2, help="Maximum number of stakes to maintain")
    parser.add_argument('--poll_interval', type=int, default=1, help="Interval (in seconds) for polling the smart contract for new tasks")
    parser.add_argument('--torch_compile', action='store_true', help="Enable torch.compile and model warmup")
    parser.add_argument('--max_tasks_handling', type=int, default=1, help="Maximum number of tasks allowed in the queue awaiting processing")
    return parser.parse_args()

args = parse_args()

# Split combined task types if any
task_types = args.task_types.split('+')
args.task_types = task_types

subnet_addresses = args.subnet_addresses.split('+')
args.subnet_addresses = subnet_addresses

private_keys = args.private_keys.split('+')
args.private_keys = private_keys

web3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(args.rpc_url))
web3.middleware_onion.inject(async_geth_poa_middleware, layer=0)

worker_accounts = [web3.eth.account.from_key(key) for key in args.private_keys]
worker_addresses = [account.address for account in worker_accounts]

subnet_manager_abi = load_abi('SubnetManager')
pool_abi = load_abi('Pool')

contracts = [web3.eth.contract(address=address, abi=subnet_manager_abi) for address in args.subnet_addresses]
pool_contract = web3.eth.contract(address=args.pool_address, abi=pool_abi)

model_initialized = False
embedding_initialized = False
latest_block_timestamps = defaultdict(lambda: 0)  # To store the latest block timestamp processed for each tensor
processed_tasks = set()

latest_model = None

class TaskQueue:
    def __init__(self):
        self.queue = []
        self.current_version = None
        self.lock = asyncio.Lock()  # Use asyncio.Lock
        logging.debug("Initialized TaskQueue")

    async def add_task(self, task):
        async with self.lock:  # Ensure thread-safe access to the queue
            self.queue.append(task)
            self.queue.sort(key=lambda t: t['time_status_changed'])
            how_old = int(time.time()) - task['time_status_changed']
            logging.debug(f"Added task: {task} that is {how_old} seconds old. Queue size is now {len(self.queue)}")

    async def get_next_task(self):
        async with self.lock:  # Ensure thread-safe access to the queue
            if self.queue:
                task = self.queue.pop(0)
                logging.debug(f"Retrieved task: {task}. Queue size is now {len(self.queue)}")
                return task
            logging.debug("No tasks in the queue.")
            return None

    async def queue_length(self):
        async with self.lock:
            return len(self.queue)

task_queue = TaskQueue()

async def download_file(url, retries=3, backoff=1, chunk_timeout=5, download_type='batch_targets', tensor_name=None):
    """
    Downloads a file with retry logic and prioritizes tensor downloads over batch/targets downloads.
    
    Args:
        url (str): The URL to download the file from.
        retries (int): Number of retry attempts.
        backoff (int): Backoff factor for retries.
        chunk_timeout (int): Timeout for each chunk in seconds.
        download_type (str): Type of download ('tensor' or 'batch_targets').
    
    Returns:
        torch.Tensor: The downloaded tensor.
    """
    params = {'tensor_name': tensor_name} if tensor_name else None
    for attempt in range(1, retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()

                    if download_type == 'tensor':
                        # Acquire tensor_download_lock to prioritize tensor downloads
                        async with tensor_download_lock:
                            # Signal that a tensor download is in progress
                            tensor_download_event.clear()
                            try:
                                content = await download_with_timeout(response, chunk_size=1024 * 1024, chunk_timeout=chunk_timeout, download_type=download_type)
                            finally:
                                # Clear the event after tensor download is complete
                                tensor_download_event.set()
                    elif download_type == 'batch_targets':
                        # Wait for any ongoing tensor download to finish
                        async with tensor_download_lock:
                            pass  # Simply wait until tensor_download_lock is free
                        content = await download_with_timeout(response, chunk_size=1024 * 1024, chunk_timeout=chunk_timeout, download_type=download_type)
                    else:
                        raise ValueError("Invalid download_type specified.")

                    return torch.load(content)

        except asyncio.TimeoutError:
            logging.error(f"Attempt {attempt}: Chunk download timed out.")
        except aiohttp.ClientError as e:
            logging.error(f"Attempt {attempt}: Client error: {e}")
        except Exception as e:
            logging.error(f"Attempt {attempt}: Unexpected error: {e}")

        if attempt < retries:
            await asyncio.sleep(backoff * attempt)

    raise Exception(f"Failed to download file after {retries} attempts")

def create_callback(encoder, pbar):
    def callback(monitor):
        pbar.update(monitor.bytes_read - pbar.n)
    return callback

async def upload_tensor(tensor, tensor_name, retries=3, backoff=1):
    tensor_bytes = BytesIO()
    torch.save(tensor, tensor_bytes)
    tensor_bytes.seek(0)

    encoder = MultipartEncoder(
        fields={
            'tensor': (tensor_name, tensor_bytes, 'application/octet-stream'),
            'label': tensor_name
        }
    )

    pbar = tqdm(total=encoder.len, unit='B', unit_scale=True, desc='Uploading')
    monitor = MultipartEncoderMonitor(encoder, create_callback(encoder, pbar))

    headers = {'Content-Type': monitor.content_type}

    for attempt in range(1, retries + 1):
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f'{args.sot_url}/upload_tensor',
                    data=monitor,
                    headers=headers,
                    timeout=300
                )
            )
            pbar.close()

            if response.status_code == 200:
                return args.sot_url + response.json().get('tensor_url')
            else:
                raise RuntimeError(f"Failed to upload tensor: {response.text}")

        except requests.exceptions.Timeout:
            logging.error(f"Attempt {attempt}: Upload request timed out.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Attempt {attempt}: Upload request failed: {e}")

        if attempt < retries:
            await asyncio.sleep(backoff * attempt)

    raise RuntimeError(f"Failed to upload tensor {tensor_name} after {retries} attempts")

async def deposit_stake():
    global concurrent_tasks_counter
    # Do not deposit stakes if queued tasks exceed the limit
    if (await task_queue.queue_length() + concurrent_tasks_counter) > args.max_tasks_handling:
        logging.debug("Too many tasks being processed. Not depositing any more stakes.")
        return

    wallets = zip(args.private_keys, subnet_ids, stake_amounts, token_contracts, pool_contracts, worker_addresses)
    
    for private_key, subnet_id, stake_amount, token_contract, pool_contract, worker_address in wallets:
        await deposit_stake_without_approval(
            web3, pool_contract, private_key, subnet_id,
            args.group, worker_address, stake_amount, args.max_stakes
        )

async def handle_event(task_id, task, time_invoked, contract_index):
    global last_handle_event_timestamp

    current_time = time.time()
    logging.debug(f'Time since invocation: {current_time - time_invoked:.2f} seconds')
    async with last_handle_event_timestamp_lock:
        if last_handle_event_timestamp is not None:
            time_since_last_event = current_time - last_handle_event_timestamp
            logging.debug(f"Time since last handle_event call: {time_since_last_event:.2f} seconds")
        last_handle_event_timestamp = current_time

    solver = task.solver

    logging.info(f"Received event for task {args.task_types[contract_index]} and id {task_id} and layer {args.layer_idx}")

    if solver.lower() != worker_addresses[contract_index].lower():
        logging.debug("Solver address does not match worker address. Ignoring event.")
        return

    task_params_bytes = task.params
    task_params = json.loads(task_params_bytes.decode('utf-8'))

    logging.debug(f"Adding task to queue with ID: {task_id} and params: {task_params}")

    await task_queue.add_task({
        'task_id': task_id,
        'task_params': task_params,
        'time_status_changed': task.timeStatusChanged,
        'contract_index': contract_index
    })
    
    blockchain_timestamp = (await web3.eth.get_block('latest'))['timestamp']
    
    time_since_change = blockchain_timestamp - task.timeStatusChanged
    logging.debug(f"Time since status change: {time_since_change} seconds")
    
    async with task_start_times_lock:
        task_start_times[task_id] = time.time()
    await process_tasks()

def tensor_memory_size(tensor):
    # Calculate size in bytes and convert to megabytes
    size_in_bytes = tensor.element_size() * tensor.numel()
    size_in_mb = size_in_bytes / (1024 * 1024)
    return size_in_mb

async def process_tasks():
    global task_queue, concurrent_tasks_counter
    retry_attempt = 0  # Initialize retry counter
    task_success = False  # Track success status
    
    if await task_queue.queue_length() == 0:
        logging.debug("No tasks in the queue to process.")
        return

    async with concurrent_tasks_counter_lock:
        # Increase the counter when a new task is started
        concurrent_tasks_counter += 1
    try:
        # Get the next task from the queue
        next_task = await task_queue.get_next_task()
                
        while retry_attempt < MAX_WORKER_TASK_RETRIES and not task_success:
            try:
                if not next_task:
                    logging.debug("No tasks in the queue to process.")
                    return

                task_id = next_task['task_id']
                task_params = next_task['task_params']
                contract_index = next_task['contract_index']
                time_status_changed = next_task['time_status_changed']  # Extract the time_status_changed

                logging.debug(f"{task_id}: Processing task with params: {task_params} and contract_index: {contract_index}")

                start_time = time.time()
                # Downloading batch and targets (asynchronously)
                logging.debug(f"{task_id}: Downloading batch from URL: {task_params['batch_url']}")
                batch = await download_file(task_params['batch_url'], download_type='batch_targets')

                logging.debug(f"{task_id}: Downloading targets from URL: {task_params['targets_url']}")
                targets = await download_file(task_params['targets_url'], download_type='batch_targets')

                try:
                    # Acquire the lock asynchronously with priority based on time_status_changed
                    await task_processing_lock.acquire(priority=time_status_changed)
                    try:
                        # Process the task
                        steps = task_params['steps']
                        max_lr = task_params['max_lr']
                        min_lr = task_params['min_lr']
                        T_0 = task_params['T_0']
                        weight_decay = task_params['weight_decay']
                        logging.debug(f"{task_id}: Executing training task")
                        time_synced = time.time()
                        model, version_number = await sync_tensors(contract_index)

                        updates, loss = model_adapter.train_task(
                            model, batch, targets, steps, max_lr, min_lr, T_0, weight_decay
                        )
                        logging.info(f"{task_id}: Updates tensor memory size: {tensor_memory_size(updates):.2f} MB")
                    finally:
                        await task_processing_lock.release()
                
                except Exception as e:
                    logging.error(f"Error during task processing: {e}")
                    raise

                try:
                    # Upload the result asynchronously with priority based on time_status_changed
                    await upload_lock.acquire(priority=time_status_changed)
                    try:
                        upload_start_time = time.time()
                        result = await upload_results(version_number, updates, loss)
                        upload_end_time = time.time()
                        logging.info(f"{task_id}: Uploaded results for task {task_id} in {upload_end_time - upload_start_time:.2f} seconds")
                    finally:
                        await upload_lock.release()
                except Exception as e:
                    logging.error(f"Error during uploading results: {e}")
                    raise

                # Submit the solution
                await submit_solution(task_id, result, contract_index)

                processed_tasks.add((task_id, contract_index))

                # Log the time taken to process the task
                async with task_start_times_lock:
                    task_start_time = task_start_times.pop(task_id, None)
                if task_start_time:
                    total_time = time.time() - task_start_time
                    logging.info(f"{task_id}: Total time to process: {total_time:.2f} seconds")
                    logging.info(f'{task_id}: Time since sync: {time.time() - time_synced:.2f} seconds')

                end_time = time.time()
                logging.info(f"{task_id}: process_tasks() completed in {end_time - start_time:.2f} seconds. Concurrent tasks: {concurrent_tasks_counter}")

                task_success = True  # Mark the task as successful if no exceptions occurred

            except Exception as e:
                retry_attempt += 1
                logging.error(f"Error processing task (attempt {retry_attempt}): {e}", exc_info=True)
                if retry_attempt >= MAX_WORKER_TASK_RETRIES:
                    logging.error(f"Max retries reached for task {task_id}.")
                    raise
                logging.info(f"Retrying task processing (attempt {retry_attempt + 1}/{MAX_WORKER_TASK_RETRIES})...")
        
    finally:
        async with concurrent_tasks_counter_lock:
            concurrent_tasks_counter -= 1

async def reclaim_stakes():
    while True:
        for task_id, contract_index in list(processed_tasks):
            contract = contracts[contract_index]
            task_tuple = await contract.functions.getTask(task_id).call()
            task = Task(*task_tuple)
            task_status = task.status
            time_status_changed = task.timeStatusChanged
            blockchain_timestamp = (await web3.eth.get_block('latest'))['timestamp']
            time_elapsed = blockchain_timestamp - time_status_changed
            
            if task_status == TaskStatus.SolutionSubmitted.value and time_elapsed >= max_dispute_times[contract_index]:
                try:
                    receipt = await async_transact_with_contract_function(
                        web3, contract, 'resolveTask',
                        args.private_keys[contract_index], task_id
                    )
                    logging.info(f"resolveTask transaction receipt: {receipt}")
                    processed_tasks.remove((task_id, contract_index))
                except Exception as e:
                    logging.error(f"Error resolving task {task_id}: {e}")
                    # Depending on requirements, you might want to continue instead of raising
                    # raise

        await asyncio.sleep(2)

async def sync_tensors(contract_index):
    relevant_tensor = get_relevant_tensor_for_task(args.task_types[contract_index])
    return await initialize_tensor(relevant_tensor)

async def submit_solution(task_id, result, contract_index):
    try:
        logging.info('Submitting solution')
        receipt = await async_transact_with_contract_function(
            web3, contracts[contract_index], 'submitSolution',
            args.private_keys[contract_index], task_id,
            json.dumps(result).encode('utf-8')
        )
        logging.info(f"submitSolution transaction receipt: {receipt}")
    except Exception as e:
        logging.error(f"Error submitting solution for task {task_id}: {e}")
        raise

async def upload_results(version_number, updates, loss):
    grads_url = await upload_tensor(updates, 'grads')

    return {
        'grads_url': grads_url,
        'loss': loss,
        'version_number': version_number
    }

async def report_sync_status():
    logging.info(f'Reporting sync status to {args.sot_url}')
    try:
        url = f"{args.sot_url}/report_sync"
        async with aiohttp.ClientSession() as session:
            async with session.post(url) as response:
                await response.text()
    except aiohttp.ClientError as e:
        logging.error(f"Exception while reporting sync status: {e}")

def time_until_next_version():
    return get_future_version_number() - time.time()

async def download_with_timeout(response, chunk_size=1024 * 1024, chunk_timeout=5, download_type='batch_targets'):
    """
    Downloads data from the response stream with a timeout for each chunk.
    Pauses if a tensor download is in progress.

    Args:
        response: The aiohttp response object.
        chunk_size: The size of each chunk to download.
        chunk_timeout: Timeout for each chunk in seconds.
        download_type: Type of download ('tensor' or 'batch_targets').

    Returns:
        A BytesIO object containing the downloaded data.
    """
    start_time = time.time()
    content = BytesIO()
    
    # Get the content length from the header, if available
    content_length = response.headers.get('Content-Length', None)
    if content_length:
        total_size = int(content_length)
        logging.debug(f"Total file size (Content-Length): {total_size} bytes")
    else:
        # No Content-Length header, could be chunked transfer encoding
        total_size = None
        logging.debug("No Content-Length header. Assuming chunked transfer encoding.")
    
    downloaded_size = 0
    next_progress = 0.1

    # Fetch each chunk with a timeout
    while True:
        try:
            chunk = await asyncio.wait_for(response.content.read(chunk_size), timeout=chunk_timeout)
        except asyncio.TimeoutError:
            logging.error(f"Chunk download timed out after {chunk_timeout} seconds")
            raise

        if not chunk:
            # No more chunks left to download
            logging.debug("No more chunks to download. Download finished.")
            break

        if download_type == 'batch_targets':
            await tensor_download_event.wait()

        content.write(chunk)
        downloaded_size += len(chunk)
        #logging.debug(f"Downloaded chunk size: {len(chunk)} bytes. Total downloaded: {downloaded_size} bytes")

        # If we have the total size, we can log progress
        if total_size:
            progress = downloaded_size / total_size
            if progress >= next_progress:
                logging.info(f"Downloaded {int(progress * 100)}%")
                next_progress += 0.1

    content.seek(0)  # Reset the stream position

    # Validate that the entire content was downloaded, if we know the total size
    if total_size and downloaded_size != total_size:
        logging.error(f"Downloaded size ({downloaded_size}) does not match expected size ({total_size}).")
        raise Exception(f"Incomplete download: expected {total_size} bytes but got {downloaded_size} bytes")

    end_time = time.time()
    logging.info(f"Download completed successfully in {start_time - end_time:.2f} seconds. Total size: {downloaded_size} bytes")
    return content

async def initialize_tensor(tensor_name, retries=3, backoff=1, chunk_timeout=5):
    global latest_model
    logging.debug(f"Starting initialization for tensor: {tensor_name}")

    init_start_time = time.time()
    valid_version = False
    max_iterations = 10
    iterations = 0
    timeout = aiohttp.ClientTimeout(total=200)

    while not valid_version and iterations < max_iterations:
        iterations += 1
        logging.debug(f"Initialization loop iteration {iterations}")
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{args.sot_url}/current_timestamp",
                    params={'tensor_name': tensor_name}
                ) as response:
                    response.raise_for_status()
                    version_number = (await response.json())['version_number']
                    logging.debug(f"Received version_number: {version_number}")
        except aiohttp.ClientError as e:
            logging.error(f"Error fetching current_timestamp: {e}")
            await asyncio.sleep(backoff * iterations)
            continue

        time_until_next = time_until_next_version()
        logging.debug(f"Time until next version: {time_until_next} seconds")

        if time_until_next < expected_worker_time:
            logging.debug(f'Not enough time left until next version. Waiting for {time_until_next} seconds.')
            await asyncio.sleep(time_until_next)
        else:
            valid_version = True

    if not valid_version:
        raise RuntimeError("initialize_tensor: failed to get a valid version")

    for attempt in range(1, retries + 1):
        try:
            url = f"{args.sot_url}/latest_state"
            logging.debug(f"Requesting tensor {tensor_name} from URL: {url}")
            fetch_start_time = time.time()

            # Use the modified download_file with download_type='tensor'
            tensor = await download_file(url, tensor_name=tensor_name, download_type='tensor')
            download_end_time = time.time()
            logging.debug(f"Downloaded in {download_end_time - fetch_start_time:.2f} seconds")

            model = model_adapter.tensor_to_model(tensor.detach(), latest_model)
            model.train()

            latest_model = model if latest_block_timestamps[tensor_name] < version_number else latest_model
            return model, version_number

        except asyncio.TimeoutError:
            logging.error(f"Attempt {attempt}: Chunk download timed out.")
        except aiohttp.ClientError as e:
            logging.error(f"Attempt {attempt}: Failed to fetch tensor {tensor_name}: {e}")
        if attempt < retries:
            await asyncio.sleep(backoff * attempt)

    raise RuntimeError(f"initialize_tensor: Failed to initialize tensor {tensor_name} after {retries} attempts")

def get_relevant_tensor_for_task(task_type):
    return TENSOR_NAME

async def fetch_task(task_id, contract_index):
    task_tuple = await contracts[contract_index].functions.getTask(task_id).call()
    task = Task(*task_tuple)
    return task_id, task

async def initialize_contracts():
    global token_addresses, token_contracts, stake_amounts, subnet_ids, pool_contracts, max_dispute_times

    # Initialize lists to store details for each contract
    token_addresses = []
    token_contracts = []
    stake_amounts = []
    subnet_ids = []
    pool_contracts = []
    max_dispute_times = []

    # Loop through each contract to set up individual details
    for contract in contracts:
        token_address = await contract.functions.token().call()
        token_addresses.append(token_address)

        token_contract = web3.eth.contract(address=token_address, abi=load_abi('ERC20'))
        token_contracts.append(token_contract)

        stake_amount = await contract.functions.solverStakeAmount().call()
        stake_amounts.append(stake_amount)

        subnet_id = await contract.functions.subnetId().call()
        subnet_ids.append(subnet_id)

        pool_contracts.append(pool_contract)  # Assuming the pool_contract remains the same

        max_dispute_time = await contract.functions.maxDisputeTime().call()
        max_dispute_times.append(max_dispute_time)

async def get_all_task_ids(last_checked_task_ids):
    all_task_ids = []
    latest_task_ids = []
    
    for contract_index, contract in enumerate(contracts):
        latest_task_id = await contract.functions.numTasks().call() - 1
        latest_task_ids.append(latest_task_id)
        
        for task_id in range(last_checked_task_ids[contract_index] + 1, latest_task_id + 1):
            all_task_ids.append((task_id, contract_index))
    
    return all_task_ids, latest_task_ids

async def main():
    logging.info("Starting main process")
    torch.set_default_device(device)
    model_adapter.initialize_environment(args.backend)

    await initialize_contracts()

    # Approve tokens once at the start
    for private_key, token_contract in zip(args.private_keys, token_contracts):
        await approve_token_once(
            web3, token_contract, private_key,
            args.pool_address, 2**256 - 1
        )

    logging.info("Starting tensor synchronization...")
    relevant_tensors = []
    for task_type in args.task_types:
        relevant_tensors.append(get_relevant_tensor_for_task(task_type))
    relevant_tensors = list(set(relevant_tensors))
    reported = False

    # Initialize the last checked task ID and pending tasks
    last_checked_task_ids = [await contract.functions.numTasks().call() - 1 for contract in contracts]
    pending_tasks = set()
    
    asyncio.create_task(reclaim_stakes())
    
    last_loop_time = time.time()
    
    tensor_download_event.set()

    while True:
        # Schedule processing tasks
        asyncio.create_task(process_tasks())
        logging.debug(f'Loop time: {time.time() - last_loop_time:.2f} seconds')
        last_loop_time = time.time()
        await deposit_stake()
        
        if not reported:
            await report_sync_status()
            reported = True

        # Gather all task fetching coroutines
        all_task_ids, latest_task_ids = await get_all_task_ids(last_checked_task_ids)
        all_task_ids.extend([(task_id, contract_index) for task_id, contract_index in pending_tasks])
        fetch_tasks = [fetch_task(task_id, contract_index) for task_id, contract_index in all_task_ids]
        fetched_tasks = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        
        if not fetched_tasks:
            logging.debug("No tasks fetched. Sleeping...")
        else:
            logging.debug(f"Fetched {len(fetched_tasks)} tasks")

        # Clear pending tasks to update with new statuses
        pending_tasks.clear()

        # Handle events for tasks
        for idx, (task_id, task) in enumerate(fetched_tasks):
            if isinstance(task, Exception):
                logging.error(f"Error fetching task {all_task_ids[idx][0]}: {task}")
                continue
            contract_index = all_task_ids[idx][1]
            if task.solver == worker_addresses[contract_index] and task.status == TaskStatus.SolverSelected.value:
                asyncio.create_task(handle_event(task_id, task, time.time(), contract_index))
            elif task.status < TaskStatus.SolverSelected.value:
                pending_tasks.add((task_id, contract_index))  # Use tuple instead of dict
        
        # Update the last checked task ID
        last_checked_task_ids = latest_task_ids
        
        await asyncio.sleep(args.poll_interval)

if __name__ == "__main__":
    asyncio.run(main())
