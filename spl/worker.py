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
from device import device
from common import Task, TaskStatus, model_adapter, load_abi, upload_tensor, async_transact_with_contract_function, TENSOR_VERSION_INTERVAL, TENSOR_NAME, PoolState, approve_token_once, deposit_stake_without_approval
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from typing import Optional, Tuple
from io import BytesIO
import time
import os
from tqdm import tqdm
import asyncio
from concurrent.futures import ThreadPoolExecutor
from aiohttp import MultipartWriter
import threading
import torch._dynamo

class SuppressTracebackFilter(logging.Filter):
    def filter(self, record):
        if 'ConnectionRefusedError' in record.getMessage() or 'MaxRetryError' in record.getMessage():
            return False
        return True

# Set up logging with timestamps
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname=s) - %(message)s'
)
logger = logging.getLogger()
logger.addFilter(SuppressTracebackFilter())

# Global counter for concurrent tasks
concurrent_tasks_counter = 0
concurrent_tasks_counter_lock = threading.Lock()  # Lock for concurrent_tasks_counter

# Global dictionary to store start times for task IDs
task_start_times = {}
task_start_times_lock = threading.Lock()  # Lock for task_start_times

# Global variable to store the last handle_event timestamp
last_handle_event_timestamp = None
last_handle_event_timestamp_lock = threading.Lock()  # Lock for last_handle_event_timestamp

# Lock to ensure only one task is processed at a time
task_processing_lock = threading.Lock()

def parse_args():
    parser = argparse.ArgumentParser(description="Worker for processing tasks based on smart contract events")
    parser.add_argument('--task_types', type=str, required=True, help="Types of tasks to process, separated by '+' if multiple")
    parser.add_argument('--subnet_addresses', type=str, required=True, help="Subnet contract addresses")
    parser.add_argument('--private_keys', type=str, required=True, help="Private keys of the worker's Ethereum accounts")
    parser.add_argument('--rpc_url', type=str, default='http://localhost:8545', help="URL of the Ethereum RPC node")
    parser.add_argument('--sot_url', type=str, required=True, help="Source of Truth URL for streaming gradient updates")
    parser.add_argument('--pool_address', type=str, required=True, help="Pool contract address")
    parser.add_argument('--group', type=int, required=True, help="Group for depositing stake")
    parser.add_argument('--local_storage_dir', type=str, default='data', help="Directory for local storage of files")
    parser.add_argument('--backend', type=str, default='nccl', help="Distributed backend to use (default: nccl, use 'gloo' for macOS)")
    parser.add_argument('--layer_idx', type=int, help="Layer index for forward and backward tasks", required=False)
    parser.add_argument('--sync_url', type=str, required=False, help="URL for reporting sync status", default='http://localhost:5002')
    parser.add_argument('--detailed_logs', action='store_true', help="Enable detailed logging for loss task")
    parser.add_argument('--max_stakes', type=int, default=4, help="Maximum number of stakes to maintain")
    parser.add_argument('--poll_interval', type=int, default=1, help="Interval (in seconds) for polling the smart contract for new tasks")
    parser.add_argument('--torch_compile', action='store_true', help="Enable torch.compile and model warmup")
    parser.add_argument('--max_queued_tasks', type=int, default=4, help="Maximum number of tasks allowed in the queue awaiting processing")
    return parser.parse_args()

args = parse_args()

# Split combined task types if any
task_types = args.task_types.split('+')
args.task_types = task_types

subnet_addresses = args.subnet_addresses.split('+')
args.subnet_addresses = subnet_addresses

private_keys = args.private_keys.split('+')
args.private_keys = private_keys

os.makedirs(args.local_storage_dir, exist_ok=True)

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

last_model = None

class TaskQueue:
    def __init__(self):
        self.queue = []
        self.current_version = None
        self.lock = threading.Lock()  # Add a lock for the queue
        logging.debug("Initialized TaskQueue")

    def add_task(self, task):
        with self.lock:  # Ensure thread-safe access to the queue
            self.queue.append(task)
            self.queue.sort(key=lambda t: t['time_status_changed'])
            logging.debug(f"Added task: {task}. Queue size is now {len(self.queue)}")

    def get_next_task(self):
        with self.lock:  # Ensure thread-safe access to the queue
            if self.queue:
                task = self.queue.pop(0)
                logging.debug(f"Retrieved task: {task}. Queue size is now {len(self.queue)}")
                return task
            logging.debug("No tasks in the queue.")
            return None

    def queue_length(self):
        with self.lock:
            return len(self.queue)

task_queue = TaskQueue()

async def download_file(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content = await response.read()
            return torch.load(BytesIO(content))

async def download_json(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()  # Raise an error for bad status codes
                data = await response.json()  # Parse and return the JSON content
                return torch.tensor(data, dtype=torch.long).to(device)  # Convert to tensor
    except aiohttp.ClientError as e:
        logging.error(f"Failed to download JSON from {url}: {e}")
        raise

async def get_json(url, params=None):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                response.raise_for_status()  # Raise an error for bad status codes
                return await response.json()  # Parse and return the JSON content
    except aiohttp.ClientError as e:
        logging.error(f"Failed to download JSON from {url}: {e}")
        raise

def create_callback(encoder, pbar):
    def callback(monitor):
        pbar.update(monitor.bytes_read - pbar.n)
    return callback

async def upload_tensor(tensor, tensor_name):
    # Convert the tensor to bytes
    tensor_bytes = BytesIO()
    torch.save(tensor, tensor_bytes)
    tensor_bytes.seek(0)

    # Get the total size of the tensor in bytes for the progress bar
    total_size = tensor_bytes.getbuffer().nbytes

    # Initialize the progress bar
    pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc='Uploading')

    async with aiohttp.ClientSession() as session:
        # Create a MultipartWriter for sending the tensor file and label
        with aiohttp.MultipartWriter() as mpwriter:
            # Add the tensor file part
            part = mpwriter.append(tensor_bytes, {'Content-Type': 'application/octet-stream'})
            part.set_content_disposition('form-data', name='tensor', filename=tensor_name)

            # Add the label part
            mpwriter.append_form({'label': tensor_name})

            headers = {'Content-Type': mpwriter.headers['Content-Type']}
            logging.debug("Starting tensor upload...")

            try:
                # Send the POST request
                async with session.post(f'{args.sot_url}/upload_tensor', data=mpwriter, headers=headers, timeout=300) as response:
                    # Read the response and update the progress bar
                    pbar.update(total_size)

                    # Check the response status
                    if response.status == 200:
                        response_json = await response.json()
                        tensor_url = response_json.get('tensor_url')
                        logging.debug("Upload completed successfully.")
                        return tensor_url
                    else:
                        logging.error(f"Failed to upload tensor: {response.status} {await response.text()}")
                        raise RuntimeError(f"Failed to upload tensor: {response.status} {await response.text()}")

            except asyncio.TimeoutError:
                logging.error("Upload request timed out.")
                raise RuntimeError("Failed to upload tensor: request timed out")
            finally:
                # Close the progress bar
                pbar.close()

async def deposit_stake():
    # Do not deposit stakes if queued tasks exceed the limit
    if task_queue.queue_length() > args.max_queued_tasks:
        logging.debug("Too many tasks in the queue. Not depositing any more stakes.")
        return

    wallets = zip(args.private_keys, subnet_ids, stake_amounts, token_contracts, pool_contracts, worker_addresses)
    
    for private_key, subnet_id, stake_amount, token_contract, pool_contract, worker_address in wallets:
        await deposit_stake_without_approval(web3, pool_contract, private_key, subnet_id, args.group, worker_address, stake_amount, args.max_stakes)

def handle_event(task_id, task, time_invoked, contract_index):
    global last_handle_event_timestamp

    current_time = time.time()
    logging.debug(f'Time since invocation: {current_time - time_invoked:.2f} seconds')
    with last_handle_event_timestamp_lock:
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

    task_queue.add_task({
        'task_id': task_id,
        'task_params': task_params,
        'time_status_changed': task.timeStatusChanged,
        'contract_index': contract_index
    })
    
    blockchain_timestamp = asyncio.run(web3.eth.get_block('latest'))['timestamp']
    
    time_since_change = blockchain_timestamp - task.timeStatusChanged
    logging.debug(f"Time since status change: {time_since_change} seconds")
    
    with task_start_times_lock:
        task_start_times[task_id] = time.time()
    asyncio.run(process_tasks())

def tensor_memory_size(tensor):
    # Calculate size in bytes and convert to megabytes
    size_in_bytes = tensor.element_size() * tensor.numel()
    size_in_mb = size_in_bytes / (1024 * 1024)
    return size_in_mb

async def process_tasks():
    global task_queue, concurrent_tasks_counter
    try:
        # Ensure only one task is processed at a time
        start_time = time.time()

        with concurrent_tasks_counter_lock:
            # Increase the counter when a new task is started
            concurrent_tasks_counter += 1
        logging.debug(f"process_tasks() started. Concurrent tasks: {concurrent_tasks_counter}")

        next_task = task_queue.get_next_task()
        if not next_task:
            logging.debug("No tasks in the queue to process.")
            with concurrent_tasks_counter_lock:
                concurrent_tasks_counter -= 1
            return
        task_id = next_task['task_id']
        task_params = next_task['task_params']
        contract_index = next_task['contract_index']
        task_type = args.task_types[contract_index]
        tensor_name = get_relevant_tensor_for_task(task_type)
        version_num_url = f'{args.sot_url}/tensor_block_timestamp'
        version_number = await get_json(
            version_num_url,
            params={'tensor_name': tensor_name}
        )['version_number']
        if (task_queue.current_version is None
            or version_number != task_queue.current_version):

            logging.debug(f"Syncing tensors for version number: {version_number}")
            sync_start_time = time.time()
            model = await sync_tensors(version_number, next_task['contract_index'])
            sync_end_time = time.time()
            logging.debug(f"Sync tensors took {sync_end_time - sync_start_time:.2f} seconds")
            task_queue.current_version = version_number

        logging.debug(f"Processing task with ID: {task_id}, params: {task_params}, and contract_index: {contract_index}")

        # Process the task...
        logging.debug(f"Downloading batch from URL: {task_params['batch_url']}")
        download_start_time = time.time()
        batch = await download_file(task_params['batch_url'])
        download_end_time = time.time()
        logging.debug(f"Downloading batch took {download_end_time - download_start_time:.2f} seconds")


        logging.debug(f"Downloading targets from URL: {task_params['targets_url']}")
        download_start_time = time.time()
        targets = await download_file(task_params['targets_url'])
        download_end_time = time.time()
        logging.debug(f"Downloading targets took {download_end_time - download_start_time:.2f} seconds")

        result = {}
        accumulation_steps = task_params['accumulation_steps']
        
        with task_processing_lock:
            logging.debug("Executing training task")
            task_start_time = time.time()
            updates, loss = model_adapter.train_task(model, batch, targets, accumulation_steps)
            task_end_time = time.time()
            logging.debug(f"Task took {task_end_time - task_start_time:.2f} seconds")
        logging.info(f"Updates tensor memory size: {tensor_memory_size(updates):.2f} MB")
        result = await upload_results(version_number, updates, loss)
        del updates
        logging.info(f"Uploaded results for task {task_id}")

        submit_solution_start_time = time.time()
        await submit_solution(task_id, result, contract_index)
        submit_solution_end_time = time.time()
        logging.debug(f"submit_solution() took {submit_solution_end_time - submit_solution_start_time:.2f} seconds")

        processed_tasks.add((task_id, contract_index))

        logging.info(f"Processed task {task_id} for task type {task_type} successfully")

        # Log the time taken to process the task
        with task_start_times_lock:
            task_start_time = task_start_times.pop(task_id, None)
        if task_start_time:
            total_time = time.time() - task_start_time
            logging.info(f"Total time to process task {task_id}: {total_time:.2f} seconds")

        end_time = time.time()
        logging.info(f"process_tasks() completed in {end_time - start_time:.2f} seconds. Concurrent tasks: {concurrent_tasks_counter}")

        with concurrent_tasks_counter_lock:
            # Decrease the counter when a task is completed
            concurrent_tasks_counter -= 1
    except Exception as e:
        logging.error(f"Error processing task: {e}", exc_info=True)
        with concurrent_tasks_counter_lock:
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
                    receipt = await async_transact_with_contract_function(web3, contract, 'resolveTask', args.private_keys[contract_index], task_id)
                    logging.info(f"resolveTask transaction receipt: {receipt}")
                    processed_tasks.remove((task_id, contract_index))
                except Exception as e:
                    logging.error(f"Error resolving task {task_id}: {e}")
                    raise

        await asyncio.sleep(2)

async def sync_tensors(sync_version_number, contract_index):
    relevant_tensor = get_relevant_tensor_for_task(args.task_types[contract_index])
    return await initialize_tensor(relevant_tensor, sync_version_number)

async def submit_solution(task_id, result, contract_index):
    try:
        receipt = await async_transact_with_contract_function(web3, contracts[contract_index], 'submitSolution', args.private_keys[contract_index], task_id, json.dumps(result).encode('utf-8'))
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
    try:
        url = f"{args.sync_url}/report_sync"
        async with aiohttp.ClientSession() as session:
            async with session.post(url) as response:
                await response.text()
    except aiohttp.ClientError as e:
        logging.error(f"Exception while reporting sync status: {e}")

async def initialize_tensor(tensor_name, sync_version_number=None):
    global latest_model
    if latest_block_timestamps[tensor_name] == sync_version_number:
        return latest_model
    try:
        url = f"{args.sot_url}/latest_state"
        logging.info(f"Loading tensor {tensor_name} from {url}")
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params={'tensor_name': tensor_name, 'version_number': sync_version_number}) as response:
                response.raise_for_status()  # Raise an error for bad status codes

                tensor = torch.load(BytesIO(await response.read()))
                logging.debug(f"Loaded tensor {tensor_name} with shape {tensor.shape}")

                logging.info(f"Successfully initialized tensor: {tensor_name}")

                first_initialization = latest_model is None

                model = model_adapter.tensor_to_model(tensor.detach(), latest_model)
                model.train()
                if args.torch_compile and first_initialization:
                    # Compile the model after loading state_dict
                    model = model_adapter.compile_model(model)
                    logging.info("Model model compiled and warmed up")
                if latest_block_timestamps[tensor_name] == 0 or sync_version_number is None or latest_block_timestamps[tensor_name] < sync_version_number:
                    latest_model = model
                    latest_block_timestamps[tensor_name] = sync_version_number
                return model

    except aiohttp.ClientError as e:
        logging.error(f"Failed to initialize tensor {tensor_name} due to request exception: {e}")
        raise
    except Exception as e:
        logging.error(f"Failed to initialize tensor {tensor_name} due to error: {e}")
        raise

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
        await approve_token_once(web3, token_contract, private_key, args.pool_address, 2**256 - 1)

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

    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=10) as executor:
        while True:
            logging.debug(f'Loop time: {time.time() - last_loop_time:.2f} seconds')
            last_loop_time = time.time()
            await deposit_stake()
            
            if not reported:
                duplicate_relevant_tensors = []
                for contract_index, task_type in enumerate(args.task_types):
                    relevant_tensor = get_relevant_tensor_for_task(task_type)
                    duplicate_relevant_tensors.append((contract_index, relevant_tensor))
                for contract_index, tensor_name in duplicate_relevant_tensors:
                    await initialize_tensor(tensor_name)
                await report_sync_status()
                reported = True

            # Gather all task fetching coroutines
            all_task_ids, latest_task_ids = await get_all_task_ids(last_checked_task_ids)
            all_task_ids.extend([(task_id, contract_index) for task_id, contract_index in pending_tasks])
            fetch_tasks = [fetch_task(task_id, contract_index) for task_id, contract_index in all_task_ids]
            fetched_tasks = await asyncio.gather(*fetch_tasks)

            # Clear pending tasks to update with new statuses
            pending_tasks.clear()

            # Handle events for tasks
            for task_id, task in fetched_tasks:
                contract_index = all_task_ids.pop(0)[1]
                if task.solver == worker_addresses[contract_index] and task.status == TaskStatus.SolverSelected.value:
                    executor.submit(handle_event, task_id, task, time.time(), contract_index)
                elif task.status < TaskStatus.SolverSelected.value:
                    pending_tasks.add((task_id, contract_index))  # Use tuple instead of dict
            
            # Update the last checked task ID
            last_checked_task_ids = latest_task_ids
            
            await asyncio.sleep(args.poll_interval)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(main())
