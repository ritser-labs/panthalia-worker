import argparse
import json
import logging
import requests
import torch
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass
from web3 import AsyncWeb3
from web3.exceptions import ContractCustomError
from web3.middleware import async_geth_poa_middleware
from collections import defaultdict
from device import device
from common import get_dummy_input, Model, Task, TaskStatus, model_args, tokenizer, initialize_distributed_environment, load_abi, upload_tensor, tensor_to_model, initialize_distributed_environment_and_globals, async_transact_with_contract_function, TENSOR_VERSION_INTERVAL, TENSOR_NAME, PoolState, approve_token_once, deposit_stake_without_approval
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from typing import Optional, Tuple
from io import BytesIO
import time
import os
from tqdm import tqdm
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
tensors = defaultdict(lambda: None)
latest_block_timestamps = defaultdict(lambda: 0)  # To store the latest block timestamp processed for each tensor
processed_tasks = set()

model = None

tensors_lock = threading.Lock()  # Lock for tensors

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

def download_file(url):
    response = requests.get(url)
    return torch.load(BytesIO(response.content))

def download_json(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()  # Parse and return the JSON content
        return torch.tensor(data, dtype=torch.long).to(device)  # Convert to tensor
    except requests.RequestException as e:
        logging.error(f"Failed to download JSON from {url}: {e}")
        raise

def get_json(url, params=None):
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()  # Parse and return the JSON content
    except requests.RequestException as e:
        logging.error(f"Failed to download JSON from {url}: {e}")
        raise

def create_callback(encoder, pbar):
    def callback(monitor):
        pbar.update(monitor.bytes_read - pbar.n)
    return callback

async def upload_tensor(tensor, tensor_name):
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
    logging.debug("Starting tensor upload...")

    loop = asyncio.get_event_loop()

    try:
        response = await loop.run_in_executor(None, lambda: requests.post(f'{args.sot_url}/upload_tensor', data=monitor, headers=headers, timeout=300))
        pbar.close()
        logging.debug("Upload completed.")
    except requests.exceptions.Timeout:
        logging.error("Upload request timed out.")
        pbar.close()
        raise RuntimeError("Failed to upload tensor: request timed out")

    if response.status_code == 200:
        return response.json().get('tensor_url')
    else:
        raise RuntimeError(f"Failed to upload tensor: {response.text}")

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
        with task_processing_lock:
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
            version_number = get_json(
                version_num_url,
                params={'tensor_name': tensor_name}
            )['version_number']
            if (task_queue.current_version is None
                or version_number != task_queue.current_version):
                logging.debug(f"Syncing tensors for version number: {version_number}")
                sync_start_time = time.time()
                await sync_tensors(version_number, next_task['contract_index'])
                sync_end_time = time.time()
                logging.debug(f"Sync tensors took {sync_end_time - sync_start_time:.2f} seconds")
                task_queue.current_version = version_number

            logging.debug(f"Processing task with ID: {task_id}, params: {task_params}, and contract_index: {contract_index}")

            # Process the task...
            logging.debug(f"Downloading batch from URL: {task_params['batch_url']}")
            download_start_time = time.time()
            batch = download_json(task_params['batch_url'])
            download_end_time = time.time()
            logging.debug(f"Downloading batch took {download_end_time - download_start_time:.2f} seconds")
            logging.info(f"Batch tensor memory size: {tensor_memory_size(batch):.2f} MB")


            logging.debug(f"Downloading targets from URL: {task_params['targets_url']}")
            download_start_time = time.time()
            targets = download_json(task_params['targets_url'])
            download_end_time = time.time()
            logging.debug(f"Downloading targets took {download_end_time - download_start_time:.2f} seconds")
            logging.info(f"Targets tensor memory size: {tensor_memory_size(targets):.2f} MB")

            result = {}
            accumulation_steps = task_params['accumulation_steps']
            
            logging.debug("Executing training task")
            task_start_time = time.time()
            updates, loss = model_task(batch, targets, accumulation_steps)
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
    await initialize_tensor(relevant_tensor, sync_version_number)

async def submit_solution(task_id, result, contract_index):
    try:
        receipt = await async_transact_with_contract_function(web3, contracts[contract_index], 'submitSolution', args.private_keys[contract_index], task_id, json.dumps(result).encode('utf-8'))
        logging.info(f"submitSolution transaction receipt: {receipt}")
    except Exception as e:
        logging.error(f"Error submitting solution for task {task_id}: {e}")
        raise

def model_task(inputs, targets, accumulation_steps):
    global model, tensors
    logging.info("Starting model_task")

    start_time = time.time()
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    logging.debug(f"Moved inputs and targets to device. Time taken: {time.time() - start_time:.2f} seconds")

    microbatch_size = inputs.shape[0] // accumulation_steps

    # Preallocate gradient accumulation tensors and zero them
    grads_accumulated = [torch.zeros_like(param, device=device) for param in model.parameters()]

    total_loss = 0.0

    logging.info(f"Accumulation steps: {accumulation_steps}, Microbatch size: {microbatch_size}")

    for i in range(accumulation_steps):
        batch_start_time = time.time()
        try:
            microbatch_inputs = inputs[i * microbatch_size:(i + 1) * microbatch_size].detach()
            microbatch_targets = targets[i * microbatch_size:(i + 1) * microbatch_size].detach()

            start_pos = 0
            # Forward pass
            output = model(microbatch_inputs, start_pos=start_pos)

            reshaped_logits = output.view(-1, model_args.vocab_size)
            reshaped_targets = microbatch_targets.view(-1)

            # Compute loss
            loss = F.cross_entropy(reshaped_logits, reshaped_targets, ignore_index=tokenizer.pad_id)
            total_loss += loss.item()

            logging.debug(f"Microbatch {i + 1}/{accumulation_steps}: Forward pass completed. Time taken: {time.time() - batch_start_time:.2f} seconds")

            # Backward pass and accumulate gradients
            loss.backward()

            with torch.no_grad():
                for j, param in enumerate(model.parameters()):
                    if param.grad is not None:
                        grads_accumulated[j] += param.grad

            # Clear gradients for next accumulation step
            model.zero_grad()

            # Detach cache tensors
            if hasattr(model, 'layers'):
                for layer in model.layers:
                    if hasattr(layer, 'attention'):
                        layer.attention.cache_k = layer.attention.cache_k.detach()
                        layer.attention.cache_v = layer.attention.cache_v.detach()

            # Delete intermediate variables
            del output, reshaped_logits, reshaped_targets, loss, microbatch_inputs, microbatch_targets
            torch.cuda.empty_cache()

            logging.debug(f"Microbatch {i + 1}/{accumulation_steps}: Backward pass completed. Time taken: {time.time() - batch_start_time:.2f} seconds")

        except Exception as e:
            logging.error(f"Error processing microbatch {i + 1}/{accumulation_steps}: {e}", exc_info=True)
            raise  # Re-raise the exception after logging

    # Normalize accumulated gradients
    with torch.no_grad():
        for grad in grads_accumulated:
            grad.div_(accumulation_steps)

    updates = torch.cat([grad.view(-1) for grad in grads_accumulated])
    loss = total_loss / accumulation_steps

    logging.info(f"Model task completed. Total loss: {loss:.4f}. Total time taken: {time.time() - start_time:.2f} seconds")

    return updates, loss

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
        response = requests.post(url)
    except requests.RequestException as e:
        logging.error(f"Exception while reporting sync status: {e}")

async def initialize_tensor(tensor_name, sync_version_number=None):
    global model

    try:
        url = f"{args.sot_url}/latest_state"
        logging.info(f"Loading tensor {tensor_name} from {url}")
        response = requests.get(url, params={'tensor_name': tensor_name, 'version_number': sync_version_number})
        response.raise_for_status()  # Raise an error for bad status codes

        tensor = torch.load(BytesIO(response.content))
        logging.debug(f"Loaded tensor {tensor_name} with shape {tensor.shape}")

        tensors[tensor_name] = tensor

        latest_block_timestamps[tensor_name] = sync_version_number
        logging.info(f"Successfully initialized tensor: {tensor_name}")

        model = tensor_to_model(tensor.detach())
        model.train()
        if args.torch_compile:
            # Compile the model after loading state_dict
            model = torch.compile(model)
            # Warmup
            _ = model(get_dummy_input())
            logging.info("Model model compiled and warmed up")

    except requests.exceptions.RequestException as e:
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
    initialize_distributed_environment_and_globals(args.backend)

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
