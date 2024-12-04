import argparse
import json
import logging
import aiohttp
import torch
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass
from collections import defaultdict
from .device import device
from .common import (
    download_file,
    Task, TaskStatus, load_abi, upload_tensor,
    async_transact_with_contract_function,
    TENSOR_NAME, PoolState, approve_token_once,
    deposit_stake_without_approval, get_future_version_number,
    CHUNK_SIZE
)
from .models import OrderType
from io import BytesIO
import time
from tqdm import tqdm
import asyncio
import torch._dynamo
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor
import datetime
from .db.db_adapter_client import DBAdapterClient
from .plugin_manager import get_plugin
from datetime import timezone

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

subnet_in_db = None

def parse_args():
    parser = argparse.ArgumentParser(description="Worker for processing tasks based on smart contract events")
    parser.add_argument('--subnet_id', type=int, required=True, help="Subnet ID")
    parser.add_argument('--private_key', type=str, required=True, help="Private key of the worker")
    parser.add_argument('--sot_url', type=str, required=True, help="Source of Truth URL for streaming gradient updates")
    parser.add_argument('--detailed_logs', action='store_true', help="Enable detailed logging for loss task")
    parser.add_argument('--max_stakes', type=int, default=2, help="Maximum number of stakes to maintain")
    parser.add_argument('--poll_interval', type=int, default=1, help="Interval (in seconds) for polling the smart contract for new tasks")
    parser.add_argument('--torch_compile', action='store_true', help="Enable torch.compile and model warmup")
    parser.add_argument('--max_tasks_handling', type=int, default=1, help="Maximum number of tasks allowed in the queue awaiting processing")
    parser.add_argument('--db_url', type=str, required=True, help="URL of the database server")
    return parser.parse_args()

args = parse_args()

db_adapter = DBAdapterClient(args.db_url, args.private_key)

subnet = asyncio.run(db_adapter.get_subnet(args.subnet_id))

model_initialized = False
embedding_initialized = False
latest_block_timestamps = defaultdict(lambda: 0)  # To store the latest block timestamp processed for each tensor


class TaskQueue:
    def __init__(self):
        self.queue = []
        self.current_version = None
        self.lock = asyncio.Lock()  # Use asyncio.Lock
        logging.debug("Initialized TaskQueue")

    async def add_task(self, task):
        async with self.lock:  # Ensure thread-safe access to the queue
            self.queue.append(task)
            self.queue.sort(key=lambda t: t['time_solver_selected'])
            how_old = int(time.time()) - task['time_solver_selected']
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

async def get_ask_price():
    return 1

async def deposit_stake():
    global concurrent_tasks_counter
    # Do not deposit stakes if queued tasks exceed the limit
    if (await task_queue.queue_length() + concurrent_tasks_counter) > args.max_tasks_handling:
        logging.debug("Too many tasks being processed. Not depositing any more stakes.")
        return
    
    num_orders = await db_adapter.get_num_orders(args.subnet_id, OrderType.Ask.name)
    
    logging.info(f"Current number of stakes: {num_orders}")
    
    for _ in range(args.max_stakes - num_orders):
        stake_multiplier = (await db_adapter.get_subnet(args.subnet_id)).stake_multiplier
        price = await get_ask_price()
        await db_adapter.deposit_account(price * stake_multiplier)
        await db_adapter.create_order(None, args.subnet_id, OrderType.Ask.name, price)

async def handle_task(task, time_invoked):
    global last_handle_event_timestamp

    current_time = time.time()
    logging.debug(f'Time since invocation: {current_time - time_invoked:.2f} seconds')
    async with last_handle_event_timestamp_lock:
        if last_handle_event_timestamp is not None:
            time_since_last_event = current_time - last_handle_event_timestamp
            logging.debug(f"Time since last handle_event call: {time_since_last_event:.2f} seconds")
        last_handle_event_timestamp = current_time

    logging.info(f"Received event for task id {task.id}")
    logging.debug(f"Task params: {task.params}")
    task_params = json.loads(task.params)

    logging.debug(f"Adding task to queue with ID: {task.id} and params: {task_params}")
    
    job_db = await db_adapter.get_job(task.job_id)
    
    task_queue_obj = {
        'task_id': task.id,
        'plugin_id': job_db.plugin_id,
        'sot_url': job_db.sot_url,
        'task_params': task_params,
        'time_solver_selected': task.time_solver_selected.timestamp()
    }

    await task_queue.add_task(task_queue_obj)
    
    await get_plugin(task_queue_obj['plugin_id'], db_adapter)
    
    time_since_change = time.time() - task_queue_obj['time_solver_selected']
    logging.debug(f"Time since solver selection: {time_since_change} seconds")
    
    async with task_start_times_lock:
        task_start_times[task.id] = time.time()
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
                plugin = await get_plugin(next_task['plugin_id'], db_adapter)
                sot_url = next_task['sot_url']
                time_solver_selected = next_task['time_solver_selected']  # Extract the time_solver_selected

                logging.debug(f"{task_id}: Processing task with params: {task_params}")

                start_time = time.time()
                # Downloading batch and targets (asynchronously)
                logging.debug(f"{task_id}: Downloading batch from URL: {task_params['batch_url']}")
                batch = await download_file(task_params['batch_url'], download_type='batch_targets')

                logging.debug(f"{task_id}: Downloading targets from URL: {task_params['targets_url']}")
                targets = await download_file(task_params['targets_url'], download_type='batch_targets')

                try:
                    # Acquire the lock asynchronously with priority based on time_solver_selected
                    await task_processing_lock.acquire(priority=time_solver_selected)
                    try:
                        # Process the task
                        steps = task_params['steps']
                        max_lr = task_params['max_lr']
                        min_lr = task_params['min_lr']
                        T_0 = task_params['T_0']
                        weight_decay = task_params['weight_decay']
                        logging.debug(f"{task_id}: Executing training task")
                        time_synced = time.time()
                        result = await plugin.call_submodule(
                            'model_adapter', 'train_task',
                            TENSOR_NAME, sot_url, await plugin.get('tensor_version_interval'), await plugin.get('expected_worker_time'),
                            batch, targets, steps, max_lr, min_lr, T_0, weight_decay
                        )
                        logging.debug(f'{task_id}: Unpacking values from {result}')
                        version_number, updates, loss = result
                        logging.info(f"{task_id}: Updates tensor memory size: {tensor_memory_size(updates):.2f} MB")
                    finally:
                        await task_processing_lock.release()
                
                except Exception as e:
                    logging.error(f"Error during task processing: {e}")
                    raise

                try:
                    # Upload the result asynchronously with priority based on time_solver_selected
                    await upload_lock.acquire(priority=time_solver_selected)
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
                await submit_solution(task_id, result)

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

async def submit_solution(task_id, result):
    try:
        logging.info('Submitting solution')
        receipt = await db_adapter.submit_task_result(task_id, result)
        logging.info(f"solution submission receipt: {receipt}")
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

async def main():
    global subnet_in_db
    logging.info("Starting main process")

    subnet_in_db = await db_adapter.get_subnet(args.subnet_id)
    logging.info("Starting tensor synchronization...")
    reported = False
    
    last_loop_time = time.time()

    while True:
        # Schedule processing tasks
        asyncio.create_task(process_tasks())
        logging.debug(f'Loop time: {time.time() - last_loop_time:.2f} seconds')
        last_loop_time = time.time()
        await deposit_stake()
        
        processed_tasks = set()
        
        if not reported:
            await report_sync_status()
            reported = True

        assigned_tasks = await db_adapter.get_assigned_tasks(subnet_in_db.id)
        for task in assigned_tasks:
            if task.id not in processed_tasks:
                asyncio.create_task(handle_task(task, time.time()))
            processed_tasks.add(task.id)

        await asyncio.sleep(args.poll_interval)

if __name__ == "__main__":
    asyncio.run(main())
