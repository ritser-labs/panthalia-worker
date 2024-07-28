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
from model import TransformerBlock, VocabParallelEmbedding, ColumnParallelLinear, precompute_freqs_cis, RMSNorm
from device import device
from common import Task, TaskStatus, model_args, tokenizer, initialize_distributed_environment, load_abi, upload_tensor, download_file, async_transact_with_contract_function, TENSOR_VERSION_INTERVAL, wait_for_state_change, PoolState, approve_token_once, deposit_stake_without_approval
from fairscale.nn.model_parallel.initialize import initialize_model_parallel, model_parallel_is_initialized
from typing import Optional, Tuple
from io import BytesIO
import time
import os
from tqdm import tqdm
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor

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

# Global dictionary to store start times for task IDs
task_start_times = {}

# Global variable to store the last handle_event timestamp
last_handle_event_timestamp = None

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    max_batch_size: int = 32
    max_seq_len: int = 2048

args = ModelArgs()

def parse_args():
    parser = argparse.ArgumentParser(description="Worker for processing tasks based on smart contract events")
    parser.add_argument('--task_type', type=str, required=True, choices=[
        'embed', 'forward', 'backward', 'final_logits', 'embed_backward'
    ], help="Type of task to process")
    parser.add_argument('--subnet_address', type=str, required=True, help="Subnet contract address")
    parser.add_argument('--private_key', type=str, required=True, help="Private key of the worker's Ethereum account")
    parser.add_argument('--rpc_url', type=str, default='http://localhost:8545', help="URL of the Ethereum RPC node")
    parser.add_argument('--sot_url', type=str, required=True, help="Source of Truth URL for streaming gradient updates")
    parser.add_argument('--pool_address', type=str, required=True, help="Pool contract address")
    parser.add_argument('--group', type=int, required=True, help="Group for depositing stake")
    parser.add_argument('--local_storage_dir', type=str, default='data', help="Directory for local storage of files")
    parser.add_argument('--backend', type=str, default='nccl', help="Distributed backend to use (default: nccl, use 'gloo' for macOS)")
    parser.add_argument('--layer_idx', type=int, help="Layer index for forward and backward tasks", required=False)
    parser.add_argument('--sync_url', type=str, required=False, help="URL for reporting sync status", default='http://localhost:5002')
    parser.add_argument('--detailed_logs', action='store_true', help="Enable detailed logging for loss task")
    parser.add_argument('--max_stakes', type=int, default=8, help="Maximum number of stakes to maintain")
    parser.add_argument('--poll_interval', type=int, default=1, help="Interval (in seconds) for polling the smart contract for new tasks")
    return parser.parse_args()

args = parse_args()

os.makedirs(args.local_storage_dir, exist_ok=True)

web3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(args.rpc_url))
web3.middleware_onion.inject(async_geth_poa_middleware, layer=0)

worker_account = web3.eth.account.from_key(args.private_key)
worker_address = worker_account.address

subnet_manager_abi = load_abi('SubnetManager')
pool_abi = load_abi('Pool')

contract_address = args.subnet_address
contract = web3.eth.contract(address=contract_address, abi=subnet_manager_abi)
pool_contract = web3.eth.contract(address=args.pool_address, abi=pool_abi)

model_initialized = False
embedding_initialized = False
tensors = defaultdict(lambda: None)
adam_m = defaultdict(lambda: None)
adam_v = defaultdict(lambda: None)
latest_block_timestamps = defaultdict(lambda: 0)  # To store the latest block timestamp processed for each tensor
gradient_update_paused = False
processed_tasks = set()

freqs_cis = None
mask = None
embedding = None  # Define the global embedding variable
final_logits_layer = None
final_logits_norm = None

# Global variable for TransformerBlock layer
transformer_layer = None

class TaskQueue:
    def __init__(self):
        self.queue = []
        self.current_version = None
        logging.debug("Initialized TaskQueue")

    def add_task(self, task):
        self.queue.append(task)
        self.queue.sort(key=lambda t: t['time_status_changed'])
        logging.debug(f"Added task: {task}. Queue size is now {len(self.queue)}")

    def get_next_task(self):
        if self.queue:
            task = self.queue.pop(0)
            logging.debug(f"Retrieved task: {task}. Queue size is now {len(self.queue)}")
            return task
        logging.debug("No tasks in the queue.")
        return None

task_queue = TaskQueue()

def block_to_tensor(block: TransformerBlock) -> torch.Tensor:
    params = list(block.parameters())
    return torch.cat([p.view(-1) for p in params])

def tensor_to_block(tensor: torch.Tensor, layer_idx: int) -> TransformerBlock:
    block = TransformerBlock(layer_idx, model_args).to(device)
    pointer = 0
    total_params = sum(p.numel() for p in block.parameters())

    if tensor.numel() != total_params:
        raise ValueError(f"Total number of parameters {total_params} does not match the size of the tensor {tensor.numel()}")

    for param in block.parameters():
        num_param = param.numel()
        logging.debug(f"Pointer: {pointer}, Num param: {num_param}, Tensor size: {tensor.numel()}")

        if pointer + num_param > tensor.numel():
            raise ValueError(f"Pointer {pointer} with num_param {num_param} exceeds tensor size {tensor.numel()}")

        param.data = tensor[pointer:pointer + num_param].view(param.size()).to(device)
        pointer += num_param

    return block

def tensor_to_final_logits(tensor: torch.Tensor) -> Tuple[RMSNorm, ColumnParallelLinear]:
    global final_logits_norm, final_logits_layer
    
    dim = model_args.dim
    vocab_size = model_args.vocab_size

    # Initialize RMSNorm and ColumnParallelLinear layers
    final_logits_norm = RMSNorm(dim, eps=model_args.norm_eps).to(device)
    final_logits_layer = ColumnParallelLinear(dim, vocab_size, bias=False).to(device)

    pointer = 0

    # Load RMSNorm weights
    norm_weight_numel = final_logits_norm.weight.numel()
    final_logits_norm.weight.data = tensor[pointer:pointer + norm_weight_numel].view(final_logits_norm.weight.size()).to(device)
    pointer += norm_weight_numel

    # Load ColumnParallelLinear weights
    linear_weight_numel = final_logits_layer.weight.numel()
    final_logits_layer.weight.data = tensor[pointer:pointer + linear_weight_numel].view(final_logits_layer.weight.size()).to(device)
    pointer += linear_weight_numel

    return final_logits_norm, final_logits_layer

def final_logits_to_tensor() -> torch.Tensor:
    norm_weight = final_logits_norm.weight.data.view(-1)
    linear_weight = final_logits_layer.weight.data.view(-1)
    return torch.cat((norm_weight, linear_weight))

def initialize_distributed_environment_and_globals():
    global freqs_cis, mask

    logging.info("Initializing distributed environment")
    initialize_distributed_environment(args.backend)
    initialize_model_parallel(model_parallel_size_=1)

    freqs_cis = precompute_freqs_cis(
        model_args.dim // model_args.n_heads,
        model_args.max_seq_len * 2,
        model_args.rope_theta,
    )

    mask = torch.triu(torch.full((model_args.max_seq_len, model_args.max_seq_len), float('-inf')), diagonal=1).to(device)
    logging.info("Environment and global variables initialized")

def initialize_relevant_tensors(task_type, layer_idx=None):
    logging.info(f"Initializing tensors relevant to the task_type: {task_type}")

    relevant_tensors = get_relevant_tensors_for_task(task_type)
    
    for tensor_name in relevant_tensors:
        initialize_tensor(tensor_name)

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

def pause_gradient_updates():
    global gradient_update_paused
    gradient_update_paused = True

def resume_gradient_updates():
    global gradient_update_paused
    gradient_update_paused = False

async def deposit_stake():
    await deposit_stake_without_approval(web3, pool_contract, args.private_key, subnet_id, args.group, worker_address, stake_amount, args.max_stakes)

def handle_event(task_id, task, time_invoked):
    global last_handle_event_timestamp

    current_time = time.time()
    logging.debug(f'Time since invocation: {current_time - time_invoked:.2f} seconds')
    if last_handle_event_timestamp is not None:
        time_since_last_event = current_time - last_handle_event_timestamp
        logging.debug(f"Time since last handle_event call: {time_since_last_event:.2f} seconds")
    last_handle_event_timestamp = current_time

    solver = task.solver

    logging.info(f"Received event for task {args.task_type} and id {task_id} and layer {args.layer_idx}")

    if solver.lower() != worker_address.lower():
        logging.debug("Solver address does not match worker address. Ignoring event.")
        return

    task_params_bytes = task.params
    task_params = json.loads(task_params_bytes.decode('utf-8'))

    logging.debug(f"Adding task to queue with ID: {task_id} and params: {task_params}")

    task_queue.add_task({
        'task_id': task_id,
        'task_params': task_params,
        'version_number': task_params.get('version_number'),
        'time_status_changed': task.timeStatusChanged
    })
    
    blockchain_timestamp = asyncio.run(web3.eth.get_block('latest'))['timestamp']
    
    time_since_change = blockchain_timestamp - task.timeStatusChanged
    logging.debug(f"Time since status change: {time_since_change} seconds")
    

    task_start_times[task_id] = time.time()
    asyncio.run(process_tasks())

async def process_tasks():
    global task_queue, concurrent_tasks_counter

    start_time = time.time()

    # Increase the counter when a new task is started
    concurrent_tasks_counter += 1
    logging.debug(f"process_tasks() started. Concurrent tasks: {concurrent_tasks_counter}")

    next_task = task_queue.get_next_task()
    if not next_task:
        logging.debug("No tasks in the queue to process.")
        return

    if (task_queue.current_version is None
        or next_task['version_number'] != task_queue.current_version):
        logging.debug(f"Syncing tensors for version number: {next_task['version_number']}")
        sync_start_time = time.time()
        await sync_tensors(next_task['version_number'])
        sync_end_time = time.time()
        logging.debug(f"Sync tensors took {sync_end_time - sync_start_time:.2f} seconds")
        task_queue.current_version = next_task['version_number']

    task_id = next_task['task_id']
    task_params = next_task['task_params']
    
    logging.debug(f"Processing task with ID: {task_id} and params: {task_params}")

    # Process the task...
    batch = None
    if 'batch_url' in task_params:
        logging.debug(f"Downloading batch from URL: {task_params['batch_url']}")
        download_start_time = time.time()
        batch = download_json(task_params['batch_url'])
        download_end_time = time.time()
        logging.debug(f"Downloading batch took {download_end_time - download_start_time:.2f} seconds")

    inputs = None
    if 'inputs_url' in task_params:
        logging.debug(f"Downloading inputs from URL: {task_params['inputs_url']}")
        download_start_time = time.time()
        inputs = download_file(task_params['inputs_url'])
        download_end_time = time.time()
        logging.debug(f"Downloading inputs took {download_end_time - download_start_time:.2f} seconds")

    error = None
    if 'error_url' in task_params:
        logging.debug(f"Downloading error from URL: {task_params['error_url']}")
        download_start_time = time.time()
        error = download_file(task_params['error_url'])
        download_end_time = time.time()
        logging.debug(f"Downloading error took {download_end_time - download_start_time:.2f} seconds")

    targets = None
    if 'targets_url' in task_params:
        logging.debug(f"Downloading targets from URL: {task_params['targets_url']}")
        download_start_time = time.time()
        targets = download_json(task_params['targets_url'])
        download_end_time = time.time()
        logging.debug(f"Downloading targets took {download_end_time - download_start_time:.2f} seconds")

    pause_gradient_updates()

    task_type = args.task_type
    layer_idx = args.layer_idx
    result = {}
    accumulation_steps = task_params.get('accumulation_steps', 1)
    
    if task_type == 'embed':
        logging.debug("Executing embed task")
        embed_start_time = time.time()
        embed_task(batch)
        embed_end_time = time.time()
        logging.debug(f"embed_task() took {embed_end_time - embed_start_time:.2f} seconds")
        result['result_url'] = await upload_tensor(tensors['outputs'], 'embed_outputs')
    elif task_type == 'forward':
        logging.debug(f"Executing forward task for layer {layer_idx}")
        forward_start_time = time.time()
        forward_task(layer_idx, inputs)
        forward_end_time = time.time()
        logging.debug(f"forward_task() took {forward_end_time - forward_start_time:.2f} seconds")
        result['result_url'] = await upload_tensor(tensors['outputs'], f'layer_{layer_idx}_outputs')
    elif task_type == 'backward':
        logging.debug(f"Executing backward task for layer {layer_idx}")
        backward_start_time = time.time()
        backward_task(layer_idx, error, inputs, accumulation_steps)
        backward_end_time = time.time()
        logging.debug(f"backward_task() took {backward_end_time - backward_start_time:.2f} seconds")
        result = await upload_tensors_and_grads(tensors['error_output'], tensors['updates'], layer_idx)
    elif task_type == 'final_logits':
        logging.debug("Executing final_logits task")
        final_logits_start_time = time.time()
        final_logits_task(inputs, targets, accumulation_steps)
        final_logits_end_time = time.time()
        logging.debug(f"final_logits_task() took {final_logits_end_time - final_logits_start_time:.2f} seconds")
        result = await upload_final_logits_results()
    elif task_type == 'embed_backward':
        logging.debug("Executing embed_backward task")
        embed_backward_start_time = time.time()
        embed_backward_task(error, batch, accumulation_steps)
        embed_backward_end_time = time.time()
        logging.debug(f"embed_backward_task() took {embed_backward_end_time - embed_backward_start_time:.2f} seconds")
        result = await upload_tensors_and_grads(None, tensors['updates'], -2)
    
    submit_solution_start_time = time.time()
    await submit_solution(task_id, result)
    submit_solution_end_time = time.time()
    logging.debug(f"submit_solution() took {submit_solution_end_time - submit_solution_start_time:.2f} seconds")

    processed_tasks.add(task_id)

    logging.info(f"Processed task {task_id} successfully")

    # Log the time taken to process the task
    task_start_time = task_start_times.pop(task_id, None)
    if task_start_time:
        total_time = time.time() - task_start_time
        logging.info(f"Total time to process task {task_id}: {total_time:.2f} seconds")

    resume_gradient_updates()

    end_time = time.time()
    logging.info(f"process_tasks() completed in {end_time - start_time:.2f} seconds. Concurrent tasks: {concurrent_tasks_counter}")

    # Decrease the counter when a task is completed
    concurrent_tasks_counter -= 1

async def reclaim_stakes():
    max_dispute_time = await contract.functions.maxDisputeTime().call()
    while True:
        for task_id in list(processed_tasks):
            task_tuple = await contract.functions.getTask(task_id).call()
            task = Task(*task_tuple)
            task_status = task.status
            time_status_changed = task.timeStatusChanged
            blockchain_timestamp = (await web3.eth.get_block('latest'))['timestamp']
            time_elapsed = blockchain_timestamp - time_status_changed
            
            if task_status == TaskStatus.SolutionSubmitted.value and time_elapsed >= max_dispute_time:
                try:
                    receipt = await async_transact_with_contract_function(web3, contract, 'resolveTask', args.private_key, task_id)
                    logging.info(f"resolveTask transaction receipt: {receipt}")
                    processed_tasks.remove(task_id)
                except Exception as e:
                    logging.error(f"Error resolving task {task_id}: {e}")
                    raise

        await asyncio.sleep(2)

async def sync_tensors(sync_version_number):
    relevant_tensors = get_relevant_tensors_for_task(args.task_type)
    for tensor_name in relevant_tensors:
        await initialize_tensor(tensor_name, sync_version_number)

async def submit_solution(task_id, result):
    try:
        receipt = await async_transact_with_contract_function(web3, contract, 'submitSolution', args.private_key, task_id, json.dumps(result).encode('utf-8'))
        logging.info(f"submitSolution transaction receipt: {receipt}")
    except Exception as e:
        logging.error(f"Error submitting solution for task {task_id}: {e}")
        raise

async def upload_tensors_and_grads(error_output, grads, layer_idx):
    if layer_idx == -1:
        layer_label = "final_logits"
    elif layer_idx == -2:
        layer_label = "embed"
    else:
        layer_label = f"layer_{layer_idx}"

    grads_flat = torch.cat([grad.view(-1).to(device) for grad in grads])

    grads_url = await upload_tensor(grads_flat, f'{layer_label}_grads')
    
    block_timestamp = (await web3.eth.get_block('latest'))['timestamp']
    version_number = block_timestamp // TENSOR_VERSION_INTERVAL * TENSOR_VERSION_INTERVAL

    result = {
        'grads_url': grads_url,
        'version_number': version_number
    }

    if error_output is not None:
        result['error_output_url'] = await upload_tensor(error_output, f'{layer_label}_error_output')

    return result

def embed_task(batch):
    global embedding

    inputs = embedding(batch)
    tensors['outputs'] = inputs

def forward_task(layer_idx, inputs):
    global freqs_cis, mask, tensors, transformer_layer

    logging.debug(f"Entering forward_task for layer {layer_idx} with inputs shape {inputs.shape}")

    if torch.isnan(inputs).any() or torch.isinf(inputs).any():
        raise ValueError(f"NaNs or Infs detected in inputs for layer {layer_idx}")

    start_pos = 0
    seqlen = inputs.shape[1]
    freqs_cis_slice = freqs_cis[start_pos: start_pos + seqlen]

    # Update mask dimensions based on input sequence length
    mask_slice = mask[:seqlen, :seqlen]

    bsz = inputs.shape[0]
    if transformer_layer.attention.cache_k is not None and transformer_layer.attention.cache_k.shape[0] != bsz:
        logging.debug(f"Resizing cache_k for layer {layer_idx}")
        transformer_layer.attention.cache_k = torch.zeros(bsz, transformer_layer.attention.cache_k.shape[1], transformer_layer.attention.cache_k.shape[2], transformer_layer.attention.cache_k.shape[3], device=device)
    if transformer_layer.attention.cache_v is not None and transformer_layer.attention.cache_v.shape[0] != bsz:
        logging.debug(f"Resizing cache_v for layer {layer_idx}")
        transformer_layer.attention.cache_v = torch.zeros(bsz, transformer_layer.attention.cache_v.shape[1], transformer_layer.attention.cache_v.shape[2], transformer_layer.attention.cache_v.shape[3], device=device)

    logging.debug(f"Performing forward pass for layer {layer_idx}")
    outputs = transformer_layer(inputs.to(device), start_pos, freqs_cis_slice.to(device), mask_slice.to(device))
    tensors['outputs'] = outputs
    logging.debug(f"Forward pass completed for layer {layer_idx}")

def backward_task(layer_idx, error, inputs, accumulation_steps):
    global freqs_cis, mask, tensors, transformer_layer

    if error is None:
        raise ValueError("Error tensor is None")

    start_pos = 0
    seqlen = inputs.shape[1]
    freqs_cis_slice = freqs_cis[start_pos: start_pos + seqlen]

    # Slice the mask to match the sequence length
    mask_slice = mask[:seqlen, :seqlen]

    microbatch_size = inputs.shape[0] // accumulation_steps

    grads_accumulated = [torch.zeros_like(param, device=device) for param in transformer_layer.parameters()]
    inputs = inputs.clone().detach().requires_grad_(True)

    # List to store gradients for each microbatch
    error_output_list = []

    for i in range(accumulation_steps):
        microbatch_inputs = inputs[i * microbatch_size:(i + 1) * microbatch_size].to(device)
        microbatch_error = error[i * microbatch_size:(i + 1) * microbatch_size].to(device)

        # Clone microbatch_inputs to make them leaf tensors
        microbatch_inputs = microbatch_inputs.clone().detach().requires_grad_(True)

        outputs = transformer_layer(microbatch_inputs, start_pos, freqs_cis_slice.to(device), mask_slice.to(device))

        outputs.retain_grad()
        outputs.backward(microbatch_error, retain_graph=True)

        for j, param in enumerate(transformer_layer.parameters()):
            grads_accumulated[j] += param.grad

        # Store the input gradients for this microbatch
        error_output_list.append(microbatch_inputs.grad.clone())

        transformer_layer.zero_grad()

    grads_accumulated = [grad / accumulation_steps for grad in grads_accumulated]

    # Concatenate the gradients for all microbatches
    tensors['error_output'] = torch.cat(error_output_list, dim=0)
    tensors['updates'] = grads_accumulated

def final_logits_task(inputs, targets, accumulation_steps):
    global final_logits_layer, final_logits_norm, tensors

    # Ensure the inputs and targets tensors are on the correct device
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Clone inputs to make them leaf tensors
    inputs = inputs.clone().detach().requires_grad_(True)

    # Apply RMSNorm to the inputs
    normalized_inputs = final_logits_norm(inputs)

    # Pass the normalized inputs through the final logits layer
    logits = final_logits_layer(normalized_inputs)
    logits.retain_grad()

    microbatch_size = inputs.shape[0] // accumulation_steps

    final_logits_grads_accumulated = [torch.zeros_like(param, device=device) for param in final_logits_layer.parameters()]
    norm_grads_accumulated = [torch.zeros_like(param, device=device) for param in final_logits_norm.parameters()]

    # List to store gradients for each microbatch
    error_output_list = []

    total_loss = 0.0

    for i in range(accumulation_steps):
        microbatch_inputs = inputs[i * microbatch_size:(i + 1) * microbatch_size]
        microbatch_targets = targets[i * microbatch_size:(i + 1) * microbatch_size]

        # Clone microbatch_inputs to make them leaf tensors
        microbatch_inputs = microbatch_inputs.clone().detach().requires_grad_(True)

        normalized_inputs = final_logits_norm(microbatch_inputs)
        logits = final_logits_layer(normalized_inputs)
        logits.retain_grad()

        # Reshape logits to [batch_size * seq_len, vocab_size]
        reshaped_logits = logits.view(-1, model_args.vocab_size)
        # Reshape targets to [batch_size * seq_len]
        reshaped_targets = microbatch_targets.view(-1)

        # Calculate the cross-entropy loss
        loss = F.cross_entropy(reshaped_logits.to(device), reshaped_targets.to(device), ignore_index=tokenizer.pad_id)

        total_loss += loss.item()

        loss.backward(retain_graph=True)

        for j, param in enumerate(final_logits_layer.parameters()):
            final_logits_grads_accumulated[j] += param.grad

        for j, param in enumerate(final_logits_norm.parameters()):
            norm_grads_accumulated[j] += param.grad

        # Store the input gradients for this microbatch
        error_output_list.append(microbatch_inputs.grad.clone())

        final_logits_layer.zero_grad()
        final_logits_norm.zero_grad()

    final_logits_grads_accumulated = [grad / accumulation_steps for grad in final_logits_grads_accumulated]
    norm_grads_accumulated = [grad / accumulation_steps for grad in norm_grads_accumulated]

    combined_grads = norm_grads_accumulated + final_logits_grads_accumulated

    # Concatenate the gradients for all microbatches
    tensors['error_output'] = torch.cat(error_output_list, dim=0)
    tensors['updates'] = combined_grads
    tensors['loss'] = total_loss / accumulation_steps

def embed_backward_task(error, batch, accumulation_steps):
    global embedding, tensors

    if error is None:
        raise ValueError("Error tensor is None")

    # Ensure error tensor is on the correct device
    error = error.to(device)

    microbatch_size = batch.shape[0] // accumulation_steps

    grads_accumulated = torch.zeros_like(embedding.weight, device=device)

    for i in range(accumulation_steps):
        microbatch_batch = batch[i * microbatch_size:(i + 1) * microbatch_size].to(device)
        microbatch_error = error[i * microbatch_size:(i + 1) * microbatch_size].to(device)

        inputs = embedding(microbatch_batch)
        inputs.backward(microbatch_error)

        grads_accumulated += embedding.weight.grad

        embedding.zero_grad()

    grads_accumulated /= accumulation_steps

    logging.debug(f"Accumulated gradients for embedding before AdamW: {grads_accumulated}")

    tensors['updates'] = grads_accumulated

async def upload_final_logits_results():
    error_output_url = await upload_tensor(tensors['error_output'], 'final_logits_error_output')
    grads_url = await upload_tensor(torch.cat([grad.view(-1).to(device) for grad in tensors['updates']]), 'final_logits_grads')
    loss = tensors['loss']

    block_timestamp = (await web3.eth.get_block('latest'))['timestamp']
    version_number = block_timestamp // TENSOR_VERSION_INTERVAL * TENSOR_VERSION_INTERVAL

    return {
        'error_output_url': error_output_url,
        'grads_url': grads_url,
        'loss': loss,
        'version_number': version_number
    }

async def report_sync_status(status):
    try:
        if hasattr(args, 'layer_idx') and args.layer_idx is not None:
            url = f"{args.sync_url}/report_sync?task_type={args.task_type}&layer_idx={args.layer_idx}&status={status}"
        else:
            url = f"{args.sync_url}/report_sync?task_type={args.task_type}&status={status}"
        response = requests.get(url)
        if response.status_code == 200:
            logging.info(f"Reported sync status: {status}")
        else:
            logging.error(f"Failed to report sync status: {response.status_code}")
    except requests.RequestException as e:
        logging.error(f"Exception while reporting sync status: {e}")

async def initialize_tensor(tensor_name, sync_version_number=None):
    global embedding, final_logits_layer, final_logits_norm, transformer_layer

    try:
        url = f"{args.sot_url}/latest_state"
        logging.info(f"Loading tensor {tensor_name} from {url}")
        response = requests.get(url, params={'tensor_name': tensor_name, 'version_number': sync_version_number})
        response.raise_for_status()  # Raise an error for bad status codes

        tensor = torch.load(BytesIO(response.content))
        logging.debug(f"Loaded tensor {tensor_name} with shape {tensor.shape}")

        if "_adam_m" in tensor_name:
            adam_m[tensor_name] = tensor
        elif "_adam_v" in tensor_name:
            adam_v[tensor_name] = tensor
        else:
            tensors[tensor_name] = tensor

        latest_block_timestamps[tensor_name] = sync_version_number
        logging.info(f"Successfully initialized tensor: {tensor_name}")

        if tensor_name == 'embed':
            vocab_size = model_args.vocab_size
            embedding_dim = model_args.dim

            # Reshape the tensor to the correct shape
            reshaped_tensor = tensor.view(vocab_size, embedding_dim)

            embedding = VocabParallelEmbedding(vocab_size, model_args.dim).to(device)

            # Convert reshaped tensor to state_dict format
            state_dict = {'weight': reshaped_tensor}
            embedding.load_state_dict(state_dict)
            logging.info("VocabParallelEmbedding initialized and loaded")

        if tensor_name == 'final_logits':
            final_logits_norm, final_logits_layer = tensor_to_final_logits(tensor)
            logging.info("Final logits layer and RMSNorm initialized and loaded")
        elif "layer_" in tensor_name and "adam_m" not in tensor_name and "adam_v" not in tensor_name:
            layer_idx = int(tensor_name.split('_')[1])
            transformer_layer = tensor_to_block(tensor, layer_idx)
            logging.info(f"TransformerBlock layer {layer_idx} initialized and loaded")

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to initialize tensor {tensor_name} due to request exception: {e}")
        raise
    except Exception as e:
        logging.error(f"Failed to initialize tensor {tensor_name} due to error: {e}")
        raise

def get_relevant_tensors_for_task(task_type):
    relevant_tensors = []
    if task_type.startswith('forward') or task_type.startswith('backward'):
        relevant_tensors = [
            f'layer_{args.layer_idx}'
        ]
    elif task_type in ['embed', 'embed_backward']:
        relevant_tensors = ['embed']
    elif task_type == 'final_logits':
        relevant_tensors = ['final_logits']
    else:
        raise ValueError(f"Invalid task type: {task_type}")
    return relevant_tensors

async def fetch_task(task_id):
    task_tuple = await contract.functions.getTask(task_id).call()
    task = Task(*task_tuple)
    return task_id, task

async def initialize_contracts():
    global token_address, token_contract, stake_amount, subnet_id
    token_address = await contract.functions.token().call()
    token_contract = web3.eth.contract(address=token_address, abi=load_abi('ERC20'))

    stake_amount = await contract.functions.solverStakeAmount().call()

    subnet_id = await contract.functions.subnetId().call()

async def main():
    logging.info("Starting main process")
    torch.set_default_device(device)
    initialize_distributed_environment_and_globals()

    await initialize_contracts()

    # Approve tokens once at the start
    await approve_token_once(web3, token_contract, args.private_key, args.pool_address, 2**256 - 1)

    logging.info("Starting tensor synchronization...")
    relevant_tensors = get_relevant_tensors_for_task(args.task_type)
    reported = False

    # Initialize the last checked task ID and pending tasks
    last_checked_task_id = await contract.functions.numTasks().call() - 1
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
                for tensor_name in relevant_tensors:
                    await initialize_tensor(tensor_name)
                await report_sync_status('synced')
                reported = True

            latest_task_id = await contract.functions.numTasks().call() - 1

            # Gather all task fetching coroutines
            all_task_ids = list(range(last_checked_task_id + 1, latest_task_id + 1)) + list(pending_tasks)
            fetch_tasks = [fetch_task(task_id) for task_id in all_task_ids]
            fetched_tasks = await asyncio.gather(*fetch_tasks)

            # Clear pending tasks to update with new statuses
            pending_tasks.clear()

            # Handle events for tasks
            for task_id, task in fetched_tasks:
                if task.solver == worker_address and task.status == TaskStatus.SolverSelected.value:
                    executor.submit(handle_event, task_id, task, time.time())
                elif task.status < TaskStatus.SolverSelected.value:
                    pending_tasks.add(task_id)
            
            # Update the last checked task ID
            last_checked_task_id = latest_task_id
            
            await asyncio.sleep(args.poll_interval)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(main())
