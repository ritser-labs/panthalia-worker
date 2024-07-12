import argparse
import json
import logging
import requests
import torch
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass
from web3 import Web3
from web3.exceptions import ContractCustomError
from web3.middleware import geth_poa_middleware
from collections import defaultdict
from model import TransformerBlock, VocabParallelEmbedding, ColumnParallelLinear, precompute_freqs_cis, RMSNorm
from device import device
from common import Task, model_args, tokenizer, initialize_distributed_environment, load_abi, upload_tensor, download_file, handle_contract_custom_error, load_error_selectors, transact_with_contract_function
from fairscale.nn.model_parallel.initialize import initialize_model_parallel, model_parallel_is_initialized
from typing import Optional, Tuple
from io import BytesIO
import time
import os
from tqdm import tqdm
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor

class SuppressTracebackFilter(logging.Filter):
    def filter(self, record):
        if 'ConnectionRefusedError' in record.getMessage() or 'MaxRetryError' in record.getMessage():
            return False
        return True

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
logger.addFilter(SuppressTracebackFilter())

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
        'embed', 'forward', 'backward', 'final_logits', 'final_logits_backward', 'embed_backward', 'loss'
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
    return parser.parse_args()

args = parse_args()

os.makedirs(args.local_storage_dir, exist_ok=True)

web3 = Web3(Web3.HTTPProvider(args.rpc_url))
web3.middleware_onion.inject(geth_poa_middleware, layer=0)

worker_account = web3.eth.account.from_key(args.private_key)
worker_address = worker_account.address

subnet_manager_abi = load_abi('SubnetManager')
pool_abi = load_abi('Pool')

contract_address = args.subnet_address
contract = web3.eth.contract(address=contract_address, abi=subnet_manager_abi)
pool_contract = web3.eth.contract(address=args.pool_address, abi=pool_abi)

token_address = contract.functions.token().call()
token_contract = web3.eth.contract(address=token_address, abi=load_abi('ERC20'))

stake_amount = contract.functions.solverStakeAmount().call()

subnet_id = contract.functions.subnetId().call()

model_initialized = False
embedding_initialized = False
tensors = defaultdict(lambda: None)
adam_m = defaultdict(lambda: None)
adam_v = defaultdict(lambda: None)
latest_block_numbers = defaultdict(lambda: 0)  # To store the latest block number processed for each tensor
gradient_update_paused = False
stake_deposited = False
processed_tasks = set()

freqs_cis = None
mask = None
embedding = None  # Define the global embedding variable
final_logits_layer = None
final_logits_norm = None

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

def upload_tensor(tensor):
    tensor_bytes = BytesIO()
    torch.save(tensor, tensor_bytes)
    tensor_bytes.seek(0)

    encoder = MultipartEncoder(
        fields={'tensor': ('tensor.pt', tensor_bytes, 'application/octet-stream')}
    )

    pbar = tqdm(total=encoder.len, unit='B', unit_scale=True, desc='Uploading')
    monitor = MultipartEncoderMonitor(encoder, create_callback(encoder, pbar))

    headers = {'Content-Type': monitor.content_type}
    logging.debug("Starting tensor upload...")

    try:
        response = requests.post(f'{args.sot_url}/upload_tensor', data=monitor, headers=headers, timeout=300)
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

def deposit_stake():
    global stake_deposited
    if not stake_deposited:
        try:
            receipt = transact_with_contract_function(web3, token_contract, 'approve', args.private_key, args.pool_address, stake_amount)
            logging.info(f"Approved token transaction receipt: {receipt}")

            receipt = transact_with_contract_function(web3, pool_contract, 'depositStake', args.private_key, subnet_id, args.group)
            logging.info(f"depositStake transaction receipt: {receipt}")

            stake_deposited = True
        except ContractCustomError as e:
            error_selectors = load_error_selectors(web3)
            handle_contract_custom_error(web3, error_selectors, e)
        except Exception as e:
            logging.error(f"Failed to deposit stake: {e}")
            raise

def handle_event(event):
    task_id = event['args']['taskId']
    solver = event['args']['solver']

    logging.info(f"Received event for task {args.task_type} and id {task_id}")

    if solver.lower() != worker_address.lower():
        return

    task_tuple = contract.functions.getTask(task_id).call()
    task = Task(*task_tuple)
    task_params_bytes = task.params
    task_params = json.loads(task_params_bytes.decode('utf-8'))

    batch = None
    if 'batch_url' in task_params:
        batch = download_json(task_params['batch_url'])

    inputs = None
    if 'inputs_url' in task_params:
        inputs = download_file(task_params['inputs_url'])

    error = None
    if 'error_url' in task_params:
        error = download_file(task_params['error_url'])

    targets = None
    if 'targets_url' in task_params:
        targets = download_json(task_params['targets_url'])

    logits = None
    if 'logits_url' in task_params:
        logits = download_file(task_params['logits_url'])

    pause_gradient_updates()

    task_type = args.task_type
    layer_idx = args.layer_idx
    result = {}
    accumulation_steps = task_params.get('accumulation_steps', 1)
    if task_type == 'embed':
        embed_task(batch)
        result['result_url'] = upload_tensor(tensors['outputs'])
    elif task_type == 'forward':
        forward_task(layer_idx, inputs)
        result['result_url'] = upload_tensor(tensors['outputs'])
    elif task_type == 'backward':
        backward_task(layer_idx, error, inputs, task_params['learning_rate'], task_params['beta1'], task_params['beta2'], task_params['epsilon'], task_params['weight_decay'], task_params['t'], accumulation_steps)
        result = upload_tensors_and_grads(tensors['error_output'], tensors['updates'], tensors[f'layer_{layer_idx}_adam_m'], tensors[f'layer_{layer_idx}_adam_v'], layer_idx)
    elif task_type == 'final_logits':
        final_logits_task(inputs)
        result['result_url'] = upload_tensor(tensors['logits'])
    elif task_type == 'final_logits_backward':
        final_logits_backward_task(error, inputs, task_params['learning_rate'], task_params['beta1'], task_params['beta2'], task_params['epsilon'], task_params['weight_decay'], task_params['t'], accumulation_steps)
        result = upload_tensors_and_grads(tensors['error_output'], tensors['updates'], tensors['final_logits_adam_m'], tensors['final_logits_adam_v'], -1)
    elif task_type == 'embed_backward':
        embed_backward_task(error, batch, task_params['learning_rate'], task_params['beta1'], task_params['beta2'], task_params['epsilon'], task_params['weight_decay'], task_params['t'], accumulation_steps)
        result = upload_tensors_and_grads(None, tensors['updates'], tensors['embed_adam_m'], tensors['embed_adam_v'], -2)
    elif task_type == 'loss':
        loss_task(logits, targets)
        logging.info(1)
        result['loss'] = tensors['loss'].item()
        logging.info(2)
        result['result_url'] = upload_tensor(tensors['logits_grad'])
        logging.info(3)

    result['last_block'] = latest_block_numbers[task_type]
    submit_solution(task_id, result)

    processed_tasks.add(task_id)

    logging.info(f"Processed task {task_id} successfully")

    resume_gradient_updates()

def submit_solution(task_id, result):
    try:
        receipt = transact_with_contract_function(web3, contract, 'submitSolution', args.private_key, task_id, json.dumps(result).encode('utf-8'))
        logging.info(f"submitSolution transaction receipt: {receipt}")
    except Exception as e:
        logging.error(f"Error submitting solution for task {task_id}: {e}")
        raise

def upload_tensors_and_grads(error_output, grads, adam_m_updates, adam_v_updates, layer_idx):
    grads_url = upload_tensor(grads)
    
    adam_m_url = upload_tensor(adam_m_updates)
    adam_v_url = upload_tensor(adam_v_updates)
    
    block_number = web3.eth.block_number
    
    result = {
        'grads_url': grads_url,
        'adam_m_url': adam_m_url,
        'adam_v_url': adam_v_url,
        'block_number': block_number
    }

    if error_output is not None:
        result['error_output_url'] = upload_tensor(error_output)

    return result


def embed_task(batch):
    global embedding

    inputs = embedding(batch)
    tensors['outputs'] = inputs

def forward_task(layer_idx, inputs):
    global freqs_cis, mask, tensors

    logging.debug(f"Entering forward_task for layer {layer_idx} with inputs shape {inputs.shape}")

    if torch.isnan(inputs).any() or torch.isinf(inputs).any():
        raise ValueError(f"NaNs or Infs detected in inputs for layer {layer_idx}")

    layer_key = f'layer_{layer_idx}'
    if layer_key not in tensors or tensors[layer_key] is None:
        logging.error(f"Layer {layer_idx} not found or not initialized")
        raise ValueError(f"Layer {layer_idx} not found or not initialized")

    layer = tensors[layer_key]
    logging.debug(f"Layer {layer_idx} successfully retrieved")

    start_pos = 0
    seqlen = inputs.shape[1]
    freqs_cis_slice = freqs_cis[start_pos: start_pos + seqlen]

    # Update mask dimensions based on input sequence length
    mask_slice = mask[:seqlen, :seqlen]

    bsz = inputs.shape[0]
    if layer.attention.cache_k is not None and layer.attention.cache_k.shape[0] != bsz:
        logging.debug(f"Resizing cache_k for layer {layer_idx}")
        layer.attention.cache_k = torch.zeros(bsz, layer.attention.cache_k.shape[1], layer.attention.cache_k.shape[2], layer.attention.cache_k.shape[3], device=device)
    if layer.attention.cache_v is not None and layer.attention.cache_v.shape[0] != bsz:
        logging.debug(f"Resizing cache_v for layer {layer_idx}")
        layer.attention.cache_v = torch.zeros(bsz, layer.attention.cache_v.shape[1], layer.attention.cache_v.shape[2], layer.attention.cache_v.shape[3], device=device)

    logging.debug(f"Performing forward pass for layer {layer_idx}")
    outputs = layer(inputs.to(device), start_pos, freqs_cis_slice.to(device), mask_slice.to(device))
    tensors['outputs'] = outputs
    logging.debug(f"Forward pass completed for layer {layer_idx}")

def backward_task(layer_idx, error, inputs, learning_rate, beta1, beta2, epsilon, weight_decay, t, accumulation_steps):
    global freqs_cis, mask, tensors

    if error is None:
        raise ValueError(f"Error tensor is None")

    layer = tensors[f'layer_{layer_idx}']

    start_pos = 0
    seqlen = inputs.shape[1]
    freqs_cis_slice = freqs_cis[start_pos: start_pos + seqlen]

    # Slice the mask to match the sequence length
    mask_slice = mask[:seqlen, :seqlen]

    microbatch_size = inputs.shape[0] // accumulation_steps

    grads_accumulated = [torch.zeros_like(param, device=device) for param in layer.parameters()]
    inputs.requires_grad = True
    
    for i in range(accumulation_steps):
        microbatch_inputs = inputs[i * microbatch_size:(i + 1) * microbatch_size].to(device)
        microbatch_error = error[i * microbatch_size:(i + 1) * microbatch_size].to(device)

        outputs = layer(microbatch_inputs, start_pos, freqs_cis_slice.to(device), mask_slice.to(device))

        outputs.retain_grad()
        outputs.backward(microbatch_error, retain_graph=True)

        for j, param in enumerate(layer.parameters()):
            grads_accumulated[j] += param.grad

        layer.zero_grad()

    grads_accumulated = [grad / accumulation_steps for grad in grads_accumulated]

    updates, m_update, v_update = apply_adamw(layer_idx, grads_accumulated, learning_rate, beta1, beta2, epsilon, weight_decay, t)
    tensors['error_output'] = inputs.grad
    tensors['updates'] = updates
    tensors[f'layer_{layer_idx}_adam_m'] = m_update
    tensors[f'layer_{layer_idx}_adam_v'] = v_update

def final_logits_task(inputs):
    global final_logits_layer, final_logits_norm, tensors

    # Apply RMSNorm before the final logits layer
    normalized_inputs = final_logits_norm(inputs)
    logits = final_logits_layer(normalized_inputs.to(device))
    tensors['logits'] = logits

def final_logits_backward_task(error, inputs, learning_rate, beta1, beta2, epsilon, weight_decay, t, accumulation_steps):
    global final_logits_layer, final_logits_norm, tensors

    logging.info(f"Error tensor shape: {error.shape}")

    # Ensure the inputs and error tensors are on the correct device
    inputs = inputs.to(device)
    error = error.to(device)

    # Ensure the inputs require gradients
    inputs.requires_grad = True

    # Apply RMSNorm to the inputs
    normalized_inputs = final_logits_norm(inputs)

    # Pass the normalized inputs through the final logits layer
    logits = final_logits_layer(normalized_inputs)
    logits.retain_grad()

    microbatch_size = inputs.shape[0] // accumulation_steps

    final_logits_grads_accumulated = [torch.zeros_like(param, device=device) for param in final_logits_layer.parameters()]
    norm_grads_accumulated = [torch.zeros_like(param, device=device) for param in final_logits_norm.parameters()]

    for i in range(accumulation_steps):
        microbatch_inputs = inputs[i * microbatch_size:(i + 1) * microbatch_size]
        microbatch_error = error[i * microbatch_size:(i + 1) * microbatch_size]

        normalized_inputs = final_logits_norm(microbatch_inputs)
        logits = final_logits_layer(normalized_inputs)
        logits.retain_grad()

        logits.backward(microbatch_error, retain_graph=True)

        for j, param in enumerate(final_logits_layer.parameters()):
            final_logits_grads_accumulated[j] += param.grad

        for j, param in enumerate(final_logits_norm.parameters()):
            norm_grads_accumulated[j] += param.grad

        final_logits_layer.zero_grad()
        final_logits_norm.zero_grad()

    final_logits_grads_accumulated = [grad / accumulation_steps for grad in final_logits_grads_accumulated]
    norm_grads_accumulated = [grad / accumulation_steps for grad in norm_grads_accumulated]

    combined_grads = final_logits_grads_accumulated + norm_grads_accumulated

    updates, m_update, v_update = apply_adamw(-1, combined_grads, learning_rate, beta1, beta2, epsilon, weight_decay, t)

    tensors['error_output'] = inputs.grad
    tensors['updates'] = updates
    tensors['final_logits_adam_m'] = m_update
    tensors['final_logits_adam_v'] = v_update

def embed_backward_task(error, batch, learning_rate, beta1, beta2, epsilon, weight_decay, t, accumulation_steps):
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

    updates, m_update, v_update = apply_adamw(-2, grads_accumulated, learning_rate, beta1, beta2, epsilon, weight_decay, t)

    tensors['updates'] = updates
    tensors['embed_adam_m'] = m_update
    tensors['embed_adam_v'] = v_update

def loss_task(logits, targets):
    global tensors

    logging.info(f"Logits for loss: {logits.shape}")
    logging.info(f"Targets for loss: {targets.shape}")

    pad_id = tokenizer.pad_id

    batch_size, seq_len, vocab_size = logits.shape

    # Print initial logits before reshaping
    logging.info(f"Initial logits (before reshaping): {logits}")

    logits = logits.reshape(batch_size * seq_len, vocab_size)

    # Print logits after reshaping
    logging.info(f"Logits (after reshaping): {logits.shape}")

    targets = targets.reshape(-1)

    # Compute the loss
    loss = F.cross_entropy(logits.to(device), targets.to(device), ignore_index=pad_id)

    # Ensure logits require gradients
    logits.retain_grad()

    # Perform backward pass to compute gradients with respect to logits
    loss.backward(retain_graph=True)

    # Check for NaNs in gradients
    logging.info(f"Logits gradients for loss: {logits.grad.shape}")

    # Reshape logits gradients to the original shape
    logits_grad = logits.grad.reshape(batch_size, seq_len, vocab_size)
    tensors['logits_grad'] = logits_grad
    tensors['loss'] = loss
    
    if args.detailed_logs:
        # Print logits gradients after reshaping
        logging.info(f"Logits gradients (after reshaping): {logits_grad.shape}")

        # Compute the softmax probabilities from the logits, not from logits_grad
        softmax_probs = F.softmax(logits.reshape(batch_size, seq_len, vocab_size), dim=-1)

        torch.set_printoptions(profile="full")
        logits_for_one_index = str(softmax_probs[0][0])
        with open('output_logits_for_one_index.txt', 'w') as f:
            f.write(logits_for_one_index)
        torch.set_printoptions(profile="default")

        # Print softmax probabilities
        logging.info(f"Softmax probabilities (for first token of first sequence): {softmax_probs[0, 0]}")

        # Identify the token with the highest probability for each position in the sequence
        max_prob_values, max_prob_tokens = torch.max(softmax_probs, dim=-1)

        # Store the max probability tokens in the tensors dictionary
        tensors['max_prob_tokens'] = max_prob_tokens

        # Print and write the entire max_prob_tokens tensor to a text file
        max_prob_tokens_list = max_prob_tokens.cpu().numpy().tolist()
        with open('output_max_prob_tokens.txt', 'w') as f:
            for item in max_prob_tokens_list:
                f.write("%s\n" % item)

        logging.info(f"Max probability tokens: {max_prob_tokens}")
        logging.info(f"Max probability tokens shape: {max_prob_tokens.shape}")
        logging.info(f"Loss: {loss.item()}")

def apply_adamw(layer_idx, grads, learning_rate, beta1, beta2, epsilon, weight_decay, t):
    max_grad_norm = 1.0

    if layer_idx == -1:
        tensor_name = "final_logits"
    elif layer_idx == -2:
        tensor_name = "embed"
    else:
        tensor_name = f"layer_{layer_idx}"

    tensor = tensors[tensor_name]
    if tensor is None:
        raise ValueError(f"Failed to load tensor for {tensor_name}")

    # Ensure tensor is on the correct device
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.to(device)
    elif isinstance(tensor, TransformerBlock):
        # Flatten the parameters of the TransformerBlock
        tensor = block_to_tensor(tensor).to(device)
    else:
        raise TypeError("Unsupported tensor type")

    # Flatten gradients to match the flattened tensor
    grads_flat = torch.cat([grad.view(-1).to(device) for grad in grads])

    m = adam_m[tensor_name]
    v = adam_v[tensor_name]

    if m is None:
        m = torch.zeros_like(tensor)
        adam_m[tensor_name] = m
    else:
        m = m.to(device)

    if v is None:
        v = torch.zeros_like(tensor)
        adam_v[tensor_name] = v
    else:
        v = v.to(device)

    # AdamW update rules applied to the flattened tensors
    m_update = beta1 * m + (1 - beta1) * grads_flat - m
    v_update = beta2 * v + (1 - beta2) * (grads_flat ** 2) - v

    m_hat = (m + m_update) / (1 - beta1 ** t)
    v_hat = (v + v_update) / (1 - beta2 ** t)

    # Gradient update term
    updates = -learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)

    # Weight decay term
    updates -= learning_rate * weight_decay * tensor

    # Clip gradients
    torch.nn.utils.clip_grad_norm_([updates], max_grad_norm)

    logging.info(f"Calculated updates for {tensor_name}")

    return updates, m_update, v_update

def check_and_finalize_verifications():
    current_time = int(time.time())
    for task_id in list(processed_tasks):
        task_tuple = contract.functions.getTask(task_id).call()
        task = Task(*task_tuple)
        task_status = task.status
        time_status_changed = task.timeStatusChanged
        max_dispute_time = contract.functions.maxDisputeTime().call()

        if task_status == 2 and (current_time - time_status_changed) >= max_dispute_time:
            if contract.functions.canVerificationBeFinalized(task_id).call():
                try:
                    receipt = transact_with_contract_function(web3, contract, 'finalizeVerification', args.private_key, task_id)
                    logging.info(f"finalizeVerification transaction receipt: {receipt}")
                    processed_tasks.remove(task_id)
                except Exception as e:
                    logging.error(f"Error finalizing verification for task {task_id}: {e}")
                    raise

def report_sync_status(status):
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

def initialize_tensor(tensor_name, force=False):
    global embedding
    global final_logits_layer
    global final_logits_norm

    try:
        url = os.path.join(args.sot_url, 'tensor_block_number')
        logging.info(f"Checking block number for tensor {tensor_name} from {url}")
        response = requests.get(url, params={'tensor_name': tensor_name})
        response.raise_for_status()
        block_number_info = response.json()
        remote_block_number = block_number_info.get('block_number', 0)
        local_block_number = latest_block_numbers.get(tensor_name, 0)

        if not force and local_block_number >= remote_block_number:
            logging.info(f"Tensor {tensor_name} is already up-to-date with block number {local_block_number}")
            return

        url = os.path.join(args.sot_url, 'latest_state')
        logging.info(f"Loading tensor {tensor_name} from {url}")
        response = requests.get(url, params={'tensor_name': tensor_name})
        response.raise_for_status()  # Raise an error for bad status codes

        tensor = torch.load(BytesIO(response.content))
        logging.debug(f"Loaded tensor {tensor_name} with shape {tensor.shape}")

        if "_adam_m" in tensor_name:
            adam_m[tensor_name] = tensor
        elif "_adam_v" in tensor_name:
            adam_v[tensor_name] = tensor
        elif "layer_" in tensor_name:
            tensors[tensor_name] = tensor_to_block(tensor, args.layer_idx)
        else:
            tensors[tensor_name] = tensor

        latest_block_numbers[tensor_name] = remote_block_number
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
            f'layer_{args.layer_idx}',
            f'layer_{args.layer_idx}_adam_m',
            f'layer_{args.layer_idx}_adam_v'
        ]
    elif task_type in ['embed', 'embed_backward']:
        relevant_tensors = ['embed', 'embed_adam_m', 'embed_adam_v']
    elif task_type in ['final_logits', 'final_logits_backward']:
        relevant_tensors = ['final_logits', 'final_logits_adam_m', 'final_logits_adam_v']
    elif task_type == 'loss':
        relevant_tensors = []
    else:
        raise ValueError(f"Invalid task type: {task_type}")
    return relevant_tensors

def main():
    logging.info("Starting main process")
    torch.set_default_device(device)
    initialize_distributed_environment_and_globals()
    event_filter = contract.events.SolverSelected.create_filter(fromBlock='latest')

    logging.info("Starting tensor synchronization...")
    relevant_tensors = get_relevant_tensors_for_task(args.task_type)
    for tensor_name in relevant_tensors:
        initialize_tensor(tensor_name, force=True)
    reported = False
    while True:
        if not gradient_update_paused:
            for tensor_name in relevant_tensors:
                initialize_tensor(tensor_name)
        deposit_stake()
        if not reported:
            report_sync_status('synced')
            reported = True
        for event in event_filter.get_new_entries():
            handle_event(event)
        check_and_finalize_verifications()
        time.sleep(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
