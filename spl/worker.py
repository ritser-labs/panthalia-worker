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
from model import TransformerBlock, VocabParallelEmbedding, ColumnParallelLinear, precompute_freqs_cis
from common import Task, model_args, tokenizer, device, initialize_distributed_environment, load_abi, upload_tensor, download_file, handle_contract_custom_error, load_error_selectors
from fairscale.nn.model_parallel.initialize import initialize_model_parallel, model_parallel_is_initialized
from typing import Optional
from io import BytesIO
import time
import os
import socket
import tqdm

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
last_gradient_update = defaultdict(lambda: None)
gradient_update_paused = False
stake_deposited = False
processed_tasks = set()

freqs_cis = None
mask = None

def initialize_model_and_embedding():
    global model_initialized, embedding_initialized, freqs_cis, mask

    if not model_initialized:
        initialize_distributed_environment(args.backend)
        initialize_model_parallel(model_parallel_size_=1)
        model_initialized = True
    
    if not embedding_initialized:
        vocab_size = tokenizer.get_vocab_size()
        global embedding
        embedding = VocabParallelEmbedding(vocab_size, model_args.dim).to(device)  # Ensure embedding is on the correct device
        embedding_initialized = True

    freqs_cis = precompute_freqs_cis(
        model_args.dim // model_args.n_heads,
        model_args.max_seq_len * 2,
        model_args.rope_theta,
    )

    mask = torch.triu(torch.full((model_args.max_seq_len, model_args.max_seq_len), float('-inf')), diagonal=1).to(device)  # Ensure mask is on the correct device


def check_for_nans(tensor, name):
    if torch.isnan(tensor).any():
        logging.error(f"NaNs detected in {name}")

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

def upload_tensor(tensor):
    tensor_name = f'{int(time.time())}.pt'
    local_file_path = os.path.join(args.local_storage_dir, f'{int(time.time())}.pt')
    torch.save(tensor, local_file_path)
    return f'{args.sot_url}/data/{tensor_name}'

def pause_gradient_updates():
    global gradient_update_paused
    gradient_update_paused = True

def resume_gradient_updates():
    global gradient_update_paused
    gradient_update_paused = False

def build_transaction(function, value=0):
    nonce = web3.eth.get_transaction_count(worker_address)
    gas_price = web3.eth.gas_price
    return function.build_transaction({
        'chainId': web3.eth.chain_id,
        'gas': 200000,
        'gasPrice': gas_price,
        'nonce': nonce,
        'value': value,
    })

def sign_and_send_transaction(tx):
    signed_tx = worker_account.sign_transaction(tx)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    return web3.eth.wait_for_transaction_receipt(tx_hash)

def deposit_stake():
    global stake_deposited
    if not stake_deposited:
        try:
            print("Submitting approve transaction for depositing stake...")
            approve_tx = build_transaction(token_contract.functions.approve(args.pool_address, stake_amount))
            sign_and_send_transaction(approve_tx)

            print("Submitting depositStake transaction...")
            deposit_tx = build_transaction(pool_contract.functions.depositStake(subnet_id, args.group))
            sign_and_send_transaction(deposit_tx)

            stake_deposited = True
            report_stake_status()
        except ContractCustomError as e:
            error_selectors = load_error_selectors(web3)
            handle_contract_custom_error(web3, error_selectors, e)
        except Exception as e:
            logging.error(f"Failed to deposit stake: {e}")
            raise

def report_stake_status():
    try:
        response = requests.post(f"{args.sot_url}/report_stake", json={'worker_address': worker_address})
        if response.status_code == 200:
            logging.info(f"Reported staking status for worker {worker_address}")
        else:
            logging.error(f"Failed to report staking status for worker {worker_address}: {response.text}")
    except requests.RequestException as e:
        logging.error(f"Exception while reporting staking status for worker {worker_address}: {e}")

def handle_event(event):
    task_id = event['args']['taskId']
    solver = event['args']['solver']

    print(f"Received event for task {args.task_type} and id {task_id}")
    
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
        targets = download_file(task_params['targets_url'])
    
    logits = None
    if 'logits_url' in task_params:
        logits = download_file(task_params['logits_url'])

    pause_gradient_updates()

    task_type = args.task_type
    layer_idx = args.layer_idx
    if task_type == 'embed':
        embed_task(batch)
        result_url = upload_tensor(tensors['outputs'])
    elif task_type == 'forward':
        forward_task(layer_idx, inputs)
        result_url = upload_tensor(tensors['outputs'])
    elif task_type == 'backward':
        backward_task(layer_idx, error, inputs, task_params['learning_rate'], task_params['beta1'], task_params['beta2'], task_params['epsilon'], task_params['weight_decay'], task_params['t'])
        result_url = upload_tensors_and_grads(tensors['error_output'], tensors['grads'], layer_idx)
    elif task_type == 'final_logits':
        final_logits_task(inputs)
        result_url = upload_tensor(tensors['logits'])
    elif task_type == 'final_logits_backward':
        final_logits_backward_task(error, inputs, task_params['learning_rate'], task_params['beta1'], task_params['beta2'], task_params['epsilon'], task_params['weight_decay'], task_params['t'])
        result_url = upload_tensors_and_grads(tensors['error_output'], tensors['grads'], -1)
    elif task_type == 'embed_backward':
        embed_backward_task(error, batch, task_params['learning_rate'], task_params['beta1'], task_params['beta2'], task_params['epsilon'], task_params['weight_decay'], task_params['t'])
        result_url = upload_tensors_and_grads(tensors['error_output'], tensors['grads'], -2)
    elif task_type == 'loss':
        loss_task(logits, targets)
        result_url = upload_tensor(tensors['logits_grad'])

    if result_url:
        last_block = last_gradient_update[task_type]
        submit_solution(task_id, result_url, last_block)

    processed_tasks.add(task_id)
    
    print(f"Processed task {task_id} successfully")

    resume_gradient_updates()

def submit_solution(task_id, result_url, last_block):
    result = {
        'result_url': result_url,
        'last_block': last_block
    }
    print(f"Submitting solution for task {task_id}...")
    tx = build_transaction(contract.functions.submitSolution(task_id, json.dumps(result).encode('utf-8')))
    sign_and_send_transaction(tx)

def upload_tensors_and_grads(error_output, grads, layer_idx):
    error_output_url = upload_tensor(error_output)
    grads_url = upload_tensor(grads_to_sparse(grads))
    adam_m_sparse, adam_v_sparse = extract_sparse_adam_params(grads, layer_idx)
    adam_m_url = upload_tensor(adam_m_sparse)
    adam_v_url = upload_tensor(adam_v_sparse)
    block_number = web3.eth.blockNumber
    return {
        'error_output_url': error_output_url,
        'grads_url': grads_url,
        'adam_m_url': adam_m_url,
        'adam_v_url': adam_v_url,
        'block_number': block_number
    }

def extract_sparse_adam_params(grads, layer_idx):
    indices, values_m, values_v = [], [], []
    if layer_idx == -1:
        tensor_name = "output"
    elif layer_idx == -2:
        tensor_name = "embedding"
    else:
        tensor_name = f"layer_{layer_idx}"
    
    for grad in grads:
        if grad is not None:
            flat_grad = grad.flatten()
            k = max(1, int(flat_grad.numel() * 0.01))
            topk = torch.topk(flat_grad.abs(), k)
            indices.append(topk.indices)
            values_m.append(adam_m[tensor_name].flatten()[topk.indices])
            values_v.append(adam_v[tensor_name].flatten()[topk.indices])
    indices = torch.cat(indices)
    values_m = torch.cat(values_m)
    values_v = torch.cat(values_v)
    shape = grads[0].shape
    return torch.sparse_coo_tensor(indices.unsqueeze(0), values_m, shape, device=device), torch.sparse_coo_tensor(indices.unsqueeze(0), values_v, shape, device=device)

def grads_to_sparse(grads):
    indices, values = [], []
    for grad in grads:
        if grad is not None:
            flat_grad = grad.flatten()
            k = max(1, int(flat_grad.numel() * 0.01))
            topk = torch.topk(flat_grad.abs(), k)
            indices.append(topk.indices)
            values.append(flat_grad[topk.indices])
            flat_grad[topk.indices] = 0
    indices = torch.cat(indices)
    values = torch.cat(values)
    shape = grads[0].shape
    return torch.sparse_coo_tensor(indices.unsqueeze(0), values, shape, device=device)

def embed_task(batch):
    global embedding

    # Ensure batch is a tensor and move to the correct device
    #if not isinstance(batch, torch.Tensor):
    #    print("Batch is not a tensor, converting now.")
    #    batch = torch.tensor(batch, dtype=torch.long)

    #batch = batch.to(device)  # Move batch to the correct device

    #print(f"Batch type: {type(batch)}, Batch shape: {batch.shape}")

    inputs = embedding(batch)
    check_for_nans(inputs, "embedding outputs")
    tensors['outputs'] = inputs

def forward_task(layer_idx, inputs):
    global freqs_cis, mask, tensors
    if torch.isnan(inputs).any() or torch.isinf(inputs).any():
        raise ValueError(f"NaNs or Infs detected in inputs for layer {layer_idx}")

    layer = tensors[f'layer_{layer_idx}']

    start_pos = 0
    seqlen = inputs.shape[1]
    freqs_cis_slice = freqs_cis[start_pos: start_pos + seqlen]

    bsz = inputs.shape[0]
    if layer.attention.cache_k is not None and layer.attention.cache_k.shape[0] != bsz:
        layer.attention.cache_k = torch.zeros(bsz, layer.attention.cache_k.shape[1], layer.attention.cache_k.shape[2], layer.attention.cache_k.shape[3], device=device)
    if layer.attention.cache_v is not None and layer.attention.cache_v.shape[0] != bsz:
        layer.attention.cache_v = torch.zeros(bsz, layer.attention.cache_v.shape[1], layer.attention.cache_v.shape[2], layer.attention.cache_v.shape[3], device=device)

    outputs = layer(inputs.to(device), start_pos, freqs_cis_slice.to(device), mask.to(device))
    check_for_nans(outputs, f"layer {layer_idx} outputs")
    tensors['outputs'] = outputs

def backward_task(layer_idx, error, inputs, learning_rate, beta1, beta2, epsilon, weight_decay, t):
    global freqs_cis, mask, tensors

    if error is None:
        raise ValueError(f"Error tensor is None")

    layer = tensors[f'layer_{layer_idx}']

    start_pos = 0
    seqlen = inputs.shape[1]
    freqs_cis_slice = freqs_cis[start_pos: start_pos + seqlen]

    outputs = layer(inputs.to(device), start_pos, freqs_cis_slice.to(device), mask.to(device))
    check_for_nans(outputs, f"layer {layer_idx} outputs")

    inputs.requires_grad = True

    outputs.retain_grad()
    outputs.backward(error.to(device), retain_graph=True)

    if inputs.grad is None:
        raise ValueError(f"Gradient for inputs is None after backward pass for layer {layer_idx}")

    check_for_nans(inputs.grad, f"Gradient for inputs in layer {layer_idx}")

    grads = [param.grad for param in layer.parameters() if param.grad is not None]
    logging.debug(f"Gradients for layer {layer_idx}: {grads}")

    for i, grad in enumerate(grads):
        check_for_nans(grad, f"Gradient {i} for layer {layer_idx}")

    apply_adamw(layer_idx, grads, learning_rate, beta1, beta2, epsilon, weight_decay, t)
    tensors['error_output'] = (inputs.grad, grads)
    tensors['grads'] = grads

def final_logits_task(inputs):
    global tensors
    state_dict = tensors['output_layer_state_dict']
    output_layer = ColumnParallelLinear(model_args.dim, state_dict['weight'].shape[0], bias=False).to(device)
    output_layer.load_state_dict(state_dict)

    logits = output_layer(inputs.to(device))
    check_for_nans(logits, "final logits")
    tensors['logits'] = logits

def final_logits_backward_task(error, inputs, learning_rate, beta1, beta2, epsilon, weight_decay, t):
    global tensors
    state_dict = tensors['output_layer_state_dict']
    output_layer = ColumnParallelLinear(model_args.dim, state_dict['weight'].shape[0], bias=False).to(device)
    output_layer.load_state_dict(state_dict)
    
    logits = output_layer(inputs.to(device))
    
    logits.retain_grad()

    logits.backward(error.to(device), retain_graph=True)

    logits_grad = logits.grad
    logging.debug(f"Gradients for logits: {logits_grad.shape}")

    logits_grad = torch.einsum('bij,jk->bik', logits_grad, output_layer.weight)

    grads = [param.grad for param in output_layer.parameters() if param.grad is not None]
    logging.debug(f"Gradients for final logits layer parameters: {grads}")

    apply_adamw(-1, grads, learning_rate, beta1, beta2, epsilon, weight_decay, t)
    tensors['error_output'] = (logits_grad, grads)
    tensors['grads'] = grads

def embed_backward_task(error, batch, learning_rate, beta1, beta2, epsilon, weight_decay, t):
    global embedding, tensors
    logging.info(f"Error tensor shape: {error.shape}")

    embeddings = embedding(batch)

    error = error.view(embeddings.shape)
    logging.info(f"Reshaped error tensor shape: {error.shape}")
    
    embeddings.retain_grad()

    embeddings.backward(error.to(device), retain_graph=True)

    grads = [param.grad for param in embedding.parameters() if param.grad is not None]
    logging.info(f"Gradients for embedding: {grads}")

    apply_adamw(-2, grads, learning_rate, beta1, beta2, epsilon, weight_decay, t)
    tensors['error_output'] = grads
    tensors['grads'] = grads

def loss_task(logits, targets):
    global tensors
    logging.info(f"Logits for loss: {logits.shape}")
    logging.info(f"Targets for loss: {targets.shape}")

    pad_id = tokenizer.pad_id

    batch_size, seq_len, vocab_size = logits.shape
    logits = logits.reshape(batch_size * seq_len, vocab_size)

    targets = targets.reshape(-1)

    loss = F.cross_entropy(logits.to(device), targets.to(device), ignore_index=pad_id)
    tensors['loss'] = loss.item()

    logits.retain_grad()
    loss.backward(retain_graph=True)
    check_for_nans(logits.grad, "logits gradients")
    logging.info(f"Logits gradients for loss: {logits.grad.shape}")

    logits_grad = logits.grad.reshape(batch_size, seq_len, vocab_size)

    tensors['logits_grad'] = logits_grad

def apply_adamw(layer_idx, grads, learning_rate, beta1, beta2, epsilon, weight_decay, t):
    max_grad_norm = 1.0

    if layer_idx == -1:
        tensor_name = "output"
    elif layer_idx == -2:
        tensor_name = "embedding"
    else:
        tensor_name = f"layer_{layer_idx}"

    tensor = tensors[tensor_name]
    if tensor is None:
        raise ValueError(f"Failed to load tensor for {tensor_name}")

    m = adam_m[tensor_name]
    v = adam_v[tensor_name]
    
    if m is None:
        m = torch.zeros_like(tensor, device=device)
        adam_m[tensor_name] = m
    if v is None:
        v = torch.zeros_like(tensor, device=device)
        adam_v[tensor_name] = v

    for i, param in enumerate(tensor):
        if param.requires_grad:
            if param.grad is not None:
                param.grad.zero_()

            param.data -= learning_rate * weight_decay * param.data

            m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
            v[i] = beta2 * v[i] + (1 - beta2) * (grads[i] ** 2)

            m_hat = m[i] / (1 - beta1 ** t)
            v_hat = v[i] / (1 - beta2 ** t)

            param.data -= learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)

            check_for_nans(param.data, f"Updated param data in {tensor_name}")

            torch.nn.utils.clip_grad_norm_(param, max_grad_norm)

    logging.info(f"Updated state dict for {tensor_name}")

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
                print(f"Submitting finalizeVerification transaction for task {task_id}...")
                tx = build_transaction(contract.functions.finalizeVerification(task_id))
                sign_and_send_transaction(tx)
                processed_tasks.remove(task_id)

def report_sync_status(status):
    try:
        if hasattr(args, 'layer_idx') and args.layer_idx is not None:
            url = f"http://localhost:5002/report_sync?task_type={args.task_type}&layer_idx={args.layer_idx}&status={status}"
        else:
            url = f"http://localhost:5002/report_sync?task_type={args.task_type}&status={status}"
        response = requests.get(url)
        if response.status_code == 200:
            logging.info(f"Reported sync status: {status}")
        else:
            logging.error(f"Failed to report sync status: {response.status_code}")
    except requests.RequestException as e:
        logging.error(f"Exception while reporting sync status: {e}")

def sync_tensors_to_latest_state(tensor_name):
    try:
        response = requests.get(f"{args.sot_url}/latest_state", params={'tensor_name': tensor_name})
        if response.status_code == 200:
            latest_state = torch.load(BytesIO(response.content))
            tensors[tensor_name] = latest_state
            last_gradient_update[tensor_name] = response.headers.get('block_number', 0)
        else:
            logging.error(f"Failed to sync tensor: {tensor_name}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request exception in sync_tensors_to_latest_state: {e}")

def apply_gradient_updates(tensor_name):
    try:
        response = requests.get(f"{args.sot_url}/latest_state", params={'tensor_name': tensor_name})
        if response.status_code == 200:
            latest_state = torch.load(BytesIO(response.content))
            tensor = tensors.get(tensor_name, torch.zeros_like(latest_state))
            tensor.add_(latest_state)
            tensors[tensor_name] = tensor
            last_gradient_update[tensor_name] = response.headers.get('block_number', 0)
        else:
            logging.error(f"Failed to apply gradient updates: {tensor_name}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request exception while updating gradients: {e}")

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
    initialize_model_and_embedding()
    event_filter = contract.events.SolverSelected.create_filter(fromBlock='latest')

    logging.info("Starting tensor synchronization...")
    relevant_tensors = get_relevant_tensors_for_task(args.task_type)
    for tensor_name in relevant_tensors:
        sync_tensors_to_latest_state(tensor_name)
    report_sync_status('synced')

    while True:
        if not gradient_update_paused:
            for tensor_name in relevant_tensors:
                apply_gradient_updates(tensor_name)
        deposit_stake()
        for event in event_filter.get_new_entries():
            handle_event(event)
        check_and_finalize_verifications()
        time.sleep(10)

if __name__ == "__main__":
    main()
