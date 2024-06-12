import argparse
import json
import logging
import requests
import torch
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass
from web3 import Web3
from web3.middleware import geth_poa_middleware
from collections import defaultdict
from model import TransformerBlock, VocabParallelEmbedding, ColumnParallelLinear, precompute_freqs_cis
from common import model_args, tokenizer, device
from fairscale.nn.model_parallel.initialize import initialize_model_parallel, model_parallel_is_initialized
from ipfs import upload_to_ipfs  # Assuming a function to upload files to IPFS or any other storage service
from typing import Optional
from io import BytesIO
import time

import os

logging.basicConfig(level=logging.DEBUG)

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    max_batch_size: int = 32
    max_seq_len: int = 2048

args = ModelArgs()

# Command-line arguments
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
    return parser.parse_args()

args = parse_args()

# Initialize web3
web3 = Web3(Web3.HTTPProvider(args.rpc_url))
web3.middleware_onion.inject(geth_poa_middleware, layer=0)

# Set the worker's Ethereum account
worker_account = web3.eth.account.from_key(args.private_key)
worker_address = worker_account.address

# Smart contract details
contract_address = args.subnet_address
contract_abi = json.loads('YourContractABI')  # Replace with your contract's ABI
pool_address = args.pool_address
pool_abi = json.loads('YourPoolABI')  # Replace with your pool contract's ABI

contract = web3.eth.contract(address=contract_address, abi=contract_abi)
pool_contract = web3.eth.contract(address=pool_address, abi=pool_abi)

# Initialize model and embedding layer in memory
model_initialized = False
embedding_initialized = False
tensors = defaultdict(lambda: None)
adam_m = defaultdict(lambda: None)
adam_v = defaultdict(lambda: None)
last_gradient_update = defaultdict(lambda: None)
gradient_update_paused = False
stake_deposited = False
processed_tasks = set()

# Placeholder for freqs_cis and mask
freqs_cis = None
mask = None

def initialize_model_and_embedding():
    global model_initialized, embedding_initialized, freqs_cis, mask

    if not model_initialized:
        initialize_distributed_environment()
        initialize_model_parallel(model_parallel_size_=1)
        model_initialized = True
    
    if not embedding_initialized:
        vocab_size = tokenizer.get_vocab_size()
        global embedding
        embedding = VocabParallelEmbedding(vocab_size, args.dim).to(device)
        embedding_initialized = True

    # Compute freqs_cis and mask
    global freqs_cis
    freqs_cis = precompute_freqs_cis(
        args.dim // args.n_heads,
        args.max_seq_len * 2,
        args.rope_theta,
    )

    global mask
    mask = torch.triu(torch.full((args.max_seq_len, args.max_seq_len), float('-inf')), diagonal=1).to(device)

def check_for_nans(tensor, name):
    if torch.isnan(tensor).any():
        logging.error(f"NaNs detected in {name}")

def initialize_distributed_environment():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

def download_file(url):
    response = requests.get(url)
    return torch.load(BytesIO(response.content))

def upload_tensor(tensor):
    return upload_to_ipfs(tensor)  # Function to upload the tensor to IPFS or any other storage service

def apply_gradient_updates():
    global tensors, adam_m, adam_v, last_gradient_update

    response = requests.get(args.sot_url, stream=True)
    for line in response.iter_lines():
        if line and not gradient_update_paused:
            update = json.loads(line)
            block_number = update['block_number']
            for tensor_name, sparse_update in update['gradients'].items():
                values = torch.tensor(sparse_update['values'], device=device)
                indices = torch.tensor(sparse_update['indices'], device=device)
                shape = sparse_update['shape']

                # Apply gradient update
                tensor = tensors[tensor_name]
                if tensor is None:
                    tensor = torch.zeros(shape, device=device)
                    tensors[tensor_name] = tensor

                grad_update = torch.sparse_coo_tensor(indices, values, shape, device=device).to_dense()
                tensor.add_(grad_update)
                
                last_gradient_update[tensor_name] = block_number

                # Apply Adam updates if needed
                if tensor_name in adam_m:
                    m = adam_m[tensor_name]
                    v = adam_v[tensor_name]
                    if m is None:
                        m = torch.zeros_like(tensor, device=device)
                        adam_m[tensor_name] = m
                    if v is None:
                        v = torch.zeros_like(tensor, device=device)
                        adam_v[tensor_name] = v

def pause_gradient_updates():
    global gradient_update_paused
    gradient_update_paused = True

def resume_gradient_updates():
    global gradient_update_paused
    gradient_update_paused = False
    apply_gradient_updates()

def sync_tensors_to_latest_state():
    while True:
        response = requests.get(f"{args.sot_url}/latest_state")
        latest_state = response.json()
        need_update = False
        for tensor_name, state in latest_state.items():
            if tensor_name not in tensors or last_gradient_update[tensor_name] < state['block_number']:
                need_update = True
                break
        if not need_update:
            break
        apply_gradient_updates()

def deposit_stake():
    global stake_deposited
    if not stake_deposited:
        tx = pool_contract.functions.depositStake(
            args.subnet_address,
            args.group
        ).transact({'from': worker_address})
        web3.eth.wait_for_transaction_receipt(tx)
        stake_deposited = True

def handle_event(event):
    task_id = event['args']['taskId']
    solver = event['args']['solver']
    
    # Check if the worker is the solver for this task
    if solver.lower() != worker_address.lower():
        return

    task = contract.functions.getTask(task_id).call()
    task_params_bytes = task[6]  # Assuming task.params is the 7th item in the Task struct
    task_params = json.loads(task_params_bytes.decode('utf-8'))

    # Extract URLs from task_params
    batch_file_url = task_params.get('batch_file')
    inputs_file_url = task_params.get('inputs_file')
    error_file_url = task_params.get('error_file')
    targets_file_url = task_params.get('targets_file')

    # Download files
    if batch_file_url:
        batch = download_file(batch_file_url)
    if inputs_file_url:
        inputs = download_file(inputs_file_url)
    if error_file_url:
        error = download_file(error_file_url)
    if targets_file_url:
        targets = download_file(targets_file_url)

    # Pause gradient updates and note block number
    pause_gradient_updates()

    task_type = args.task_type
    if task_type == 'embed':
        embed_task(batch)
        result_url = upload_tensor(tensors['outputs'])
    elif task_type == 'forward':
        forward_task(task_params['layer_idx'], inputs)
        result_url = upload_tensor(tensors['outputs'])
    elif task_type == 'backward':
        backward_task(task_params['layer_idx'], error, inputs, task_params['learning_rate'], task_params['beta1'], task_params['beta2'], task_params['epsilon'], task_params['weight_decay'], task_params['t'])
        result_url = upload_tensors_and_grads(tensors['error_output'], tensors['grads'], task_params['layer_idx'])
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
        loss_task(targets)
        result_url = upload_tensor(tensors['loss'])

    if result_url:
        last_block = last_gradient_update[task_type]
        submit_solution(task_id, result_url, last_block)

    processed_tasks.add(task_id)

    # Resume gradient updates and ensure tensors are up-to-date
    resume_gradient_updates()
    sync_tensors_to_latest_state()

def submit_solution(task_id, result_url, last_block):
    result = {
        'result_url': result_url,
        'last_block': last_block
    }
    contract.functions.submitSolution(task_id, json.dumps(result).encode('utf-8')).transact({'from': worker_address})

def upload_tensors_and_grads(error_output, grads, layer_idx):
    error_output_url = upload_tensor(error_output)
    grads_url = upload_tensor(grads_to_sparse(grads))
    adam_m_sparse, adam_v_sparse = extract_sparse_adam_params(grads, layer_idx)
    adam_m_url = upload_tensor(adam_m_sparse)
    adam_v_url = upload_tensor(adam_v_sparse)
    return json.dumps({'error_output_url': error_output_url, 'grads_url': grads_url, 'adam_m_url': adam_m_url, 'adam_v_url': adam_v_url})

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
            k = max(1, int(flat_grad.numel() * 0.01))  # 1% of the highest elements in magnitude
            topk = torch.topk(flat_grad.abs(), k)
            indices.append(topk.indices)
            values_m.append(adam_m[tensor_name].flatten()[topk.indices])
            values_v.append(adam_v[tensor_name].flatten()[topk.indices])
    indices = torch.cat(indices)
    values_m = torch.cat(values_m)
    values_v = torch.cat(values_v)
    shape = grads[0].shape  # Assuming all grads have the same shape
    return torch.sparse_coo_tensor(indices.unsqueeze(0), values_m, shape, device=device), torch.sparse_coo_tensor(indices.unsqueeze(0), values_v, shape, device=device)

def grads_to_sparse(grads):
    indices, values = [], []
    for grad in grads:
        if grad is not None:
            flat_grad = grad.flatten()
            k = max(1, int(flat_grad.numel() * 0.01))  # 1% of the highest elements in magnitude
            topk = torch.topk(flat_grad.abs(), k)
            indices.append(topk.indices)
            values.append(flat_grad[topk.indices])
            # Zero out the rest of the gradient
            flat_grad[topk.indices] = 0
    indices = torch.cat(indices)
    values = torch.cat(values)
    shape = grads[0].shape  # Assuming all grads have the same shape
    return torch.sparse_coo_tensor(indices.unsqueeze(0), values, shape, device=device)

def embed_task(batch):
    global embedding
    inputs = embedding(batch)
    check_for_nans(inputs, "embedding outputs")
    tensors['outputs'] = inputs

def forward_task(layer_idx, inputs):
    global freqs_cis, mask, tensors
    if torch.isnan(inputs).any() or torch.isinf(inputs).any():
        raise ValueError(f"NaNs or Infs detected in inputs for layer {layer_idx}")

    layer = tensors[f'layer_{layer_idx}']

    start_pos = 0  # Adjust as necessary
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

    start_pos = 0  # Adjust as necessary
    seqlen = inputs.shape[1]
    freqs_cis_slice = freqs_cis[start_pos: start_pos + seqlen]

    outputs = layer(inputs.to(device), start_pos, freqs_cis_slice.to(device), mask.to(device))
    check_for_nans(outputs, f"layer {layer_idx} outputs")

    inputs.requires_grad = True  # Ensure inputs require gradients

    outputs.retain_grad()  # Ensure that gradients for outputs are retained
    outputs.backward(error.to(device), retain_graph=True)  # Backward pass on outputs

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
    output_layer = ColumnParallelLinear(args.dim, state_dict['weight'].shape[0], bias=False).to(device)
    output_layer.load_state_dict(state_dict)

    logits = output_layer(inputs.to(device))
    check_for_nans(logits, "final logits")
    tensors['logits'] = logits

def final_logits_backward_task(error, inputs, learning_rate, beta1, beta2, epsilon, weight_decay, t):
    global tensors
    state_dict = tensors['output_layer_state_dict']
    output_layer = ColumnParallelLinear(args.dim, state_dict['weight'].shape[0], bias=False).to(device)
    output_layer.load_state_dict(state_dict)
    
    logits = output_layer(inputs.to(device))
    
    logits.retain_grad()

    logits.backward(error.to(device), retain_graph=True)  # Backward pass on logits

    logits_grad = logits.grad  # Get the gradients with respect to logits
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

    error = error.view(embeddings.shape)  # Ensure the gradient tensor matches the output tensor shape
    logging.info(f"Reshaped error tensor shape: {error.shape}")
    
    embeddings.retain_grad()

    embeddings.backward(error.to(device), retain_graph=True)

    grads = [param.grad for param in embedding.parameters() if param.grad is not None]
    logging.info(f"Gradients for embedding: {grads}")

    apply_adamw(-2, grads, learning_rate, beta1, beta2, epsilon, weight_decay, t)
    tensors['error_output'] = grads
    tensors['grads'] = grads

def loss_task(targets):
    global tensors
    logits = tensors['logits']
    logging.info(f"Logits for loss: {logits.shape}")
    logging.info(f"Targets for loss: {targets.shape}")

    pad_id = tokenizer.pad_id

    batch_size, seq_len, vocab_size = logits.shape
    logits = logits.reshape(batch_size * seq_len, vocab_size)  # Shape: [batch_size * seq_len, vocab_size]

    targets = targets.reshape(-1)  # Shape: [batch_size * seq_len]

    loss = F.cross_entropy(logits.to(device), targets.to(device), ignore_index=pad_id)
    tensors['loss'] = loss.item()

    logits.retain_grad()
    loss.backward(retain_graph=True)  # Retain graph for backward pass
    check_for_nans(logits.grad, "logits gradients")
    logging.info(f"Logits gradients for loss: {logits.grad.shape}")

    logits_grad = logits.grad.reshape(batch_size, seq_len, vocab_size)  # Shape: [batch_size, seq_len, vocab_size]

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
        task = contract.functions.getTask(task_id).call()
        task_status = task[0]  # Assuming task.status is the first item in the Task struct
        time_status_changed = task[3]  # Assuming task.timeStatusChanged is the fourth item in the Task struct
        max_dispute_time = contract.functions.maxDisputeTime().call()

        if task_status == 2 and (current_time - time_status_changed) >= max_dispute_time:  # Status 2 means SolutionSubmitted
            if contract.functions.canVerificationBeFinalized(task_id).call():
                tx = contract.functions.finalizeVerification(task_id).transact({'from': worker_address})
                web3.eth.wait_for_transaction_receipt(tx)
                processed_tasks.remove(task_id)

def main():
    initialize_model_and_embedding()
    event_filter = contract.events.SolverSelected.createFilter(fromBlock='latest')
    while True:
        # Ensure tensors are up-to-date before processing tasks
        apply_gradient_updates()
        sync_tensors_to_latest_state()
        deposit_stake()
        for event in event_filter.get_new_entries():
            handle_event(event)
        check_and_finalize_verifications()
        time.sleep(10)  # Adjust the sleep time as needed

if __name__ == "__main__":
    main()
