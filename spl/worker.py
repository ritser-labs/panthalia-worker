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
from common import model_args, tokenizer, device, initialize_distributed_environment, load_abi, upload_tensor, download_file, handle_contract_custom_error, load_error_selectors
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
    parser.add_argument('--local_storage_dir', type=str, default='local_storage', help="Directory for local storage of files")
    parser.add_argument('--backend', type=str, default='nccl', help="Distributed backend to use (default: nccl, use 'gloo' for macOS)")
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
        embedding = VocabParallelEmbedding(vocab_size, model_args.dim).to(device)
        embedding_initialized = True

    freqs_cis = precompute_freqs_cis(
        model_args.dim // model_args.n_heads,
        model_args.max_seq_len * 2,
        model_args.rope_theta,
    )

    mask = torch.triu(torch.full((model_args.max_seq_len, model_args.max_seq_len), float('-inf')), diagonal=1).to(device)

def check_for_nans(tensor, name):
    if torch.isnan(tensor).any():
        logging.error(f"NaNs detected in {name}")

def download_file(url):
    response = requests.get(url)
    return torch.load(BytesIO(response.content))

def upload_tensor(tensor):
    local_file_path = os.path.join(args.local_storage_dir, f'{int(time.time())}.pt')
    torch.save(tensor, local_file_path)
    return f'file://{local_file_path}'

def stream_gradients(task_id, gradients, block_number=0):
    data = {
        'task_id': task_id,
        'gradients': gradients,
        'block_number': block_number
    }
    logging.debug(f"Sending data to /stream_gradients: {data}")
    response = requests.post(f"{args.sot_url}/stream_gradients", json=data)
    if response.status_code != 200:
        logging.error(f"Failed to stream gradients: {response.text}")
    else:
        logging.info(f"Streamed gradients successfully for task {task_id}")

def pause_gradient_updates():
    global gradient_update_paused
    gradient_update_paused = True

def resume_gradient_updates():
    global gradient_update_paused
    gradient_update_paused = False
    apply_gradient_updates()

def deposit_stake():
    global stake_deposited
    if not stake_deposited:
        try:
            # Approve transaction
            approve_tx = token_contract.functions.approve(
                args.pool_address,
                stake_amount
            ).build_transaction({
                'chainId': web3.eth.chain_id,
                'gas': 200000,
                'gasPrice': web3.eth.gas_price,
                'nonce': web3.eth.get_transaction_count(worker_address),
            })

            signed_approve_tx = worker_account.sign_transaction(approve_tx)
            approve_tx_hash = web3.eth.send_raw_transaction(signed_approve_tx.rawTransaction)
            web3.eth.wait_for_transaction_receipt(approve_tx_hash)

            # Deposit stake transaction
            deposit_tx = pool_contract.functions.depositStake(
                subnet_id,
                args.group
            ).build_transaction({
                'chainId': web3.eth.chain_id,
                'gas': 500000,
                'gasPrice': web3.eth.gas_price,
                'nonce': web3.eth.get_transaction_count(worker_address),
            })

            signed_deposit_tx = worker_account.sign_transaction(deposit_tx)
            deposit_tx_hash = web3.eth.send_raw_transaction(signed_deposit_tx.rawTransaction)
            web3.eth.wait_for_transaction_receipt(deposit_tx_hash)

            stake_deposited = True

            # Report staking status
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

    task = contract.functions.getTask(task_id).call()
    task_params_bytes = task[6]
    task_params = json.loads(task_params_bytes.decode('utf-8'))

    batch_file_url = task_params.get('batch_file')
    inputs_file_url = task_params.get('inputs_file')
    error_file_url = task_params.get('error_file')
    targets_file_url = task_params.get('targets_file')

    if batch_file_url:
        batch = download_file(batch_file_url)
    if inputs_file_url:
        inputs = download_file(inputs_file_url)
    if error_file_url:
        error = download_file(error_file_url)
    if targets_file_url:
        targets = download_file(targets_file_url)

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
    block_number = web3.eth.blockNumber  # Fetch the current block number
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
        layer.attention.cache_v = torch.zeros(bsz, layer.attention.cache_v.shape[1], layer.attention.cache_v.shape[2], layer.attention.cache_v.shape[3], device.device)

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

def loss_task(targets):
    global tensors
    logits = tensors['logits']
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
        task = contract.functions.getTask(task_id).call()
        task_status = task[0]
        time_status_changed = task[3]
        max_dispute_time = contract.functions.maxDisputeTime().call()

        if task_status == 2 and (current_time - time_status_changed) >= max_dispute_time:
            if contract.functions.canVerificationBeFinalized(task_id).call():
                tx = contract.functions.finalizeVerification(task_id).transact({'from': worker_address})
                web3.eth.wait_for_transaction_receipt(tx)
                processed_tasks.remove(task_id)

def report_sync_status(status):
    try:
        if "layer_idx" not in args:
            url = f"http://localhost:5002/report_sync?task_type={args.task_type}&status={status}"
        else:
            url = f"http://localhost:5002/report_sync?task_type={args.task_type}&layer_idx={args.layer_idx}&status={status}"
        response = requests.get(url)
        if response.status_code == 200:
            logging.info(f"Reported sync status: {status}")
        else:
            logging.error(f"Failed to report sync status: {response.status_code}")
    except requests.RequestException as e:
        logging.error(f"Exception while reporting sync status: {e}")

def sync_tensors_to_latest_state(task_type):
    relevant_tensors = get_relevant_tensors_for_task(task_type)
    outdated_tensors = []
    try:
        response = requests.get(f"{args.sot_url}/latest_state", stream=True)
        for line in response.iter_lines():
            if line:
                latest_state = json.loads(line)
                for tensor_name in relevant_tensors:
                    current_block_number = last_gradient_update.get(tensor_name, -1)
                    latest_block_number = latest_state.get(tensor_name, {}).get('block_number', 0)
                    if current_block_number < latest_block_number:
                        outdated_tensors.append(tensor_name)
                break  # Stop after receiving the latest state once
    except requests.exceptions.RequestException as e:
        logging.error(f"Request exception in sync_tensors_to_latest_state: {e}")

    if outdated_tensors:
        apply_gradient_updates(outdated_tensors)  # Update only outdated tensors

def apply_gradient_updates(outdated_tensors):
    try:
        for tensor_name in outdated_tensors:
            # Fetch tensor size
            size_response = requests.get(f"{args.sot_url}/tensor_size", params={'tensor_name': tensor_name})
            size_data = size_response.json()
            tensor_size = size_data.get('size')

            response = requests.get(f"{args.sot_url}/latest_state", params={'tensor_names': tensor_name}, stream=True)
            pbar = tqdm.tqdm(total=tensor_size, desc=f"Syncing {tensor_name}", unit='elements')
            
            for line in response.iter_lines():
                if line:
                    update = json.loads(line)
                    if tensor_name in update:
                        sparse_update = update[tensor_name]['state']
                        values = torch.tensor(sparse_update['values'], device=device)
                        indices = torch.tensor(sparse_update['indices'], device=device)
                        shape = sparse_update['shape']
                        tensor = tensors.get(tensor_name)
                        if tensor is None:
                            tensor = torch.zeros(shape, device=device)
                            tensors[tensor_name] = tensor
                        grad_update = torch.sparse_coo_tensor(indices, values, shape, device=device).to_dense()
                        tensor.add_(grad_update)
                        pbar.update(values.numel())
                        last_gradient_update[tensor_name] = update[tensor_name]['block_number']
            pbar.close()
    except requests.exceptions.RequestException as e:
        logging.error(f"Request exception while updating gradients: {e}")

def get_relevant_tensors_for_task(task_type):
    if task_type.startswith('forward_layer') or task_type.startswith('backward_layer'):
        layer_idx = int(task_type.split('_')[-1])
        return [f'layer_{layer_idx}_weights', f'layer_{layer_idx}_adam_m', f'layer_{layer_idx}_adam_v']
    elif task_type in ['embed', 'embed_backward']:
        return ['embedding_weights', 'embedding_adam_m', 'embedding_adam_v']
    elif task_type in ['final_logits', 'final_logits_backward']:
        return ['final_logits_weights', 'final_logits_adam_m', 'final_logits_adam_v']
    elif task_type == 'loss':
        return ['logits']
    return []

def main():
    initialize_model_and_embedding()
    event_filter = contract.events.SolverSelected.create_filter(fromBlock='latest')

    # Synchronize tensors to the latest state and report status
    logging.info("Starting tensor synchronization...")
    sync_tensors_to_latest_state(args.task_type)
    report_sync_status('synced')

    while True:
        apply_gradient_updates(args.task_type)
        sync_tensors_to_latest_state(args.task_type)
        deposit_stake()
        for event in event_filter.get_new_entries():
            handle_event(event)
        check_and_finalize_verifications()
        time.sleep(10)

if __name__ == "__main__":
    main()
