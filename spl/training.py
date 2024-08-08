import subprocess
import json
import os
import torch
import time
import requests
from model import ModelArgs, Transformer
from common import Model, wait_for_sot, tensor_to_model, initialize_distributed_environment_and_globals, model_args, tokenizer
from device import device
from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3
from io import BytesIO
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s', handlers=[
    logging.StreamHandler()
])

# Constants
SOT_URL = 'http://localhost:5001'
MASTER_WALLETS_FILE = 'master_wallets.json'
batch_size = 32  # Fixed batch size

def random_init_all_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            nn.init.normal_(param, mean=0.0, std=0.02)

# Load the master wallet information
def load_master_wallet():
    with open(MASTER_WALLETS_FILE, 'r') as f:
        wallets = json.load(f)
    if not wallets:
        raise ValueError("No wallets found in the master_wallets.json file.")
    return wallets[0]  # Use the first wallet for simplicity

# Start the SOT service
def start_sot():
    sot_process = subprocess.Popen(
        ['python', 'sot.py', '--public_keys_file', 'master_public_keys.json'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return sot_process

# Load the model from a tensor state
def load_model(tensor_path):
    model_tensor = torch.load(tensor_path, map_location=device)
    model = tensor_to_model(model_tensor)
    random_init_all_params(model)  # Apply random initialization to all parameters
    model.train()  # Set model to training mode
    return model

# Function to sign a message using a private key
def sign_message(message, private_key):
    account = Account.from_key(private_key)
    message_hash = encode_defunct(text=message)
    signed_message = account.sign_message(message_hash)
    return signed_message.signature.hex()

# Function to retrieve a batch from the SOT service
def get_batch(wallet):
    private_key = wallet['private_key']
    address = wallet['address']

    message = json.dumps({
        'endpoint': 'get_batch',
        'nonce': str(int(time.time())),
        'timestamp': int(time.time())
    }, sort_keys=True)

    signature = sign_message(message, private_key)

    headers = {
        'Authorization': f"{message}:{signature}"
    }

    response = requests.post(f'{SOT_URL}/get_batch', headers=headers)
    if response.status_code == 200:
        batch_info = response.json()
        batch_data = requests.get(batch_info['batch_url']).json()
        targets_data = requests.get(batch_info['targets_url']).json()

        # Ensure the batch size is exactly 32
        return torch.tensor(batch_data[:batch_size], dtype=torch.long), torch.tensor(targets_data[:batch_size], dtype=torch.long)
    else:
        print(f"Failed to retrieve batch: {response.text}")
        return None, None

def stable_adamw_update(params, grads, m, v, lr=0.01, weight_decay=0.01, beta1=0.9, beta2=0.999, eps=1e-8, step=1):
    grads = torch.clamp(grads, -1.0, 1.0)  # Clip gradients

    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * grads * grads

    m_hat = m / (1 - beta1 ** step)
    v_hat = v / (1 - beta2 ** step)

    param_update = m_hat / (torch.sqrt(v_hat) + eps)
    param_update += weight_decay * params

    params = params - lr * param_update

    return params, m, v
def apply_adamw(params, grads, m, v, lr=0.002, weight_decay=0.2, beta1=0.9, beta2=0.99, eps=1e-6, step=1):
    # Clip gradients
    grads = torch.nn.utils.clip_grad_norm_(grads, 1.0).to(device)

    if torch.isnan(grads).any() or torch.isinf(grads).any():
        logging.error(f"NaNs or Infs detected in gradients before AdamW update")
        raise ValueError(f"NaNs or Infs detected in gradients")

    params, m, v = stable_adamw_update(
        params,
        grads,
        m,
        v,
        lr,
        weight_decay,
        beta1,
        beta2,
        eps,
        step=step
    )

    return params, m, v

# Function to train the model on the retrieved batch
def train_model(model, batch, targets, params, m, v, lr=0.002, weight_decay=0.2, beta1=0.9, beta2=0.99, eps=1e-6, accumulation_steps=4):
    model.train()

    microbatch_size = batch_size // accumulation_steps

    total_loss = 0.0
    step = 1
    grads_accumulated = torch.zeros(sum(p.numel() for p in model.parameters()), device=device)

    for i in range(accumulation_steps):
        inputs = batch[i * microbatch_size:(i + 1) * microbatch_size].to(device)
        target = targets[i * microbatch_size:(i + 1) * microbatch_size].to(device)

        # Forward pass
        outputs = model(inputs, start_pos=0)
        loss = F.cross_entropy(outputs.view(-1, model_args.vocab_size), target.view(-1), ignore_index=tokenizer.pad_id)

        total_loss += loss.item()

        # Backward pass and accumulate gradients
        model.zero_grad()
        loss.backward()

        # Flatten and collect gradients
        grads = torch.cat([param.grad.view(-1) for param in model.parameters() if param.grad is not None])
        grads_accumulated += grads

        # Detach cache tensors
        if hasattr(model, 'layers'):
            for layer in model.layers:
                if hasattr(layer, 'attention'):
                    layer.attention.cache_k = layer.attention.cache_k.detach()
                    layer.attention.cache_v = layer.attention.cache_v.detach()

        logging.info(f"Accumulation step {i+1}/{accumulation_steps} - Loss: {loss.item()}")

    # Apply AdamW after all accumulation steps
    grads_accumulated /= accumulation_steps
    params, m, v = apply_adamw(params, grads_accumulated, m, v, lr, weight_decay, beta1, beta2, eps, step)

    # Update model parameters
    idx = 0
    for param in model.parameters():
        if param.requires_grad:
            param_size = param.numel()
            param.data = params[idx:idx + param_size].view_as(param).data
            idx += param_size

    logging.info(f"Training completed for the batch. Total loss: {total_loss / accumulation_steps}")

# Adjust main() to initialize optimizer outside the train_model() function call
def main():
    global model
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    initialize_distributed_environment_and_globals('nccl')

    # Load the master wallet
    wallet = load_master_wallet()

    # Start the SOT service
    sot_process = start_sot()
    wait_for_sot(SOT_URL)

    # Load the initial model
    model = Model(model_args).to(device)
    random_init_all_params(model)  # Apply random initialization to all parameters

    if model:
        print("Model loaded and initialized successfully.")
    else:
        print("Failed to load the model. Exiting...")
        sot_process.terminate()
        return

    # Initialize parameter groups for AdamW
    params = torch.cat([param.view(-1) for param in model.parameters() if param.requires_grad])
    m = torch.zeros_like(params, device=device)
    v = torch.zeros_like(params, device=device)

    # Train indefinitely
    try:
        while True:
            print("Retrieving batch from SOT...")
            batch, targets = get_batch(wallet)
            if batch is not None and targets is not None:
                print("Batch retrieved. Starting training...")
                train_model(model, batch, targets, params, m, v, lr=0.001, weight_decay=0.2, beta1=0.9, beta2=0.99, eps=1e-6, accumulation_steps=4)
            else:
                print("Failed to retrieve a valid batch, skipping this round.")
            time.sleep(1)  # Optional: Add a short delay to avoid hammering the SOT service
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        sot_process.terminate()

if __name__ == "__main__":
    main()
