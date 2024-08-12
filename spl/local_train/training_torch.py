import subprocess
import json
import os
import torch
import time
import requests
from spl.adapters.llama3 import ModelArgs, Transformer
from common import Model, wait_for_sot, tensor_to_model, model_args, tokenizer, model_adapter
from device import device
from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3
from io import BytesIO
import torch.nn as nn
import torch.nn.functional as F

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

# Function to train the model on the retrieved batch
def train_model(model, optimizer, batch, targets, accumulation_steps):
    model.train()

    microbatch_size = batch_size // accumulation_steps

    total_loss = 0.0
    for i in range(accumulation_steps):
        inputs = batch[i * microbatch_size:(i + 1) * microbatch_size].to(device)
        target = targets[i * microbatch_size:(i + 1) * microbatch_size].to(device)

        optimizer.zero_grad()
        print(f'Inputs: {inputs}')
        print(f'Targets: {target}')

        outputs = model(inputs, start_pos=0)
        print(f'Outputs: {outputs}')

        # Ensure the shapes are correct for cross-entropy loss
        assert outputs.shape[-1] == model_args.vocab_size, f"Expected outputs last dimension to be {model_args.vocab_size}, got {outputs.shape[-1]}"
        assert outputs.shape[:-1] == target.shape, f"Expected outputs shape {outputs.shape[:-1]} to match targets shape {target.shape}"

        loss = torch.nn.functional.cross_entropy(outputs.view(-1, model_args.vocab_size), target.view(-1), ignore_index=tokenizer.pad_id)
        print(f'Loss: {loss.item()}')

        loss.backward()
        total_loss += loss.item()

        optimizer.step()

        # Detach cache tensors
        if hasattr(model, 'layers'):
            for layer in model.layers:
                if hasattr(layer, 'attention'):
                    layer.attention.cache_k = layer.attention.cache_k.detach()
                    layer.attention.cache_v = layer.attention.cache_v.detach()

        print(f"Accumulation step {i+1}/{accumulation_steps} - Loss: {loss.item()}")

    print(f"Training completed for the batch. Total loss: {total_loss / accumulation_steps}")

# Adjust main() to initialize optimizer outside the train_model() function call
def main():
    global model
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    model_adapter.initialize_environment('nccl')

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

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Train indefinitely
    try:
        while True:
            print("Retrieving batch from SOT...")
            batch, targets = get_batch(wallet)
            if batch is not None and targets is not None:
                print("Batch retrieved. Starting training...")
                train_model(model, optimizer, batch, targets, accumulation_steps=4)
            else:
                print("Failed to retrieve a valid batch, skipping this round.")
            time.sleep(1)  # Optional: Add a short delay to avoid hammering the SOT service
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        sot_process.terminate()

if __name__ == "__main__":
    main()
