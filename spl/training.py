import subprocess
import json
import os
import torch
import time
import requests
from model import ModelArgs, Transformer
from common import Model, wait_for_sot, tensor_to_model, initialize_distributed_environment_and_globals, model_args
from device import device
from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3
from io import BytesIO

# Constants
SOT_URL = 'http://localhost:5001'
MASTER_WALLETS_FILE = 'master_wallets.json'
batch_size = 32  # Fixed batch size

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
def train_model(batch, targets, accumulation_steps):
    global model

    # Set model to training mode
    model.train()

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Divide the batch into microbatches for gradient accumulation
    microbatch_size = batch_size // accumulation_steps

    for i in range(accumulation_steps):
        inputs = batch[i * microbatch_size:(i + 1) * microbatch_size].to(device)
        target = targets[i * microbatch_size:(i + 1) * microbatch_size].to(device)

        optimizer.zero_grad()

        outputs = model(inputs, start_pos=0)
        loss = torch.nn.functional.cross_entropy(outputs.view(-1, model_args.vocab_size), target.view(-1))
        loss.backward()

        optimizer.step()
        
        # Detach cache tensors
        if hasattr(model, 'layers'):
            for layer in model.layers:
                if hasattr(layer, 'attention'):
                    layer.attention.cache_k = layer.attention.cache_k.detach()
                    layer.attention.cache_v = layer.attention.cache_v.detach()

        print(f"Accumulation step {i+1}/{accumulation_steps} - Loss: {loss.item()}")

    print("Training completed for the batch.")

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
    if model:
        print("Model loaded successfully.")
    else:
        print("Failed to load the model. Exiting...")
        sot_process.terminate()
        return

    # Train indefinitely
    try:
        while True:
            print("Retrieving batch from SOT...")
            batch, targets = get_batch(wallet)
            if batch is not None and targets is not None:
                print("Batch retrieved. Starting training...")
                train_model(batch, targets, accumulation_steps=4)
            else:
                print("Failed to retrieve a valid batch, skipping this round.")
            time.sleep(1)  # Optional: Add a short delay to avoid hammering the SOT service
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        sot_process.terminate()

if __name__ == "__main__":
    main()
