from collections import defaultdict
import os
import json
import logging
import threading
from flask import Flask, request, jsonify, send_file, send_from_directory
import torch
from .common import model_config, model_adapter, batch_size, TENSOR_VERSION_INTERVAL, TENSOR_NAME, dataset, SOT_PRIVATE_PORT
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from .device import device
import requests
import time
import random
from werkzeug.utils import secure_filename
from tqdm import tqdm
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3
import functools

app = Flask(__name__)
sync_status = {}
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', handlers=[
    logging.StreamHandler()
])
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

logging.info("Initializing or loading initial state...")
state_dir = os.path.join(data_dir, 'state')
os.makedirs(state_dir, exist_ok=True)

# Create the temp directory within state_dir
temp_dir = os.path.join(state_dir, 'temp')
os.makedirs(temp_dir, exist_ok=True)

# File paths to store block timestamps, num_updates, and last_future_version_number
block_timestamps_file = os.path.join(state_dir, 'block_timestamps.json')
num_updates_file = os.path.join(state_dir, 'num_updates.json')
last_future_version_file = os.path.join(state_dir, 'last_future_version_number.json')

# Initialize locks for these dicts
block_timestamps_lock = threading.Lock()
num_updates_lock = threading.Lock()
last_future_version_lock = threading.Lock()

def load_json(file_path, default):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        save_json(file_path, default)
        return default

def save_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f)

# Load existing block timestamps, num_updates, and last_future_version_number on startup
with block_timestamps_lock:
    block_timestamps = load_json(block_timestamps_file, {})
with num_updates_lock:
    num_updates = load_json(num_updates_file, {})
with last_future_version_lock:
    last_future_version_number = load_json(last_future_version_file, {})

# Dictionary to store gradient updates
executor = ThreadPoolExecutor(max_workers=10)

# Global variable to store master's public keys
master_public_keys = []

# Dictionary to store used nonces
used_nonces = {}

latest_loss = {'value': None}
latest_loss_lock = threading.Lock()

synced_workers = 0

def initialize_tensor(name, sync_version_number=None, zero_init=False):
    if sync_version_number is None:
        sync_version_number = block_timestamps.get(
            0, int(time.time()) // TENSOR_VERSION_INTERVAL * TENSOR_VERSION_INTERVAL)
    
    file_path = os.path.join(state_dir, f'{name}_{sync_version_number}.pt')
    if os.path.exists(file_path):
        return

    if TENSOR_NAME not in name:
        raise ValueError(f"Unsupported tensor name: {name}")
    
    tensor = model_adapter.init_tensor(zero_init)

    torch.save(tensor, file_path)
    with block_timestamps_lock:
        block_timestamps[name] = sync_version_number
    save_json(block_timestamps_file, block_timestamps)
    with last_future_version_lock:
        last_future_version_number[name] = sync_version_number
    save_json(last_future_version_file, last_future_version_number)

def initialize_all_tensors():
    initialize_tensor(TENSOR_NAME, zero_init=False)
    initialize_tensor(f'{TENSOR_NAME}_adam_m', zero_init=True)
    initialize_tensor(f'{TENSOR_NAME}_adam_v', zero_init=True)

preloaded_batch = None
preloaded_batch_lock = threading.Lock()
preloaded_batch_condition = threading.Condition(lock=preloaded_batch_lock)

def preload_batch():
    global preloaded_batch
    batch = []
    targets = []

    for inputs, target_tokens in dataset:
        if len(batch) >= batch_size:
            break
        # Convert each list to a tensor, adding a new dimension
        if isinstance(inputs, list):
            inputs = torch.tensor(inputs)  # Convert list to tensor
        if isinstance(target_tokens, list):
            target_tokens = torch.tensor(target_tokens)  # Convert list to tensor
        batch.append(inputs)
        targets.append(target_tokens)

    if batch:
        # Stack the tensors along a new dimension
        batch_tensor = torch.stack(batch)
        targets_tensor = torch.stack(targets)

        timestamp = int(time.time())
        random_suffix = random.randint(1000, 9999)
        batch_filename = f'batch_{timestamp}_{random_suffix}.pt'
        targets_filename = f'targets_{timestamp}_{random_suffix}.pt'

        # Save the stacked tensors
        torch.save(batch_tensor, os.path.join(temp_dir, batch_filename))
        torch.save(targets_tensor, os.path.join(temp_dir, targets_filename))

        # Set the global preloaded_batch variable here
        with preloaded_batch_lock:
            preloaded_batch = (batch_filename, targets_filename)
            preloaded_batch_condition.notify_all()

        return batch_filename, targets_filename


def initialize_service():
    logging.info("Initializing distributed environment and tensors")
    model_adapter.initialize_environment('gloo')
    initialize_all_tensors()
    preload_batch()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

def fetch(session, url):
    try:
        with session.get(url, timeout=10) as response:
            response.raise_for_status()
            return response.content
    except requests.RequestException as e:
        logging.error(f"Error fetching {url}: {e}")
        raise

def verify_signature(message, signature):
    message = encode_defunct(text=message)
    recovered_address = Account.recover_message(message, signature=signature)
    logging.debug(f"Recovered address: {recovered_address}, Expected addresses: {master_public_keys}")
    return recovered_address.lower() in [key.lower() for key in master_public_keys]

def requires_authentication(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        logging.debug(f"Authorization header: {auth_header}")
        if not auth_header:
            logging.error("Authorization header missing")
            return jsonify({'error': 'Authorization header missing'}), 401

        try:
            message, signature = auth_header.rsplit(':', 1)
        except ValueError:
            logging.error("Invalid Authorization header format")
            return jsonify({'error': 'Invalid Authorization header format'}), 401

        if not verify_signature(message, signature):
            logging.error("Invalid signature")
            return jsonify({'error': 'Invalid signature'}), 403

        # Parse the message to extract the nonce and timestamp
        try:
            message_data = json.loads(message)
            nonce = message_data['nonce']
            timestamp = message_data['timestamp']
            logging.debug(f"Message nonce: {nonce}, timestamp: {timestamp}")
        except (KeyError, json.JSONDecodeError):
            logging.error("Invalid message format")
            return jsonify({'error': 'Invalid message format'}), 401

        # Check if the nonce has been used before
        if nonce in used_nonces:
            logging.error("Nonce already used")
            return jsonify({'error': 'Nonce already used'}), 403

        # Check if the message has expired (validity period of 5 minutes)
        current_time = int(time.time())
        if current_time - timestamp > 300:
            logging.error("Message expired")
            return jsonify({'error': 'Message expired'}), 403

        # Store the nonce to prevent reuse
        used_nonces[nonce] = True

        return f(*args, **kwargs)
    return decorated_function

@app.route('/report_sync', methods=['POST'])
def report_sync():
    global synced_workers
    synced_workers += 1
    return jsonify({'status': 'ok'})

@app.route('/get_num_synced', methods=['GET'])
def get_num_synced():
    global synced_workers
    return jsonify(synced_workers)

@app.route('/get_batch', methods=['POST'])
@requires_authentication
def get_batch():
    logging.info("Accessing /get_batch endpoint")
    global preloaded_batch

    with preloaded_batch_condition:
        while preloaded_batch is None:
            logging.info("Waiting for batch to be preloaded...")
            preloaded_batch_condition.wait()

        batch_filename, targets_filename = preloaded_batch
        preloaded_batch = None

    preload_batch()

    try:
        return jsonify({
            'batch_url': f'/data/state/temp/{batch_filename}',
            'targets_url': f'/data/state/temp/{targets_filename}'
        })
    except Exception as e:
        logging.error(f"Error in /get_batch: {e}", exc_info=True)
        return jsonify({'error': 'Could not get batch'}), 500

def stable_adamw_update(params, grads, m, v, lr=0.002, weight_decay=0.2, beta1=0.9, beta2=0.99, eps=1e-6, clip_thresh=1.0, step=1):
    logging.debug(f"Params before update: {params}")
    logging.debug(f"Grads: {grads}")
    logging.debug(f"m before update: {m}")
    logging.debug(f"v before update: {v}")

    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * grads ** 2
    
    logging.debug(f"Updated m: {m}")
    logging.debug(f"Updated v: {v}")
    
    m_hat = m / (1 - beta1 ** step)
    v_hat = v / (1 - beta2 ** step)
    
    denominator = torch.sqrt(v_hat) + eps
    
    rms = torch.sqrt(torch.mean(grads * grads / torch.max(v, (eps * eps) * torch.ones_like(v))))
    
    new_lr = lr * (1. / max(1., rms / clip_thresh))
    
    params = params * (1.0 - new_lr * weight_decay) - new_lr * m_hat / denominator
    
    logging.debug(f"m_hat: {m_hat}")
    logging.debug(f"v_hat: {v_hat}")
    logging.debug(f"denominator: {denominator}")
    logging.debug(f"rms: {rms}")
    logging.debug(f"new_lr: {new_lr}")
    logging.debug(f"Updated params: {params}")

    return params, m, v


def apply_adamw(version_number, tensor_name, grads_flat, learning_rate, beta1, beta2, epsilon, weight_decay, t, clip_grad=1.0):
    tensor_path = os.path.join(state_dir, f'{tensor_name}_{version_number}.pt')
    if not os.path.exists(tensor_path):
        raise FileNotFoundError(f"Tensor file for {tensor_name} not found at {tensor_path}")
    
    tensor = torch.load(tensor_path, map_location=device)

    if tensor is None:
        raise ValueError(f"Failed to load tensor for {tensor_name}")

    # Ensure tensor is on the correct device and convert to flat tensor if necessary
    tensor = tensor.to(device)

    logging.debug(f"Tensor before AdamW: {tensor}")

    logging.debug(f"Flattened gradients: {grads_flat}")


    if torch.isnan(grads_flat).any() or torch.isinf(grads_flat).any():
        logging.error(f"NaNs or Infs detected in gradients before AdamW update for {tensor_name}")
        raise ValueError(f"NaNs or Infs detected in gradients for {tensor_name}")

    tensor_adam_m_path = os.path.join(state_dir, f'{tensor_name}_adam_m_{version_number}.pt')
    tensor_adam_v_path = os.path.join(state_dir, f'{tensor_name}_adam_v_{version_number}.pt')

    adam_m, adam_v = None, None

    if os.path.exists(tensor_adam_m_path):
        adam_m = torch.load(tensor_adam_m_path, map_location=device).to(device)
    if os.path.exists(tensor_adam_v_path):
        adam_v = torch.load(tensor_adam_v_path, map_location=device).to(device)

    if adam_m is None or adam_v is None:
        logging.debug(f'adam_m or adam_v not found for {tensor_name}, initializing to zeros')
        adam_m = torch.zeros_like(tensor, device=device)
        adam_v = torch.zeros_like(tensor, device=device)

    logging.debug(f"m before AdamW: {adam_m}")
    logging.debug(f"v before AdamW: {adam_v}")
    
    clip_threshold = 1.0

    # Get the StableAdamW updates
    param_update, m_update, v_update = stable_adamw_update(
        tensor,
        grads_flat,
        adam_m,
        adam_v,
        learning_rate,
        weight_decay,
        beta1,
        beta2,
        epsilon,
        clip_threshold,
        t
    )

    logging.debug(f"Updates after applying StableAdamW: {param_update}")
    logging.debug(f"m after AdamW: {m_update}")
    logging.debug(f"v after AdamW: {v_update}")

    return param_update.view(-1), m_update.view(-1), v_update.view(-1)

@app.route('/update_state', methods=['POST'])
@requires_authentication
def update_state():
    logging.info("Accessing /update_state endpoint")
    data = request.get_json()
    tensor_name = data.get('tensor_name')
    result_url = data.get('result_url')

    logging.debug(f"Received tensor_name: {tensor_name}, result_url: {result_url}")

    if not tensor_name or not result_url:
        logging.error("Missing tensor_name or result_url in /update_state request")
        return jsonify({'error': 'Missing tensor_name or result_url'}), 400

    future_version_number = (int(time.time()) // TENSOR_VERSION_INTERVAL + 1) * TENSOR_VERSION_INTERVAL

    if last_future_version_number.get(tensor_name, 0) < future_version_number:
        if last_future_version_number.get(tensor_name, 0) > block_timestamps.get(tensor_name, 0):
            with block_timestamps_lock:
                block_timestamps[tensor_name] = last_future_version_number.get(tensor_name, 0)
            save_json(block_timestamps_file, block_timestamps)
    with num_updates_lock:
        num_updates[tensor_name] = 0
        save_json(num_updates_file, num_updates)
    logging.info(f"Future version number for {tensor_name}: {future_version_number}")

    try:
        with requests.Session() as session:
            tensor_data = fetch(session, result_url)

        tensor = torch.load(BytesIO(tensor_data), map_location=device)  # Load tensor to the correct device

        # Paths for accumulated grads and future tensor
        accumulated_grads_path = os.path.join(state_dir, f'accumulated_grads_{tensor_name}_{future_version_number}.pt')
        future_tensor_path = os.path.join(state_dir, f'{tensor_name}_{future_version_number}.pt')
        unversioned_tensor_path = os.path.join(state_dir, f'{tensor_name}.pt')
        future_tensor_adam_m_path = os.path.join(state_dir, f'{tensor_name}_adam_m_{future_version_number}.pt')
        future_tensor_adam_v_path = os.path.join(state_dir, f'{tensor_name}_adam_v_{future_version_number}.pt')

        # Load or initialize the accumulated_grads tensor
        if os.path.exists(accumulated_grads_path):
            accumulated_grads = torch.load(accumulated_grads_path, map_location=device).to(device)
        else:
            accumulated_grads = torch.zeros_like(tensor, device=device)

        # Update the accumulated_grads tensor
        accumulated_grads += tensor.to(device)
        torch.save(accumulated_grads, accumulated_grads_path)

        # Calculate the future tensor
        current_version_number = block_timestamps.get(tensor_name, 0)
        logging.info(f'Updating state for {tensor_name}, future version number: {future_version_number}, current version number: {current_version_number}')

        with num_updates_lock:
            num_of_updates = num_updates[tensor_name] + 1
            num_updates[tensor_name] = num_of_updates
            save_json(num_updates_file, num_updates)
            
        averaged_grads = (accumulated_grads / num_of_updates).to(device)
        future_tensor, m_update, v_update = apply_adamw(
            current_version_number,
            tensor_name,
            averaged_grads,
            data.get('learning_rate', 0.002) * num_of_updates,
            data.get('beta1', 0.9),
            data.get('beta2', 0.99),
            data.get('epsilon', 1e-6),
            data.get('weight_decay', 0.2),
            data.get('t', 1)
        )

        torch.save(future_tensor, future_tensor_path)
        torch.save(future_tensor, unversioned_tensor_path)
        torch.save(m_update, future_tensor_adam_m_path)
        torch.save(v_update, future_tensor_adam_v_path)
        
        if last_future_version_number.get(tensor_name, 0) < future_version_number:
            with last_future_version_lock:
                last_future_version_number[tensor_name] = future_version_number
                save_json(last_future_version_file, last_future_version_number)

        # Cleanup old accumulated grads tensors
        for filename in os.listdir(state_dir):
            if filename.startswith(f'accumulated_grads_{tensor_name}_') and not filename.endswith(f'{future_version_number}.pt'):
                os.remove(os.path.join(state_dir, filename))

        logging.debug(f"Updated state for {tensor_name}")
        return jsonify({'status': 'success', 'version_number': future_version_number})
    except requests.RequestException as e:
        logging.error(f"Failed to update tensor {tensor_name} due to request exception: {e}")
    except Exception as e:
        logging.error(f"Failed to update tensor {tensor_name} due to error: {e}")
    return jsonify({'error': 'Could not update state'}), 500


@app.route('/latest_state', methods=['GET'])
def latest_state():
    logging.info("Accessing /latest_state endpoint")
    tensor_name = request.args.get('tensor_name')
    if not tensor_name:
        return jsonify({'error': 'Missing tensor_name parameter'}), 400

    latest_version_number = request.args.get('version_number')
    if latest_version_number is None:
        with block_timestamps_lock:
            latest_version_number = block_timestamps.get(tensor_name, 0)
    else:
        latest_version_number = int(latest_version_number)

    # Directory where tensor files are stored
    tensor_files = [f for f in os.listdir(state_dir) if f.startswith(tensor_name)]

    # Extract version numbers from file names
    version_numbers = []
    for file in tensor_files:
        if file.startswith(tensor_name):
            parts = file.rsplit('_', 1)
            if len(parts) == 2 and parts[0] == tensor_name:
                try:
                    version = int(parts[1].split('.')[0])
                    if version <= latest_version_number:
                        version_numbers.append(version)
                except ValueError:
                    continue

    if not version_numbers:
        return jsonify({'error': 'Tensor not found'}), 404

    # Get the latest available version number
    latest_available_version_number = max(version_numbers)
    state_file_path = os.path.join(state_dir, f'{tensor_name}_{latest_available_version_number}.pt')

    if not os.path.exists(state_file_path):
        return jsonify({'error': 'Tensor not found'}), 404

    try:
        response = send_file(state_file_path, mimetype='application/octet-stream')
        response.headers['version_number'] = latest_available_version_number
        return response
    except Exception as e:
        logging.error(f"Error in /latest_state: {e}", exc_info=True)
        return jsonify({'error': 'Could not retrieve latest state'}), 500

@app.route('/tensor_block_timestamp', methods=['GET'])
def tensor_block_timestamp():
    logging.info("Accessing /tensor_block_timestamp endpoint")
    tensor_name = request.args.get('tensor_name')
    if not tensor_name:
        return jsonify({'error': 'Missing tensor_name parameter'}), 400

    with block_timestamps_lock:
        latest_version_number = block_timestamps.get(tensor_name, 0)
    return jsonify({'version_number': latest_version_number})

@app.route('/tensor_size', methods=['GET'])
def get_tensor_size():
    logging.info("Accessing /tensor_size endpoint")
    tensor_name = request.args.get('tensor_name')
    if not tensor_name:
        return jsonify({'error': 'Missing tensor_name parameter'}), 400

    state_file_path = os.path.join(data_dir, f'state/{tensor_name}.pt')
    if not os.path.exists(state_file_path):
        return jsonify({'error': 'Tensor not found'}), 404

    tensor = torch.load(state_file_path, map_location=device)
    size = tensor.numel()
    return jsonify({'size': size})

@app.route('/data/state/<path:filename>', methods=['GET'])
def get_data_file(filename):
    logging.info(f"Accessing file: {filename}")
    try:
        return send_from_directory(state_dir, filename)
    except Exception as e:
        logging.error(f"Error accessing file {filename}: {e}", exc_info=True)
        return jsonify({'error': 'File not found'}), 404

@app.route('/upload_tensor', methods=['POST'])
def upload_tensor():
    if 'tensor' not in request.files:
        return jsonify({'error': 'No tensor file provided'}), 400

    if 'label' not in request.form:
        return jsonify({'error': 'No label provided'}), 400

    tensor_file = request.files['tensor']
    label = request.form['label']
    update_version_number = int(time.time())
    random_suffix = random.randint(1000, 9999)
    filename = secure_filename(f'{label}_{update_version_number}_{random_suffix}.pt')
    local_file_path = os.path.join(temp_dir, filename)

    logging.debug("Receiving tensor upload...")
    total_size = request.content_length
    chunk_size = 1024 * 1024  # 1MB

    with open(local_file_path, 'wb') as f:
        for chunk in tqdm(tensor_file.stream, total=total_size // chunk_size, unit='MB', desc='Receiving'):
            if chunk:
                f.write(chunk)
    logging.debug("Tensor upload completed.")

    tensor_name = filename.split('.')[0]

    # Save the tensor state
    tensor_state = torch.load(local_file_path, map_location=device)
    torch.save(tensor_state, os.path.join(temp_dir, filename))  # Use filename directly

    logging.debug(f"Tensor {tensor_name} uploaded and saved with version_number {update_version_number}")

    return jsonify({'message': 'Tensor uploaded successfully', 'tensor_url': f'/data/state/temp/{filename}'}), 200

@app.route('/update_loss', methods=['POST'])
def update_loss():
    global latest_loss

    # Get the loss value from the request data
    data = request.get_json()
    if not data or 'loss' not in data:
        return jsonify({'error': 'Missing loss value'}), 400

    loss = data['loss']
    
    # Update the loss value with thread-safety
    with latest_loss_lock:
        latest_loss['value'] = loss

    logging.info(f"Updated latest loss: {loss}")
    return jsonify({'status': 'success'}), 200

@app.route('/get_loss', methods=['GET'])
def get_loss():
    global latest_loss
    with latest_loss_lock:
        loss = latest_loss['value']
    return jsonify({'loss': loss}), 200


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Source of Truth (SOT) Service")
    parser.add_argument('--public_keys_file', type=str, required=True, help="Path to the file containing public keys of the master for verifying requests")

    args = parser.parse_args()

    with open(args.public_keys_file, 'r') as f:
        master_public_keys = json.load(f)

    logging.info("Starting SOT service...")
    initialize_service()

    # Check if the script is run directly or if it's managed by a WSGI server like Gunicorn
    if 'gunicorn' not in os.environ.get('SERVER_SOFTWARE', ''):
        # If not running under Gunicorn, use Flask's built-in server
        logging.info("Running in development mode with Flask's built-in server...")
        app.run(host='0.0.0.0', port=SOT_PRIVATE_PORT, debug=True)
