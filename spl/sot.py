import os
import json
import logging
import threading
from flask import Flask, request, jsonify, send_file, send_from_directory
import torch
from common import model_args, tokenizer, batch_size, initialize_distributed_environment
from datasets import load_dataset
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from device import device
import requests
import time
import random
from werkzeug.utils import secure_filename
from tqdm import tqdm
import torch.nn.init as init
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from model import VocabParallelEmbedding, RMSNorm, ColumnParallelLinear, TransformerBlock

app = Flask(__name__)
sync_status = {}
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s', handlers=[
    logging.StreamHandler()
])

data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

logging.info("Initializing or loading initial state...")
state_dir = os.path.join(data_dir, 'state')
os.makedirs(state_dir, exist_ok=True)

# File to store block numbers
block_numbers_file = os.path.join(state_dir, 'block_numbers.json')

def load_block_numbers():
    if os.path.exists(block_numbers_file):
        with open(block_numbers_file, 'r') as f:
            return json.load(f)
    return {}

def save_block_numbers(block_numbers):
    with open(block_numbers_file, 'w') as f:
        json.dump(block_numbers, f)

# Load existing block numbers on startup
block_numbers = load_block_numbers()

# Dictionary to store gradient updates
executor = ThreadPoolExecutor(max_workers=10)

# Replace hardcoded URL with a variable
BASE_URL = 'http://localhost:5001'

def calculate_transformer_block_size(args):
    head_dim = args.dim // args.n_heads
    n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
    hidden_dim = int(2 * 4 * args.dim / 3)
    hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
    
    attention_size = (
        args.dim * args.n_heads * head_dim +  # wq
        args.dim * n_kv_heads * head_dim +    # wk
        args.dim * n_kv_heads * head_dim +    # wv
        args.dim * args.n_heads * head_dim    # wo
    )
    feedforward_size = (
        args.dim * hidden_dim +  # w1
        hidden_dim * args.dim +  # w2
        args.dim * hidden_dim    # w3
    )
    norm_size = 2 * args.dim  # Two RMSNorm layers

    total_size = attention_size + feedforward_size + norm_size
    return total_size

# Update tensor sizes for each layer
tensor_sizes = {
    'embed': (model_args.vocab_size * model_args.dim,),
    'embed_adam_m': (model_args.vocab_size * model_args.dim,),
    'embed_adam_v': (model_args.vocab_size * model_args.dim,),
    'final_logits': (model_args.dim + model_args.dim * model_args.vocab_size,),  # Add the size of RMSNorm and ColumnParallelLinear combined
}

for i in range(model_args.n_layers):
    block_size = calculate_transformer_block_size(model_args)
    tensor_sizes[f'layer_{i}'] = (block_size,)
    tensor_sizes[f'layer_{i}_adam_m'] = (block_size,)
    tensor_sizes[f'layer_{i}_adam_v'] = (block_size,)

def initialize_distributed_environment_and_globals():
    logging.info("Initializing distributed environment")
    initialize_distributed_environment('gloo')
    initialize_model_parallel(model_parallel_size_=1)
    logging.info("Environment and global variables initialized")

def initialize_tensor(name, random_init=True):
    file_path = os.path.join(state_dir, f'{name}.pt')
    if os.path.exists(file_path):
        return

    if "embed" in name:
        module = VocabParallelEmbedding(model_args.vocab_size, model_args.dim, init_method=lambda x: x).to(device)
    elif "final_logits" in name:
        norm = RMSNorm(model_args.dim, eps=model_args.norm_eps).to(device)
        linear = ColumnParallelLinear(model_args.dim, model_args.vocab_size, bias=False, init_method=lambda x: x).to(device)
        module = [norm, linear]
    elif "layer_" in name:
        layer_idx = int(name.split('_')[1])
        module = TransformerBlock(layer_idx, model_args).to(device)
    else:
        raise ValueError(f"Unsupported tensor name: {name}")

    if random_init and 'layer_' in name:
        for param in module.parameters():
            if param.dim() > 1:
                init.xavier_uniform_(param)

    if isinstance(module, list):  # final_logits case
        tensors = [param.data for submodule in module for param in submodule.parameters()]
        tensor = torch.cat([tensor.view(-1) for tensor in tensors])
    else:
        tensors = [param.data for param in module.parameters()]
        tensor = torch.cat([tensor.view(-1) for tensor in tensors])

    torch.save(tensor, file_path)
    block_numbers[name] = 0
    save_block_numbers(block_numbers)

def initialize_all_tensors():
    # Initialize the embedding tensor
    initialize_tensor('embed')
    initialize_tensor('embed_adam_m', random_init=False)
    initialize_tensor('embed_adam_v', random_init=False)

    # Initialize the layer tensors
    for i in range(model_args.n_layers):
        initialize_tensor(f'layer_{i}')
        initialize_tensor(f'layer_{i}_adam_m', random_init=False)
        initialize_tensor(f'layer_{i}_adam_v', random_init=False)
    
    # Initialize the final logits tensor
    initialize_tensor('final_logits')
    initialize_tensor('final_logits_adam_m', random_init=False)
    initialize_tensor('final_logits_adam_v', random_init=False)

logging.info("Loading Wikipedia dataset...")
dataset = load_dataset("wikipedia", "20220301.en", split='train', streaming=True)
dataset_iter = iter(dataset)

preloaded_batch = None

def truncate_tokens(tokens, max_seq_len, pad_token=tokenizer.pad_id):
    if len(tokens) < max_seq_len:
        tokens += [pad_token] * (max_seq_len - len(tokens))
    elif len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]
    return tokens

def preload_batch():
    global preloaded_batch, dataset_iter
    max_seq_len = 512
    batch = []
    targets = []

    for _ in range(batch_size):
        try:
            example = next(dataset_iter)
            tokens = tokenizer.encode(
                example['text'], 
                bos=False, 
                eos=False, 
                allowed_special=set(), 
                disallowed_special=(), 
            )

            # Make a copy of tokens for target creation
            inputs = truncate_tokens(tokens, max_seq_len)

            batch.append(inputs)

            # Create targets from tokens
            target_tokens = truncate_tokens(tokens[1:], max_seq_len, tokenizer.eos_id)

            targets.append(target_tokens)
        except StopIteration:
            break

    if batch:
        preloaded_batch = batch
        timestamp = int(time.time())
        random_suffix = random.randint(1000, 9999)
        batch_filename = f'batch_{timestamp}_{random_suffix}.json'
        targets_filename = f'targets_{timestamp}_{random_suffix}.json'

        with open(os.path.join(state_dir, batch_filename), 'w') as file:
            file.write(json.dumps(batch))
        with open(os.path.join(state_dir, targets_filename), 'w') as file:
            file.write(json.dumps(targets))
        
        return batch_filename, targets_filename

def clean_up_state_directory():
    logging.info("Cleaning up state directory...")
    for filename in os.listdir(state_dir):
        if filename.startswith("batch_") or filename.startswith("targets_"):
            file_path = os.path.join(state_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    logging.info(f"Deleted {file_path}")
            except Exception as e:
                logging.error(f"Error deleting file {file_path}: {e}")

def initialize_service():
    logging.info("Initializing distributed environment and tensors")
    initialize_distributed_environment_and_globals()
    initialize_all_tensors()
    clean_up_state_directory()
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

@app.route('/latest_model_params', methods=['GET'])
def get_latest_model_params():
    logging.info("Accessing /latest_model_params endpoint")
    try:
        with open(os.path.join(data_dir, 'latest_model_params.json'), 'r') as file:
            model_params = json.load(file)
        return jsonify(model_params)
    except Exception as e:
        logging.error(f"Error accessing /latest_model_params: {e}", exc_info=True)
        return jsonify({'error': 'Could not load model parameters'}), 500

@app.route('/get_batch', methods=['POST'])
def get_batch():
    logging.info("Accessing /get_batch endpoint")
    global preloaded_batch

    if preloaded_batch is None:
        logging.error("No preloaded batch available")
        return jsonify({"error": "No preloaded batch available"}), 404

    batch_filename, targets_filename = preload_batch()
    preloaded_batch = None

    try:
        return jsonify({
            'batch_url': f'{BASE_URL}/data/state/{batch_filename}',
            'targets_url': f'{BASE_URL}/data/state/{targets_filename}'
        })
    except Exception as e:
        logging.error(f"Error in /get_batch: {e}", exc_info=True)
        return jsonify({'error': 'Could not get batch'}), 500

@app.route('/update_state', methods=['POST'])
def update_state():
    logging.info("Accessing /update_state endpoint")
    data = request.get_json()
    tensor_name = data.get('tensor_name')
    result_url = data.get('result_url')
    block_number = data.get('block_number')

    logging.debug(f"Received tensor_name: {tensor_name}, result_url: {result_url}, block_number: {block_number}")

    if not tensor_name or not result_url or block_number is None:
        logging.error("Missing tensor_name, result_url, or block_number in /update_state request")
        return jsonify({'error': 'Missing tensor_name, result_url, or block_number'}), 400

    try:
        with requests.Session() as session:
            tensor_data = fetch(session, result_url)

        tensor = torch.load(BytesIO(tensor_data), map_location=device)  # Load tensor to the correct device
        state_file_path = os.path.join(state_dir, f'{tensor_name}.pt')
        
        if os.path.exists(state_file_path):
            current_tensor = torch.load(state_file_path, map_location=device)  # Load existing tensor to the correct device
            updated_tensor = current_tensor + tensor  # Perform addition without in-place operation
        else:
            updated_tensor = tensor

        torch.save(updated_tensor, state_file_path)

        block_numbers[tensor_name] = block_number
        save_block_numbers(block_numbers)

        logging.debug(f"Updated state for {tensor_name}")
        return jsonify({'status': 'success'})
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

    state_file_path = os.path.join(state_dir, f'{tensor_name}.pt')
    if not os.path.exists(state_file_path):
        return jsonify({'error': 'Tensor not found'}), 404

    try:
        response = send_file(state_file_path, mimetype='application/octet-stream')
        response.headers['block_number'] = block_numbers.get(tensor_name, 0)
        return response
    except Exception as e:
        logging.error(f"Error in /latest_state: {e}", exc_info=True)
        return jsonify({'error': 'Could not retrieve latest state'}), 500

@app.route('/tensor_block_number', methods=['GET'])
def tensor_block_number():
    logging.info("Accessing /tensor_block_number endpoint")
    tensor_name = request.args.get('tensor_name')
    if not tensor_name:
        return jsonify({'error': 'Missing tensor_name parameter'}), 400

    block_number = block_numbers.get(tensor_name, 0)
    return jsonify({'block_number': block_number})

@app.route('/tensor_size', methods=['GET'])
def get_tensor_size():
    logging.info("Accessing /tensor_size endpoint")
    tensor_name = request.args.get('tensor_name')
    if not tensor_name:
        return jsonify({'error': 'Missing tensor_name parameter'}), 400

    state_file_path = os.path.join(data_dir, f'state/{tensor_name}.pt')
    if not os.path.exists(state_file_path):
        return jsonify({'error': 'Tensor not found'}), 404

    tensor = torch.load(state_file_path)
    size = tensor.numel()
    return jsonify({'size': size})

@app.route('/data/<path:filename>', methods=['GET'])
def get_data_file(filename):
    logging.info(f"Accessing file: {filename}")
    try:
        return send_from_directory(data_dir, filename)
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
    timestamp = int(time.time())
    random_suffix = random.randint(1000, 9999)
    filename = secure_filename(f'{label}_{timestamp}_{random_suffix}.pt')
    local_file_path = os.path.join(data_dir, filename)

    logging.debug("Receiving tensor upload...")
    total_size = request.content_length
    chunk_size = 1024 * 1024  # 1MB

    with open(local_file_path, 'wb') as f:
        for chunk in tqdm(tensor_file.stream, total=total_size // chunk_size, unit='MB', desc='Receiving'):
            if chunk:
                f.write(chunk)
    logging.debug("Tensor upload completed.")

    tensor_name = filename.split('.')[0]

    try:
        block_number = int(filename.split('_')[2])
    except (IndexError, ValueError):
        block_number = 0

    # Save the tensor state
    tensor_state = torch.load(local_file_path)
    torch.save(tensor_state, os.path.join(state_dir, filename))  # Use filename directly

    # Update block number
    block_numbers[tensor_name] = block_number
    save_block_numbers(block_numbers)

    logging.debug(f"Tensor {tensor_name} uploaded and saved with block number {block_number}")

    return jsonify({'message': 'Tensor uploaded successfully', 'tensor_url': f'{BASE_URL}/data/{filename}'}), 200


if __name__ == "__main__":
    logging.info("Starting SOT service...")
    initialize_service()
    app.run(host='0.0.0.0', port=5001, debug=True)
