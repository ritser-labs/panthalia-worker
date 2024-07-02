from io import BytesIO
import os
import json
import logging
import threading
import requests
from flask import Flask, request, jsonify, send_file, send_from_directory
import torch
from common import model_args, tokenizer
from datasets import load_dataset

app = Flask(__name__)
sync_status = {}
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s', handlers=[
    logging.FileHandler("sot.log"),
    logging.StreamHandler()
])

data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

logging.info("Initializing or loading initial state...")
state_dir = os.path.join(data_dir, 'state')
os.makedirs(state_dir, exist_ok=True)

def initialize_tensor(name, shape, random_init=True):
    file_path = os.path.join(state_dir, f'{name}.pt')
    if os.path.exists(file_path):
        pass
    else:
        if random_init:
            tensor = torch.randn(*shape)
        else:
            tensor = torch.zeros(*shape)
        torch.save(tensor, file_path)

initialize_tensor('embed', (model_args.max_seq_len, model_args.dim))
initialize_tensor('embed_adam_m', (model_args.max_seq_len, model_args.dim), random_init=False)
initialize_tensor('embed_adam_v', (model_args.max_seq_len, model_args.dim), random_init=False)

for i in range(model_args.n_layers):
    initialize_tensor(f'layer_{i}', (model_args.max_seq_len, model_args.dim))
    initialize_tensor(f'layer_{i}_adam_m', (model_args.max_seq_len, model_args.dim), random_init=False)
    initialize_tensor(f'layer_{i}_adam_v', (model_args.max_seq_len, model_args.dim), random_init=False)

initialize_tensor('final_logits', (model_args.max_seq_len, model_args.dim))
initialize_tensor('final_logits_adam_m', (model_args.max_seq_len, model_args.dim), random_init=False)
initialize_tensor('final_logits_adam_v', (model_args.max_seq_len, model_args.dim), random_init=False)

logging.info("Loading Wikipedia dataset...")
dataset = load_dataset("wikipedia", "20220301.en", split='train', streaming=True)
dataset_iter = iter(dataset)

preloaded_batch = None
batch_lock = threading.Lock()

def preload_batch():
    global preloaded_batch, dataset_iter
    batch_size = 2
    max_seq_len = 512
    batch = []

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
            if len(tokens) < max_seq_len:
                tokens += [tokenizer.pad_id] * (max_seq_len - len(tokens))
            elif len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
            batch.append(tokens)
        except StopIteration:
            break

    if batch:
        preloaded_batch = batch
        with open(os.path.join(data_dir, 'batch.json'), 'w') as file:
            json.dump(batch, file)

preload_batch()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

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

@app.route('/publish_result', methods=['POST'])
def publish_result():
    logging.info("Accessing /publish_result endpoint")
    data = request.json
    task_id = data.get('task_id')
    result = data.get('result')

    if not task_id or not result:
        logging.error("Missing task_id or result in /publish_result request")
        return jsonify({'error': 'Missing task_id or result'}), 400

    try:
        os.makedirs(os.path.join(data_dir, 'task_results'), exist_ok=True)
        with open(os.path.join(data_dir, f'task_results/{task_id}.json'), 'w') as file:
            json.dump(result, file)
        return jsonify({'status': 'success'})
    except Exception as e:
        logging.error(f"Error in /publish_result: {e}", exc_info=True)
        return jsonify({'error': 'Could not publish result'}), 500

@app.route('/get_batch', methods=['GET'])
def get_batch():
    logging.info("Accessing /get_batch endpoint")
    global preloaded_batch

    with batch_lock:
        if preloaded_batch is None:
            logging.error("No preloaded batch available")
            return jsonify({"error": "No preloaded batch available"}), 404

        batch = preloaded_batch
        preloaded_batch = None

    try:
        batch_file_path = os.path.join(data_dir, 'batch.json')
        threading.Thread(target=preload_batch).start()
        return jsonify({'batch_url': f'http://localhost:5001/data/{os.path.basename(batch_file_path)}'})
    except Exception as e:
        logging.error(f"Error in /get_batch: {e}", exc_info=True)
        return jsonify({'error': 'Could not get batch'}), 500

@app.route('/get_targets', methods=['GET'])
def get_targets():
    logging.info("Accessing /get_targets endpoint")
    try:
        with open(os.path.join(data_dir, 'targets.json'), 'r') as file:
            targets = json.load(file)
        return jsonify(targets)
    except Exception as e:
        logging.error(f"Error in /get_targets: {e}", exc_info=True)
        return jsonify({'error': 'Could not get targets'}), 500

@app.route('/update_state', methods=['POST'])
def update_state():
    logging.info("Accessing /update_state endpoint")
    data = request.json
    task_type = data.get('task_type')
    result_url = data.get('result_url')
    block_number = data.get('block_number')

    if not task_type or not result_url or block_number is None:
        logging.error("Missing task_type, result_url, or block_number in /update_state request")
        return jsonify({'error': 'Missing task_type, result_url, or block_number'}), 400

    try:
        response = requests.get(result_url)
        if response.status_code != 200:
            raise ValueError(f"Failed to download tensor from {result_url}")

        tensor_data = BytesIO(response.content)
        tensor = torch.load(tensor_data)

        state_file_path = os.path.join(data_dir, f'state/{task_type}.pt')
        if os.path.exists(state_file_path):
            current_tensor = torch.load(state_file_path)
            current_tensor.add_(tensor)
        else:
            current_tensor = tensor

        torch.save(current_tensor, state_file_path)

        # Store the block number along with the state
        block_number_file_path = os.path.join(data_dir, f'state/{task_type}_block_number.json')
        with open(block_number_file_path, 'w') as block_file:
            json.dump({'block_number': block_number}, block_file)

        return jsonify({'status': 'success'})
    except Exception as e:
        logging.error(f"Error in /update_state: {e}", exc_info=True)
        return jsonify({'error': 'Could not update state'}), 500

@app.route('/latest_state', methods=['GET'])
def latest_state():
    logging.info("Accessing /latest_state endpoint")
    tensor_name = request.args.get('tensor_name')
    if not tensor_name:
        return jsonify({'error': 'Missing tensor_name parameter'}), 400

    state_file_path = os.path.join(data_dir, 'state', f'{tensor_name}.pt')
    if not os.path.exists(state_file_path):
        return jsonify({'error': 'Tensor not found'}), 404

    try:
        return send_file(state_file_path, mimetype='application/octet-stream')
    except Exception as e:
        logging.error(f"Error in /latest_state: {e}", exc_info=True)
        return jsonify({'error': 'Could not retrieve latest state'}), 500

@app.route('/stream_gradients', methods=['POST'])
def stream_gradients():
    logging.info("Accessing /stream_gradients endpoint")
    data = request.json
    logging.debug(f"Received data: {data}")

    if not data:
        logging.error("Empty request data in /stream_gradients")
        return jsonify({'error': 'Empty request data'}), 400

    task_id = data.get('task_id')
    gradients = data.get('gradients')
    block_number = data.get('block_number', 0)

    if not task_id or not gradients:
        logging.error("Missing task_id or gradients in /stream_gradients request")
        return jsonify({'error': 'Missing task_id or gradients'}), 400

    try:
        os.makedirs(os.path.join(data_dir, 'gradients'), exist_ok=True)
        gradient_data = {
            'gradients': gradients,
            'block_number': block_number
        }
        with open(os.path.join(data_dir, f'gradients/{task_id}.json'), 'w') as file:
            json.dump(gradient_data, file)
        return jsonify({'status': 'success'})
    except Exception as e:
        logging.error(f"Error in /stream_gradients: {e}", exc_info=True)
        return jsonify({'error': 'Could not stream gradients'}), 500

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

@app.route('/report_stake', methods=['POST'])
def report_stake():
    data = request.json
    worker_address = data.get('worker_address')
    if worker_address:
        sync_status[worker_address] = 'staked'
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'error', 'message': 'Missing worker_address'}), 400

@app.route('/check_stake', methods=['GET'])
def check_stake():
    total_workers = int(request.args.get('total_workers', 0))
    staked_workers = len(sync_status)
    if staked_workers >= total_workers:
        return jsonify({'status': 'all_staked'})
    else:
        return jsonify({'status': 'waiting', 'staked_workers': staked_workers, 'total_workers': total_workers})

@app.route('/data/<path:filename>', methods=['GET'])
def get_data_file(filename):
    logging.info(f"Accessing file: {filename}")
    try:
        return send_from_directory(data_dir, filename)
    except Exception as e:
        logging.error(f"Error accessing file {filename}: {e}", exc_info=True)
        return jsonify({'error': 'File not found'}), 404

if __name__ == "__main__":
    logging.info("Starting SOT service...")
    app.run(host='0.0.0.0', port=5001)
