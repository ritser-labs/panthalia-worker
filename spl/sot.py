import os
import json
import logging
from flask import Flask, request, jsonify
import torch
from common import model_args, tokenizer
from datasets import load_dataset

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s', handlers=[
    logging.FileHandler("sot.log"),
    logging.StreamHandler()
])

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

# Ensure the data directory exists
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Initialize or load initial state
logging.info("Initializing or loading initial state...")
initial_state = {}
state_dir = os.path.join(data_dir, 'state')
os.makedirs(state_dir, exist_ok=True)

for layer in ["embed", "final_logits", "final_logits_backward", "embed_backward"]:
    state_file_path = os.path.join(state_dir, f'{layer}.pt')
    if os.path.exists(state_file_path):
        initial_state[layer] = torch.load(state_file_path)
    else:
        initial_state[layer] = torch.randn(model_args.max_seq_len, model_args.dim)
        torch.save(initial_state[layer], state_file_path)

for i in range(model_args.n_layers):
    for layer in [f"forward_layer_{i}", f"backward_layer_{i}"]:
        state_file_path = os.path.join(state_dir, f'{layer}.pt')
        if os.path.exists(state_file_path):
            initial_state[layer] = torch.load(state_file_path)
        else:
            initial_state[layer] = torch.randn(model_args.max_seq_len, model_args.dim)
            torch.save(initial_state[layer], state_file_path)

logging.info("Loading Wikipedia dataset...")
dataset = load_dataset("wikipedia", "20220301.en", split='train', streaming=True)
dataset_iter = iter(dataset)

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

@app.route('/get_batch', methods=['GET'])
def get_batch():
    logging.info("Accessing /get_batch endpoint")
    batch_size = 2
    max_seq_len = 512

    batch = []
    for _ in range(batch_size):
        try:
            example = next(dataset_iter)
            tokens = tokenizer(example['text'], max_length=max_seq_len, padding='max_length', truncation=True, return_tensors='pt')
            batch.append(tokens['input_ids'][0].tolist())
        except StopIteration:
            break

    if not batch:
        logging.error("No more data available in /get_batch")
        return jsonify({"error": "No more data available"}), 404

    try:
        batch_url = os.path.join(data_dir, 'batch.json')
        with open(batch_url, 'w') as file:
            json.dump(batch, file)
        return jsonify({'batch_url': batch_url})
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
    result = data.get('result')
    block_number = data.get('block_number', 0)

    if not task_type or result is None:
        logging.error("Missing task_type or result in /update_state request")
        return jsonify({'error': 'Missing task_type or result'}), 400

    try:
        os.makedirs(os.path.join(data_dir, 'state'), exist_ok=True)
        state_data = {
            'state': result,
            'block_number': block_number
        }
        state_file_path = os.path.join(data_dir, f'state/{task_type}.json')
        with open(state_file_path, 'w') as file:
            json.dump(state_data, file)
        return jsonify({'status': 'success'})
    except Exception as e:
        logging.error(f"Error in /update_state: {e}", exc_info=True)
        return jsonify({'error': 'Could not update state'}), 500

@app.route('/update_adam', methods=['POST'])
def update_adam():
    logging.info("Accessing /update_adam endpoint")
    data = request.json
    task_type = data.get('task_type')
    adam_m = data.get('adam_m')
    adam_v = data.get('adam_v')

    if not task_type or adam_m is None or adam_v is None:
        logging.error("Missing task_type, adam_m, or adam_v in /update_adam request")
        return jsonify({'error': 'Missing task_type, adam_m, or adam_v'}), 400

    try:
        os.makedirs(os.path.join(data_dir, 'adam_m'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'adam_v'), exist_ok=True)
        torch.save(torch.tensor(adam_m), os.path.join(data_dir, f'adam_m/{task_type}.pt'))
        torch.save(torch.tensor(adam_v), os.path.join(data_dir, f'adam_v/{task_type}.pt'))
        return jsonify({'status': 'success'})
    except Exception as e:
        logging.error(f"Error in /update_adam: {e}", exc_info=True)
        return jsonify({'error': 'Could not update Adam state'}), 500

@app.route('/latest_state', methods=['GET'])
def latest_state():
    logging.info("Accessing /latest_state endpoint")
    try:
        state_files = os.listdir(os.path.join(data_dir, 'state'))
        latest_state = {}
        for state_file in state_files:
            state_file_path = os.path.join(data_dir, 'state', state_file)
            try:
                if state_file.endswith('.json'):
                    with open(state_file_path, 'r') as file:
                        state_data = json.load(file)
                else:
                    state_data = torch.load(state_file_path)
                task_type = state_file.split('.')[0]
                logging.debug(f"Loaded state data type for {state_file}: {type(state_data)}")

                if isinstance(state_data, dict):
                    latest_state[task_type] = {
                        'block_number': state_data['block_number'],
                        'state': state_data['state']
                    }
                elif isinstance(state_data, torch.Tensor):
                    latest_state[task_type] = {
                        'block_number': 0,  # Default block number for tensors
                        'state': state_data.tolist()  # Convert tensor to list for JSON serialization
                    }
                else:
                    logging.error(f"Unsupported data type in state file: {state_file}")
            except Exception as e:
                logging.error(f"Error loading state file {state_file_path}: {e}", exc_info=True)
        return jsonify(latest_state)
    except Exception as e:
        logging.error(f"Error in /latest_state: {e}", exc_info=True)
        return jsonify({'error': 'Could not retrieve latest state'}), 500

if __name__ == "__main__":
    logging.info("Starting SOT service...")
    app.run(host='0.0.0.0', port=5001)
