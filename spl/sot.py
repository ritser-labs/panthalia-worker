import os
import json
import logging
from flask import Flask, request, jsonify
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from common import model_args

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

# Ensure the data directory exists
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Initialize tokenizer
logging.info("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load Wikipedia dataset
logging.info("Loading Wikipedia dataset...")
dataset = load_dataset("wikipedia", "20220301.en", split='train', streaming=True)
dataset_iter = iter(dataset)

# Initialize or load initial state
logging.info("Initializing or loading initial state...")
initial_state = {}
state_dir = os.path.join(data_dir, 'state')
os.makedirs(state_dir, exist_ok=True)

for layer in ["embed", "final_logits", "final_logits_backward", "embed_backward"]:
    state_file_path = os.path.join(state_dir, f'{layer}.json')
    if os.path.exists(state_file_path):
        with open(state_file_path, 'r') as file:
            initial_state[layer] = json.load(file)
    else:
        initial_state[layer] = {"block_number": 0, "state": torch.randn(model_args.max_seq_len, model_args.dim).tolist()}
        with open(state_file_path, 'w') as file:
            json.dump(initial_state[layer], file)

for i in range(model_args.n_layers):
    for layer in [f"forward_layer_{i}", f"backward_layer_{i}"]:
        state_file_path = os.path.join(state_dir, f'{layer}.json')
        if os.path.exists(state_file_path):
            with open(state_file_path, 'r') as file:
                initial_state[layer] = json.load(file)
        else:
            initial_state[layer] = {"block_number": 0, "state": torch.randn(model_args.max_seq_len, model_args.dim).tolist()}
            with open(state_file_path, 'w') as file:
                json.dump(initial_state[layer], file)

@app.route('/latest_model_params', methods=['GET'])
def get_latest_model_params():
    logging.info("Accessing /latest_model_params endpoint")
    with open(os.path.join(data_dir, 'latest_model_params.json'), 'r') as file:
        model_params = json.load(file)
    return jsonify(model_params)

@app.route('/publish_result', methods=['POST'])
def publish_result():
    logging.info("Accessing /publish_result endpoint")
    data = request.json
    task_id = data['task_id']
    result = data['result']

    os.makedirs(os.path.join(data_dir, 'task_results'), exist_ok=True)
    with open(os.path.join(data_dir, f'task_results/{task_id}.json'), 'w') as file:
        json.dump(result, file)

    return jsonify({'status': 'success'})

@app.route('/stream_gradients', methods=['POST'])
def stream_gradients():
    logging.info("Accessing /stream_gradients endpoint")
    data = request.json
    task_id = data['task_id']
    gradients = data['gradients']

    os.makedirs(os.path.join(data_dir, 'gradients'), exist_ok=True)
    with open(os.path.join(data_dir, f'gradients/{task_id}.json'), 'w') as file:
        json.dump(gradients, file)

    return jsonify({'status': 'success'})

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
        return jsonify({"error": "No more data available"}), 404

    batch_url = os.path.join(data_dir, 'batch.json')
    with open(batch_url, 'w') as file:
        json.dump(batch, file)

    return jsonify({'batch_url': batch_url})

@app.route('/get_targets', methods=['GET'])
def get_targets():
    logging.info("Accessing /get_targets endpoint")
    with open(os.path.join(data_dir, 'targets.json'), 'r') as file:
        targets = json.load(file)
    return jsonify(targets)

@app.route('/update_state', methods=['POST'])
def update_state():
    logging.info("Accessing /update_state endpoint")
    data = request.json
    task_type = data['task_type']
    result = data['result']

    os.makedirs(os.path.join(data_dir, 'state'), exist_ok=True)
    with open(os.path.join(data_dir, f'state/{task_type}.json'), 'w') as file:
        json.dump(result, file)

    return jsonify({'status': 'success'})

@app.route('/update_adam', methods=['POST'])
def update_adam():
    logging.info("Accessing /update_adam endpoint")
    data = request.json
    task_type = data['task_type']
    adam_m = data['adam_m']
    adam_v = data['adam_v']

    os.makedirs(os.path.join(data_dir, 'adam_m'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'adam_v'), exist_ok=True)
    with open(os.path.join(data_dir, f'adam_m/{task_type}.json'), 'w') as file_m, \
         open(os.path.join(data_dir, f'adam_v/{task_type}.json'), 'w') as file_v:
        json.dump(adam_m, file_m)
        json.dump(adam_v, file_v)

    return jsonify({'status': 'success'})

@app.route('/latest_state', methods=['GET'])
def latest_state():
    logging.info("Accessing /latest_state endpoint")
    state_files = os.listdir(os.path.join(data_dir, 'state'))
    latest_state = {}
    for state_file in state_files:
        with open(os.path.join(data_dir, f'state/{state_file}'), 'r') as file:
            state = json.load(file)
            task_type = state_file.split('.')[0]
            latest_state[task_type] = state
    return jsonify(latest_state)

if __name__ == "__main__":
    logging.info("Starting SOT service...")
    app.run(host='0.0.0.0', port=5001)
