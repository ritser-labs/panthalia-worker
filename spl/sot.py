import os
import json
from flask import Flask, request, jsonify
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from common import model_args

app = Flask(__name__)

# Ensure the data directory exists
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load Wikipedia dataset
dataset = load_dataset("wikipedia", "20220301.en", split='train', streaming=True)
dataset_iter = iter(dataset)

# Initialize random state based on model_args
initial_state = {
    "embed": {"block_number": 0, "state": torch.randn(model_args.max_seq_len, model_args.dim).tolist()}
}
for i in range(model_args.n_layers):
    initial_state[f"forward_layer_{i}"] = {"block_number": 0, "state": torch.randn(model_args.max_seq_len, model_args.dim).tolist()}
    initial_state[f"backward_layer_{i}"] = {"block_number": 0, "state": torch.randn(model_args.max_seq_len, model_args.dim).tolist()}
initial_state["final_logits"] = {"block_number": 0, "state": torch.randn(model_args.max_seq_len, model_args.vocab_size).tolist()}
initial_state["final_logits_backward"] = {"block_number": 0, "state": torch.randn(model_args.max_seq_len, model_args.dim).tolist()}
initial_state["embed_backward"] = {"block_number": 0, "state": torch.randn(model_args.max_seq_len, model_args.dim).tolist()}

# Save the initial state to individual layer files if they do not exist
os.makedirs(os.path.join(data_dir, 'state'), exist_ok=True)
for layer, state in initial_state.items():
    state_file_path = os.path.join(data_dir, f'state/{layer}.json')
    if not os.path.exists(state_file_path):
        with open(state_file_path, 'w') as file:
            json.dump(state, file)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/latest_model_params', methods=['GET'])
def get_latest_model_params():
    with open(os.path.join(data_dir, 'latest_model_params.json'), 'r') as file:
        model_params = json.load(file)
    return jsonify(model_params)


@app.route('/publish_result', methods=['POST'])
def publish_result():
    data = request.json
    task_id = data['task_id']
    result = data['result']

    os.makedirs(os.path.join(data_dir, 'task_results'), exist_ok=True)
    with open(os.path.join(data_dir, f'task_results/{task_id}.json'), 'w') as file:
        json.dump(result, file)

    return jsonify({'status': 'success'})


@app.route('/stream_gradients', methods=['POST'])
def stream_gradients():
    data = request.json
    task_id = data['task_id']
    gradients = data['gradients']

    os.makedirs(os.path.join(data_dir, 'gradients'), exist_ok=True)
    with open(os.path.join(data_dir, f'gradients/{task_id}.json'), 'w') as file:
        json.dump(gradients, file)

    return jsonify({'status': 'success'})


@app.route('/get_batch', methods=['GET'])
def get_batch():
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
    with open(os.path.join(data_dir, 'targets.json'), 'r') as file:
        targets = json.load(file)
    return jsonify(targets)


@app.route('/update_state', methods=['POST'])
def update_state():
    data = request.json
    task_type = data['task_type']
    result = data['result']

    os.makedirs(os.path.join(data_dir, 'state'), exist_ok=True)
    with open(os.path.join(data_dir, f'state/{task_type}.json'), 'w') as file:
        json.dump(result, file)

    return jsonify({'status': 'success'})


@app.route('/update_adam', methods=['POST'])
def update_adam():
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
    state_files = os.listdir(os.path.join(data_dir, 'state'))
    latest_state = {}
    for state_file in state_files:
        with open(os.path.join(data_dir, f'state/{state_file}'), 'r') as file:
            state = json.load(file)
            task_type = state_file.split('.')[0]
            latest_state[task_type] = state
    return jsonify(latest_state)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
