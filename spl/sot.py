import os
import json
import logging
from flask import Flask, request, jsonify, Response
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

@app.route('/', methods=['GET'])
def root():
    return jsonify({'status': 'ok'}), 200

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

@app.route('/upload_gradient', methods=['POST'])
def upload_gradient():
    logging.info("Accessing /upload_gradient endpoint")
    data = request.json
    task_type = data['task_type']
    gradients = data['gradients']
    block_number = data['block_number']

    gradient_file_path = os.path.join(data_dir, f'gradients/{task_type}.json')
    os.makedirs(os.path.dirname(gradient_file_path), exist_ok=True)

    gradient_data = {
        'block_number': block_number,
        'gradients': gradients
    }

    with open(gradient_file_path, 'w') as file:
        json.dump(gradient_data, file)

    return jsonify({'status': 'success'})

@app.route('/latest_state', methods=['GET'])
def latest_state():
    logging.info("Accessing /latest_state endpoint")
    task_type = request.args.get('task_type')
    state_file_path = os.path.join(state_dir, f'{task_type}.pt')

    if os.path.exists(state_file_path):
        state = torch.load(state_file_path)
        state_dict = {
            'state': state.tolist()
        }
        return jsonify(state_dict)
    else:
        return jsonify({'error': 'State not found'}), 404

@app.route('/stream_gradients', methods=['GET'])
def stream_gradients():
    logging.info("Accessing /stream_gradients endpoint")
    gradient_files = os.listdir(os.path.join(data_dir, 'gradients'))

    def generate():
        for gradient_file in gradient_files:
            with open(os.path.join(data_dir, f'gradients/{gradient_file}'), 'r') as file:
                gradients = json.load(file)
                yield f"{json.dumps(gradients)}\n"

    return Response(generate(), content_type='application/json')

@app.route('/stream_specific_tensors', methods=['POST'])
def stream_specific_tensors():
    logging.info("Accessing /stream_specific_tensors endpoint")
    data = request.json
    tensors = data['tensors']

    def generate():
        for tensor_name in tensors:
            tensor_file_path = os.path.join(state_dir, f'{tensor_name}.pt')
            if os.path.exists(tensor_file_path):
                tensor = torch.load(tensor_file_path)
                yield f"{json.dumps({'tensor_name': tensor_name, 'tensor': tensor.tolist()})}\n"

    return Response(generate(), content_type='application/json')

if __name__ == "__main__":
    logging.info("Starting SOT service...")
    app.run(host='0.0.0.0', port=5001)
