import os
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# Ensure the data directory exists
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

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
    with open(os.path.join(data_dir, 'batch.json'), 'r') as file:
        batch = json.load(file)
    return jsonify(batch)

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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
