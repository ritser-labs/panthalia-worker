import boto3
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

s3 = boto3.client('s3')
bucket_name = 'your-s3-bucket'

@app.route('/latest_model_params', methods=['GET'])
def get_latest_model_params():
    response = s3.get_object(Bucket=bucket_name, Key='latest_model_params.json')
    model_params = json.loads(response['Body'].read().decode('utf-8'))
    return jsonify(model_params)

@app.route('/publish_result', methods=['POST'])
def publish_result():
    data = request.json
    task_id = data['task_id']
    result = data['result']

    key = f'task_results/{task_id}.json'
    s3.put_object(Bucket=bucket_name, Key=key, Body=json.dumps(result))

    return jsonify({'status': 'success'})

@app.route('/stream_gradients', methods=['POST'])
def stream_gradients():
    data = request.json
    task_id = data['task_id']
    gradients = data['gradients']

    key = f'gradients/{task_id}.json'
    s3.put_object(Bucket=bucket_name, Key=key, Body=json.dumps(gradients))

    return jsonify({'status': 'success'})

@app.route('/get_batch', methods=['GET'])
def get_batch():
    response = s3.get_object(Bucket=bucket_name, Key='batch.json')
    batch = json.loads(response['Body'].read().decode('utf-8'))
    return jsonify(batch)

@app.route('/get_targets', methods=['GET'])
def get_targets():
    response = s3.get_object(Bucket=bucket_name, Key='targets.json')
    targets = json.loads(response['Body'].read().decode('utf-8'))
    return jsonify(targets)

@app.route('/update_state', methods=['POST'])
def update_state():
    data = request.json
    task_type = data['task_type']
    result = data['result']

    key = f'state/{task_type}.json'
    s3.put_object(Bucket=bucket_name, Key=key, Body=json.dumps(result))

    return jsonify({'status': 'success'})

@app.route('/update_adam', methods=['POST'])
def update_adam():
    data = request.json
    task_type = data['task_type']
    adam_m = data['adam_m']
    adam_v = data['adam_v']

    key_m = f'adam_m/{task_type}.json'
    key_v = f'adam_v/{task_type}.json'
    s3.put_object(Bucket=bucket_name, Key=key_m, Body=json.dumps(adam_m))
    s3.put_object(Bucket=bucket_name, Key=key_v, Body=json.dumps(adam_v))

    return jsonify({'status': 'success'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
