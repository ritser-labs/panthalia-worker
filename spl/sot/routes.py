import logging
from quart import request, jsonify, Response
from ..auth.api_auth import requires_authentication
from ..plugins.manager import get_plugin
import inspect

def register_routes(app):
    db_adapter = lambda: app.config['db_adapter']
    perm_db_lambda = lambda: app.config['perm_db']
    requires_auth = requires_authentication(db_adapter, perm_db_lambda)

    @app.route('/health', methods=['GET'])
    async def health_check():
        return jsonify({'status': 'healthy'}), 200

    @app.route('/get_batch', methods=['POST'])
    @requires_auth
    async def get_batch():
        plugin = app.config['plugin']
        batch_data = await plugin.call_submodule('sot_adapter', 'get_batch')
        if batch_data is None:
            return jsonify({'error': 'No more batches available'}), 404

        if inspect.isasyncgen(batch_data):
            return Response(batch_data, mimetype='application/octet-stream')
        else:
            return jsonify(batch_data), 200

    @app.route('/update_state', methods=['POST'])
    @requires_auth
    async def update_state():
        plugin = app.config['plugin']
        data = await request.get_json()
        tensor_name = data.get('tensor_name')
        result_url = data.get('result_url')
        version_number = data.get('version_number')
        input_url = data.get('input_url')
        learning_params = {k: v for k, v in data.items() if k not in ['tensor_name', 'result_url', 'version_number', 'input_url']}

        res = await plugin.call_submodule('sot_adapter', 'update_state', tensor_name, result_url, version_number, input_url, learning_params)
        if inspect.isasyncgen(res):
            return Response(res, mimetype='application/octet-stream')
        else:
            return jsonify({'status': 'success'}), 200

    @app.route('/update_loss', methods=['POST'])
    async def update_loss():
        plugin = app.config['plugin']
        data = await request.get_json()
        if not data or 'loss' not in data:
            return jsonify({'error': 'Missing loss value'}), 400
        loss = data['loss']
        version_number = data.get('version_number', 0)
        res = await plugin.call_submodule('sot_adapter', 'update_loss', loss, version_number)
        if inspect.isasyncgen(res):
            return Response(res, mimetype='application/octet-stream')
        else:
            return jsonify({'status': 'success'}), 200

    @app.route('/get_loss', methods=['GET'])
    async def get_loss():
        plugin = app.config['plugin']
        result = await plugin.call_submodule('sot_adapter', 'get_loss')
        if inspect.isasyncgen(result):
            return Response(result, mimetype='application/octet-stream')
        return jsonify(result), 200

    @app.route('/upload_tensor', methods=['POST'])
    async def upload_tensor():
        plugin = app.config['plugin']
        request_files = await request.files
        if 'tensor' not in request_files:
            return jsonify({'error': 'No tensor file provided'}), 400

        request_form = await request.form
        if 'label' not in request_form:
            return jsonify({'error': 'No label provided'}), 400

        tensor_file = request_files['tensor']
        label = request_form['label']

        import torch
        import io
        tensor_bytes = tensor_file.read()
        tensor_state = torch.load(io.BytesIO(tensor_bytes), map_location='cpu')

        res = await plugin.call_submodule('sot_adapter', 'upload_tensor', tensor_state, label)
        if inspect.isasyncgen(res):
            return Response(res, mimetype='application/octet-stream')
        return jsonify({'message': 'Tensor uploaded successfully', 'tensor_url': res}), 200

    @app.route('/data/state/temp/<path:filename>', methods=['GET'])
    async def get_data_file(filename):
        plugin = app.config['plugin']
        chunk_generator = await plugin.call_submodule('sot_adapter', 'stream_data_file', filename)
        if inspect.isasyncgen(chunk_generator):
            return Response(chunk_generator, mimetype='application/octet-stream')
        elif isinstance(chunk_generator, dict) and 'error' in chunk_generator:
            return jsonify(chunk_generator), 404
        else:
            # If it's a normal object, not a generator, treat as binary data
            return Response(chunk_generator, mimetype='application/octet-stream')

    @app.route('/latest_state', methods=['GET'])
    async def latest_state():
        plugin = app.config['plugin']
        tensor_name = request.args.get('tensor_name')
        if not tensor_name:
            return jsonify({'error': 'Missing tensor_name parameter'}), 400

        result = await plugin.call_submodule('sot_adapter', 'get_latest_state', tensor_name)
        if 'error' in result:
            return jsonify(result), 404

        data = result['data']
        version_number = result['version_number']
        # If data is bytes, just return directly
        if inspect.isasyncgen(data):
            return Response(data, mimetype='application/octet-stream')
        response = Response(data, mimetype='application/octet-stream')
        response.headers['X-Version-Number'] = str(version_number)
        return response
