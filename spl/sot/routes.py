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
        try:
            batch_data = await plugin.call_submodule('sot_adapter', 'get_batch')
            if batch_data is None:
                # No more data available scenario.
                logging.error("No more data available.")
                return jsonify({'error': 'No more data available'}), 404

            if 'input_url' not in batch_data:
                logging.error("Batch data missing 'input_url'.")
                return jsonify({'error': 'Batch missing input_url'}), 500

            return jsonify(batch_data), 200
        except Exception as e:
            logging.error(f"Error getting batch: {e}", exc_info=True)
            # If other error occurs, also return 404 to signal no batch obtained.
            return jsonify({'error': str(e)}), 404

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

        # Check if returned an error dict
        if isinstance(chunk_generator, dict) and 'error' in chunk_generator:
            error_msg = chunk_generator['error']
            logging.error(f"File retrieval error: {error_msg}")
            if error_msg == 'file not found':
                return jsonify({'error': 'File not found'}), 404
            elif error_msg == 'empty_file':
                return jsonify({'error': 'File is empty'}), 500
            else:
                return jsonify({'error': 'Unexpected error'}), 500

        # If it's an async generator, stream it directly
        if inspect.isasyncgen(chunk_generator):
            async def async_gen_wrapper():
                try:
                    async for chunk in chunk_generator:
                        logging.debug(f"SOT route: forwarding {len(chunk)} bytes to worker")
                        yield chunk
                        logging.debug("SOT route: successfully forwarded chunk to worker")
                except Exception as e:
                    logging.error("SOT route: exception while forwarding chunks", exc_info=True)
                    # Raise or just stop yielding. Here we raise to trigger a 500 at the client side.
                    raise

            return Response(async_gen_wrapper(), mimetype='application/octet-stream', status=200)

        # If chunk_generator is bytes
        if isinstance(chunk_generator, (bytes, bytearray)):
            if len(chunk_generator) == 0:
                logging.error("File is empty when returning bytes.")
                return jsonify({'error': 'File is empty'}), 500
            return Response(chunk_generator, mimetype='application/octet-stream', status=200)

        # If none of the above, return an error
        logging.error("unexpected_return_type from get_data_file")
        return jsonify({'error': 'unexpected_return_type'}), 500

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
