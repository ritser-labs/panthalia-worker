# spl/plugins/plugin_server.py
import asyncio
import importlib
import sys
import os
import json
import traceback
import logging
import signal
from functools import partial

from quart import Quart, jsonify, request, Response
from .serialize import serialize_data, deserialize_data

PLUGIN_DIR = os.environ.get('DOCKER_PLUGIN_DIR', '/app/plugin_code')
if PLUGIN_DIR not in sys.path:
    sys.path.append(PLUGIN_DIR)

parent_dir = os.path.dirname(PLUGIN_DIR)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

current_package = __package__
exported_plugin = None

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)

app = Quart(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024 * 1024  # 1 TB
app.config['BODY_TIMEOUT'] = 60 * 60 * 24  # 1 day

def _handle_termination(signum, frame):
    logger.info("[plugin_server] Received termination signal (SIGTERM). Cleaning up if needed...")
    # If you have Docker or other containers to remove from inside, do so here.
    # Or just exit gracefully:
    sys.exit(0)

# Catch SIGTERM and SIGINT:
signal.signal(signal.SIGTERM, _handle_termination)
signal.signal(signal.SIGINT, _handle_termination)

@app.route('/execute', methods=['POST'])
async def handle():
    try:
        payload = await request.get_json()
        action = payload.get('action')

        if action == 'call_function':
            func_name = payload.get('function')
            args_serialized = payload.get('args', [])
            kwargs_serialized = payload.get('kwargs', {})
            args = deserialize_data(args_serialized) if args_serialized else []
            kwargs = deserialize_data(kwargs_serialized) if kwargs_serialized else {}

            func = getattr(exported_plugin, func_name, None)
            if not func:
                return jsonify({'error': f"Function '{func_name}' not found"}), 404
            if not callable(func):
                return jsonify({'error': f"'{func_name}' is not callable"}), 400

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
            except Exception as e:
                error_trace = traceback.format_exc()
                logger.error(f"Exception during function '{func_name}': {error_trace}")
                if "No more data available" in str(e):
                    return jsonify({'error': 'No more data available'}), 404
                return jsonify({'error': str(e), 'traceback': error_trace}), 500

            if isinstance(result, dict) and 'error' in result:
                return jsonify(result), 400

            if isinstance(result, (bytes, bytearray)):
                length = len(result)
                return Response(result, mimetype='application/octet-stream', headers={'Content-Length': str(length)})

            serialized_result = serialize_data(result)
            return jsonify({'result': serialized_result}), 200

        elif action == 'health_check':
            return jsonify({'result': serialize_data({'status': 'ok'})}), 200
        else:
            return jsonify({'error': 'Invalid or unsupported action'}), 400

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Unhandled exception: {error_trace}")
        return jsonify({'error': str(e), 'traceback': error_trace}), 500

@app.route('/health', methods=['GET'])
async def health_check():
    return jsonify({'result': serialize_data({'status': 'ok'})})

async def init_plugin():
    global exported_plugin
    try:
        plugin_id = os.environ.get('PLUGIN_ID')
        if not plugin_id:
            logger.error("PLUGIN_ID environment variable is not set.")
            return

        plugin_module_name = f".plugin_{plugin_id}"
        plugin_module = importlib.import_module(plugin_module_name, package=current_package)
        exported_plugin = getattr(plugin_module, 'exported_plugin', None)

        if not exported_plugin:
            logger.error(f"'exported_plugin' not found in module '{plugin_module_name}'.")
        else:
            if hasattr(exported_plugin, 'model_adapter'):
                exported_plugin.model_adapter.initialize_environment()
            logger.info("Plugin loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load plugin: {e}")
        logger.debug(traceback.format_exc())

@app.before_serving
async def startup():
    await init_plugin()

if __name__ == '__main__':
    from uvicorn import Config, Server
    loop = asyncio.new_event_loop()
    config = Config(app=app, host='0.0.0.0', port=int(os.environ['PORT']), log_level='info', loop=loop)
    server = Server(config)
    loop.run_until_complete(server.serve())
