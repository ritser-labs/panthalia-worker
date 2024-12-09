import asyncio
import importlib
import sys
import os
import json
import traceback
import logging
from quart import Quart, jsonify, request
from functools import partial
from .serialize import serialize_data, deserialize_data

# Adjust the plugin directory as needed
PLUGIN_DIR = os.environ.get('DOCKER_PLUGIN_DIR', '/app/plugin_code')
if PLUGIN_DIR not in sys.path:
    sys.path.append(PLUGIN_DIR)

# Set the current package for the plugin_code package context
current_package = __package__
exported_plugin = None

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)

app = Quart(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 * 1024  # 100 GB limit

@app.route('/execute', methods=['POST'])
async def handle():
    try:
        payload = await request.get_json()
        # logger.debug(f"Received payload: {payload}")
        action = payload.get('action')

        if not exported_plugin and action != 'health_check':
            return jsonify({'result': serialize_data({'error': 'Plugin not loaded'})}), 500

        if action == 'call_function':
            func_name = payload.get('function')
            args_serialized = payload.get('args', [])
            kwargs_serialized = payload.get('kwargs', {})

            args = deserialize_data(args_serialized) if args_serialized else []
            kwargs = deserialize_data(kwargs_serialized) if kwargs_serialized else {}

            func = getattr(exported_plugin, func_name, None)
            if not func:
                return jsonify({'result': serialize_data({'error': f"Function '{func_name}' not found"})}), 404

            if not callable(func):
                return jsonify({'result': serialize_data({'error': f"'{func_name}' is not callable"})}), 400

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)  # Await coroutine functions
                else:
                    # Offload blocking functions to a thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, partial(func, *args, **kwargs))
            except StopAsyncIteration:
                return jsonify({'result': serialize_data({'error': 'StopAsyncIteration'})}), 500
            except Exception as e:
                error_trace = traceback.format_exc()
                logger.error(f"Exception during function '{func_name}': {error_trace}")
                return jsonify({'result': serialize_data({'error': str(e), 'traceback': error_trace})}), 500

            # Serialize and return the result
            serialized_result = serialize_data(result)
            return jsonify({'result': serialized_result})

        elif action == 'health_check':
            # Simple health check action
            return jsonify({'result': serialize_data({'status': 'ok'})})

        else:
            return jsonify({'result': serialize_data({'error': 'Invalid or unsupported action'})}), 400

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Unhandled exception: {error_trace}")
        serialized_error = serialize_data({'error': str(e), 'traceback': error_trace})
        return jsonify({'result': serialized_error}), 500


@app.route('/health', methods=['GET'])
async def health_check():
    return jsonify({'result': serialize_data({'status': 'ok'})})


async def init_plugin():
    global exported_plugin
    # Import the plugin as a module within the plugin_code package
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
            return

        # Initialize the plugin environment if required
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
    config = Config(
        app=app,
        host='0.0.0.0',
        port=int(os.environ['PORT']),
        log_level='info',
        loop=loop
    )
    server = Server(config)
    loop.run_until_complete(server.serve())
