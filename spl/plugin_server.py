import asyncio
import importlib
import sys
import os
import json
import traceback
import logging
from quart import Quart, jsonify, request
import uuid  # Import uuid for object ID generation
from functools import partial
import base64
from safetensors.torch import save_file as safetensors_save
from safetensors.torch import load_file as safetensors_load
import torch  # Assuming tensors are PyTorch tensors
from .serialize import serialize_data, deserialize_data

# Adjust the plugin directory as needed
PLUGIN_DIR = os.environ['DOCKER_PLUGIN_DIR']
if PLUGIN_DIR not in sys.path:
    sys.path.append(PLUGIN_DIR)

# Set the current package for the plugin_code package context
current_package = __package__
exported_plugin = None

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)

app = Quart(__name__)

# Object registry to manage object references
object_registry = {}

def register_object(obj):
    """
    Register an object and return its unique ID.
    """
    object_id = str(uuid.uuid4())
    object_registry[object_id] = obj
    return object_id

def unregister_object(object_id):
    """
    Unregister an object by its ID.
    """
    if object_id in object_registry:
        del object_registry[object_id]
        logger.info(f"Unregistered object_id {object_id}.")
    else:
        logger.warning(f"Attempted to unregister non-existent object_id {object_id}.")

@app.route('/execute', methods=['POST'])
async def handle():
    try:
        payload = await request.get_json()
        logger.debug(f"Received payload: {payload}")
        action = payload.get('action')

        # Directly access 'action' and other non-serialized fields
        object_id = payload.get('object_id', None)

        if not exported_plugin and action != 'release_object':
            return jsonify({'result': serialize_data({'error': 'Plugin not loaded'})}), 500

        if object_id and action != 'release_object':
            target = object_registry.get(object_id, None)
            if not target:
                return jsonify({'result': serialize_data({'error': f"Invalid object_id: {object_id}"})}), 404
        else:
            target = exported_plugin

        if action == 'call_function':
            func_name = payload.get('function')
            args_serialized = payload.get('args', None)
            kwargs_serialized = payload.get('kwargs', None)

            args = deserialize_data(args_serialized) if args_serialized else []
            kwargs = deserialize_data(kwargs_serialized) if kwargs_serialized else {}

            func = getattr(target, func_name, None)
            if not func:
                return jsonify({'result': serialize_data({'error': f"Function {func_name} not found"})}), 404

            if not callable(func):
                return jsonify({'result': serialize_data({'error': f"{func_name} is not callable"})}), 400

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)  # Await coroutine functions
                else:
                    # Offload blocking functions to a thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, partial(func, *args, **kwargs))
            except StopAsyncIteration:
                return jsonify({'result': serialize_data({'error': 'StopAsyncIteration'})}), 500

            if isinstance(result, (int, float, str, bool, list, dict, type(None))):
                serialized_result = serialize_data(result)
                return jsonify({'result': serialized_result})
            else:
                new_object_id = register_object(result)
                serialized_object = serialize_data({'object_id': new_object_id})
                return jsonify({'result': serialized_object})

        elif action == 'get_attribute':
            attr_name = payload.get('attribute')
            attr = getattr(target, attr_name, None)
            if attr is None:
                return jsonify({'result': serialize_data({'error': f"Attribute {attr_name} not found"})}), 404

            response = {"is_callable": callable(attr)}
            if callable(attr):
                response['is_async'] = asyncio.iscoroutinefunction(attr)

            if isinstance(attr, (int, float, str, bool, list, dict, type(None))):
                response['value'] = attr
            elif not callable(attr):
                new_object_id = register_object(attr)
                response['object_id'] = new_object_id

            serialized_response = serialize_data(response)
            return jsonify({'result': serialized_response})

        elif action == 'set_attribute':
            attr_name = payload.get('attribute')
            value_serialized = payload.get('value')
            value = deserialize_data(value_serialized)
            setattr(target, attr_name, value)
            serialized_message = serialize_data('Attribute set successfully')
            return jsonify({'result': serialized_message})

        elif action == 'release_object':
            obj_id = payload.get('object_id')
            if obj_id:
                unregister_object(obj_id)
                serialized_message = serialize_data('Object released successfully')
                return jsonify({'result': serialized_message})
            else:
                return jsonify({'result': serialize_data({'error': 'No object_id provided'})}), 400

        else:
            return jsonify({'result': serialize_data({'error': 'Invalid action'})}), 400

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Exception occurred: {error_trace}")
        serialized_error = serialize_data({'error': str(e), 'traceback': error_trace})
        return jsonify({'result': serialized_error}), 500


@app.route('/health', methods=['GET'])
async def health_check():
    return jsonify({'result': serialize_data({'status': 'ok'})})

async def init_plugin():
    global exported_plugin
    # Import the plugin as a module within the plugin_code package
    try:
        plugin_module_name = f".plugin_" + os.environ.get('PLUGIN_ID')
        plugin_module = importlib.import_module(plugin_module_name, package=current_package)
        exported_plugin = getattr(plugin_module, 'exported_plugin')
        
        # Define a basic ping function for status check
        async def __ping__():
            return {"status": "ok"}
        
        # Attach the ping function to the plugin
        setattr(exported_plugin, '__ping__', __ping__)
        
        # Initialize the plugin environment if required
        if hasattr(exported_plugin, 'model_adapter'):
            exported_plugin.model_adapter.initialize_environment()
        logger.info("Plugin loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load plugin: {e}")

@app.before_serving
async def startup():
    await init_plugin()

if __name__ == '__main__':
    from uvicorn import Config, Server
    loop = asyncio.new_event_loop()
    config = Config(
        app=app,
        host='0.0.0.0',
        port=int(os.environ.get('PORT')),
        log_level='info',
        loop=loop
    )
    server = Server(config)
    loop.run_until_complete(server.serve())
