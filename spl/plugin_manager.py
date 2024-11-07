import aiofiles
import os
import sys
import shutil
import logging
import docker
import time
import requests
import asyncio
import json
from functools import partial
import hashlib
import threading
import uuid  # Added for unique object IDs

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Plugin management variables
global_plugin_dir = '/tmp/my_plugins'
docker_plugin_dir = '/app/plugin_code'  # Updated to match Dockerfile
server_script_name = 'server.py'
server_script_host = os.path.join(global_plugin_dir, server_script_name)
server_script_container = f"{docker_plugin_dir}/{server_script_name}"

# Docker and server configuration
DOCKER_IMAGE = "panthalia_plugin"  # Updated to custom image name
DOCKERFILE_PATH = "Dockerfile"
CONTAINER_NAME_TEMPLATE = "secure_plugin_container_{plugin_id}"
HOST_PORT_BASE = 8000

# Security options
security_options = [
    "no-new-privileges:true",
]

# Resource limits
mem_limit = "512m"
pids_limit = 100

# Initialize Docker client
docker_client = docker.from_env()

# Cache for last plugin
last_plugin_id = None
last_plugin_proxy = None

class PluginProxy:
    """
    Proxy class to interact with the remote plugin server.
    Supports recursive access to child objects by managing object IDs.
    Handles coroutine functions by returning awaitable wrappers.
    """
    def __init__(self, host='localhost', port=HOST_PORT_BASE, object_id=None):
        self.__dict__['host'] = host
        self.__dict__['port'] = port
        self.__dict__['object_id'] = object_id
        self.__dict__['base_url'] = f"http://{host}:{port}/execute"

    def serialize_data(self, data):
        try:
            return json.dumps(data)
        except (TypeError, OverflowError) as e:
            logger.error(f"Serialization error: {e}")
            raise

    def deserialize_data(self, data):
        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            logger.error(f"Deserialization error: {e}")
            raise

    def call_remote(self, action, function=None, args=None, kwargs=None, attribute=None, value=None):
        payload = {'action': action}
        if self.object_id:
            payload['object_id'] = self.object_id
        if function:
            payload['function'] = function
        if args:
            payload['args'] = args
        if kwargs:
            payload['kwargs'] = kwargs
        if attribute:
            payload['attribute'] = attribute
        if value is not None:
            payload['value'] = value

        try:
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()
            result = self.deserialize_data(response.text)
            if 'error' in result:
                logger.error(f"Error during '{action}' for '{function}': {result['error']}")
                raise Exception(result['error'])
            return result.get('result')
        except requests.exceptions.RequestException as e:
            # Added response details here
            try:
                status_code = response.status_code
                response_text = response.text
            except:
                status_code = 'No Response'
                response_text = 'No Response'
            logger.error(f"Error during '{action}': {e}, response status: {status_code}, response text: {response_text}")
            raise

    def __getattr__(self, name):
        """
        Handle dynamic attribute access and method calls.
        If the attribute is a method, return a callable that invokes the remote method.
        If the attribute is an async method, return an awaitable coroutine.
        If the attribute is an object, return a new PluginProxy representing the child object.
        If the attribute is a simple value, return it directly.
        """
        try:
            attr = self.call_remote('get_attribute', attribute=name)
            if isinstance(attr, dict):
                if attr.get('is_callable'):
                    if attr.get('is_async'):
                        # Return an async wrapper
                        async def async_method(*args, **kwargs):
                            # Use a separate thread to perform the blocking HTTP request
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(
                                None,
                                partial(
                                    self.call_remote,
                                    'call_function',
                                    function=name,
                                    args=args,
                                    kwargs=kwargs
                                )
                            )
                            return result
                        return async_method
                    else:
                        # Return a synchronous callable
                        def method(*args, **kwargs):
                            return self.call_remote('call_function', function=name, args=args, kwargs=kwargs)
                        return method
                elif 'object_id' in attr:
                    # Return a new PluginProxy for the nested object
                    return PluginProxy(host=self.host, port=self.port, object_id=attr['object_id'])
                elif 'value' in attr:
                    # Return the attribute's value directly
                    return attr['value']
                else:
                    # Unexpected format
                    logger.error(f"Unexpected attribute format for '{name}': {attr}")
                    raise Exception(f"Unexpected attribute format for '{name}'")
            else:
                # Unexpected format
                logger.error(f"Unexpected attribute response for '{name}': {attr}")
                raise Exception(f"Unexpected attribute response for '{name}'")
        except Exception as e:
            logger.error(f"Error accessing attribute '{name}': {e}")
            raise

    def __setattr__(self, name, value):
        """
        Handle setting attributes on the remote object.
        """
        if name in ('host', 'port', 'object_id', 'base_url'):
            super().__setattr__(name, value)
        else:
            self.call_remote('set_attribute', attribute=name, value=value)

    def __del__(self):
        """
        Destructor to release the object_id on the server to prevent memory leaks.
        """
        if self.object_id:
            try:
                self.call_remote('release_object')
                logger.info(f"Released object_id {self.object_id} on server.")
            except Exception as e:
                logger.error(f"Failed to release object_id {self.object_id}: {e}")

async def get_plugin(plugin_id, db_adapter):
    """
    Fetch the plugin from the database, set it up, start the Docker container, and return a PluginProxy instance.
    Implements caching to reuse the last loaded plugin.
    """
    global last_plugin_id, last_plugin_proxy
    setup_dir()
    logger.info(f'Fetching plugin "{plugin_id}"')

    if plugin_id == last_plugin_id and last_plugin_proxy is not None:
        logger.info(f'Reusing cached plugin "{plugin_id}"')
        return last_plugin_proxy

    # If a different plugin was previously loaded, stop its container
    if last_plugin_proxy is not None:
        await teardown_docker_container(last_plugin_id)
        last_plugin_proxy = None
        last_plugin_id = None

    plugin_package_dir = os.path.join(global_plugin_dir, f'plugin_{plugin_id}')
    create_subdirectory(plugin_package_dir)

    # Fetch and write plugin code
    plugin_file_name = f'plugin_{plugin_id}.py'
    plugin_path = os.path.join(plugin_package_dir, plugin_file_name)
    await fetch_and_write_plugin_code(plugin_id, db_adapter, plugin_path)

    # Copy necessary plugin resources
    setup_plugin_files(plugin_package_dir)

    # Create __init__.py to make it a package
    init_file_path = os.path.join(plugin_package_dir, '__init__.py')
    if not os.path.exists(init_file_path):
        async with aiofiles.open(init_file_path, mode='w') as f:
            await f.write('# Init file for plugin package\n')
        logger.info(f"Created __init__.py at {init_file_path}")

    host_port = get_port(plugin_id)

    # Write the server script inside the plugin directory
    await write_server_script(
        plugin_id, plugin_package_dir, host_port)

    # Ensure the custom Docker image is built
    await ensure_docker_image()

    # Set up Docker and PluginProxy
    plugin_proxy = await setup_docker_container(
        plugin_id, plugin_package_dir, host_port)
    last_plugin_proxy = plugin_proxy
    last_plugin_id = plugin_id
    logger.info(f'Plugin "{plugin_id}" is set up and ready to use.')

    return plugin_proxy

def setup_dir():
    """
    Ensure the global plugin directory exists and is in sys.path.
    """
    if not os.path.exists(global_plugin_dir):
        os.makedirs(global_plugin_dir)
        logger.info(f"Created plugin directory at {global_plugin_dir}")
    if global_plugin_dir not in sys.path:
        sys.path.append(global_plugin_dir)
        logger.info(f"Added {global_plugin_dir} to sys.path")
    
    # Ensure /tmp directory exists
    tmp_dir = "/tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
        logger.info(f"Created temporary directory at {tmp_dir}")

def create_subdirectory(path):
    """
    Create a subdirectory if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created subdirectory at {path}")

def copy_if_missing(src, dst):
    """
    Copy a file or directory from src to dst if dst does not exist.
    """
    if os.path.exists(src) and not os.path.exists(dst):
        if os.path.isdir(src):
            shutil.copytree(src, dst)
            logger.info(f"Copied directory from {src} to {dst}")
        else:
            shutil.copy(src, dst)
            logger.info(f"Copied file from {src} to {dst}")

async def fetch_and_write_plugin_code(plugin_id, db_adapter, plugin_path):
    """
    Fetches the plugin code from the database and writes it to the specified path.
    """
    plugin_record = await db_adapter.get_plugin(plugin_id)
    if not plugin_record or not hasattr(plugin_record, 'code'):
        logger.error(f"No plugin code found for plugin_id: {plugin_id}")
        raise ValueError(f"No plugin code found for plugin_id: {plugin_id}")
    plugin_code = plugin_record.code
    async with aiofiles.open(plugin_path, mode='w') as f:
        await f.write(plugin_code)
    logger.info(f"Fetched and wrote plugin code to {plugin_path}")

def setup_plugin_files(plugin_package_dir):
    """
    Copies necessary plugin resources into the plugin package directory.
    """
    resources = {
        'adapters': 'adapters',
        'datasets': 'datasets',
        'tokenizer.py': 'tokenizer.py',
        'device.py': 'device.py',
        'common.py': 'common.py',
        'requirements.txt': 'requirements.txt'
    }
    
    for local, global_target in resources.items():
        src = os.path.join(os.path.dirname(__file__), local)
        dst = os.path.join(plugin_package_dir, global_target)
        copy_if_missing(src, dst)

async def write_server_script(plugin_id, plugin_package_dir, host_port):
    """
    Generate and write the server.py script within the plugin's directory.
    Enhanced to handle object references for recursive proxying and coroutine functions.
    """
    server_script_content = f'''
import asyncio
import importlib
import sys
import os
import json
import traceback
import logging
from quart import Quart, jsonify, request
import uuid  # Import uuid for object ID generation

# Adjust the plugin directory as needed
PLUGIN_DIR = "{docker_plugin_dir}"
if PLUGIN_DIR not in sys.path:
    sys.path.append(PLUGIN_DIR)

# Set the current package for the plugin_code package context
current_package = __package__
exported_plugin = None

logger = logging.getLogger(__name__)

app = Quart(__name__)

# Object registry to manage object references
object_registry = {{}}

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
        logger.info(f"Unregistered object_id {{object_id}}.")
    else:
        logger.warning(f"Attempted to unregister non-existent object_id {{object_id}}.")

@app.route('/execute', methods=['POST'])
async def handle():
    try:
        data = await request.get_json()
        action = data.get('action')
        object_id = data.get('object_id', None)

        if not exported_plugin and action != 'release_object':
            return jsonify(error='Plugin not loaded'), 500

        if object_id and action != 'release_object':
            target = object_registry.get(object_id, None)
            if not target:
                return jsonify(error=f'Invalid object_id: {{object_id}}'), 404
        else:
            target = exported_plugin

        if action == 'call_function':
            func_name = data['function']
            args = data.get('args', [])
            kwargs = data.get('kwargs', {{}})
            
            func = getattr(target, func_name, None)
            if not func:
                return jsonify(error=f'Function {{func_name}} not found'), 404
            
            if not callable(func):
                return jsonify(error=f'{{func_name}} is not callable'), 400

            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)  # Await coroutine functions
            else:
                result = func(*args, **kwargs)

            # If the result is an object, register it and return its object_id
            if isinstance(result, (int, float, str, bool, list, dict, type(None))):
                return jsonify(result=result)
            else:
                new_object_id = register_object(result)
                return jsonify(result={{'object_id': new_object_id}})
        
        elif action == 'get_attribute':
            attr_name = data['attribute']
            attr = getattr(target, attr_name, None)
            if attr is None:
                return jsonify(error=f'Attribute {{attr_name}} not found'), 404
            
            # If the attribute is callable, indicate it and if it's a coroutine
            response = {{'is_callable': callable(attr)}}
            if callable(attr):
                response['is_async'] = asyncio.iscoroutinefunction(attr)
            
            if isinstance(attr, (int, float, str, bool, list, dict, type(None))):
                response['value'] = attr
            elif not callable(attr):
                # If it's an object, register it and return its object_id
                new_object_id = register_object(attr)
                response['object_id'] = new_object_id
            
            return jsonify(result=response)
        
        elif action == 'set_attribute':
            attr_name = data['attribute']
            value = data['value']
            setattr(target, attr_name, value)
            return jsonify(result='Attribute set successfully')
        
        elif action == 'release_object':
            obj_id = data.get('object_id')
            if obj_id:
                unregister_object(obj_id)
                return jsonify(result='Object released successfully')
            else:
                return jsonify(error='No object_id provided'), 400

        else:
            return jsonify(error='Invalid action'), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500

@app.route('/health', methods=['GET'])
async def health_check():
    return jsonify(status='ok')

async def init_plugin():
    global exported_plugin
    # Import the plugin as a module within the plugin_code package
    try:
        plugin_module_name = f".plugin_{plugin_id}"
        plugin_module = importlib.import_module(plugin_module_name, package=current_package)
        exported_plugin = getattr(plugin_module, 'exported_plugin')
        
        # Define a basic ping function for status check
        async def __ping__():
            return {{'status': 'ok'}}
        
        # Attach the ping function to the plugin
        setattr(exported_plugin, '__ping__', __ping__)
        
        # Initialize the plugin environment if required
        if hasattr(exported_plugin, 'model_adapter'):
            exported_plugin.model_adapter.initialize_environment()
        print("Plugin loaded successfully.")
    except Exception as e:
        print(f"Failed to load plugin: {{e}}")

@app.before_serving
async def startup():
    await init_plugin()

if __name__ == '__main__':
    from hypercorn.asyncio import serve
    from hypercorn.config import Config
        
    config = Config()
    config.bind = ['0.0.0.0:8001']
    config.loglevel = 'info'
    asyncio.run(serve(app, config))
'''.strip()
    
    server_script_path = os.path.join(plugin_package_dir, server_script_name)
    async with aiofiles.open(server_script_path, mode='w') as f:
        await f.write(server_script_content)
    logger.info(f"Generated server script at {server_script_path}")

async def ensure_docker_image():
    """
    Ensure the custom Docker image is built before running containers.
    """
    logger.info("Ensuring Docker image is built.")
    try:
        # Check if image exists
        docker_client.images.get(DOCKER_IMAGE)
        logger.info(f"Docker image '{DOCKER_IMAGE}' already exists.")
    except docker.errors.ImageNotFound:
        logger.info(f"Docker image '{DOCKER_IMAGE}' not found. Building the image.")
        # Build the image
        await build_image()

async def build_image():
    """
    Build the custom Docker image using the Dockerfile.
    """
    logger.info(f"Building Docker image '{DOCKER_IMAGE}'. This may take a while...")
    try:
        # Use the Docker SDK to build the image
        image, logs = docker_client.images.build(
            path=".",  # Context is current directory
            dockerfile=DOCKERFILE_PATH,
            tag=DOCKER_IMAGE,
            rm=True
        )
        for chunk in logs:
            if 'stream' in chunk:
                for line in chunk['stream'].splitlines():
                    logger.debug(line)
        logger.info(f"Docker image '{DOCKER_IMAGE}' built successfully.")
    except docker.errors.BuildError as e:
        logger.error(f"Failed to build Docker image: {e}")
        raise
    except docker.errors.APIError as e:
        logger.error(f"Docker API error during image build: {e}")
        raise

async def setup_docker_container(plugin_id, plugin_package_dir, host_port):
    container_name = CONTAINER_NAME_TEMPLATE.format(plugin_id=plugin_id)

    server_url = f"http://localhost:{host_port}/execute"
    tmp_dir = "/tmp"

    try:
        # Check if container exists
        container = docker_client.containers.get(container_name)
        logger.info(f"Container {container_name} already exists.")
        if container.status != 'running':
            container.start()
            logger.info(f"Started existing container {container_name}")
    except docker.errors.NotFound:
        logger.info(f"Creating and starting container {container_name}")
        try:
            # Mount the plugin directory
            container = docker_client.containers.run(
                DOCKER_IMAGE,
                name=container_name,
                detach=True,
                security_opt=security_options,
                ports={'8001/tcp': host_port},  # Map container's 8001 to host_port
                mem_limit=mem_limit,
                pids_limit=pids_limit,
                volumes={
                    plugin_package_dir: {'bind': docker_plugin_dir, 'mode': 'rw'},
                    tmp_dir: {'bind': tmp_dir, 'mode': 'rw'}
                },
                environment={
                    "TMPDIR": tmp_dir,
                    "PIP_NO_CACHE_DIR": "off",
                    "HOME": tmp_dir
                },
                user="nobody"
            )

            logger.info(f"Container {container_name} started with ID: {container.id}")
        except docker.errors.DockerException as e:
            logger.error(f"Failed to start Docker container: {e}")
            raise

    if not wait_for_server(server_url):
        show_container_logs(container)
        logger.error("Exiting due to server startup failure.")
        await teardown_docker_container(plugin_id)
        raise Exception("Docker container server failed to start.")
    
    plugin_proxy = PluginProxy(host='localhost', port=host_port)
    logger.info(f"Plugin proxy created for plugin '{plugin_id}' at {server_url}")
    return plugin_proxy

def get_port(plugin_id):
    """
    Generate a unique port number for the given plugin_id.
    """
    try:
        # Attempt to convert plugin_id to int
        return HOST_PORT_BASE + int(plugin_id)
    except ValueError:
        # If plugin_id is not an int, generate a hash-based port
        hash_object = hashlib.sha256(str(plugin_id).encode())
        hash_int = int(hash_object.hexdigest(), 16)
        return HOST_PORT_BASE + (hash_int % 1000)  # Ports 8000-8999

def wait_for_server(url, timeout=30):
    logger.info(f"Waiting for server at {url} to be ready...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Ping the server by calling the __ping__ function
            response = requests.post(url, json={'action': 'call_function', 'function': '__ping__'})
            logger.info(f"Server response: {response.text}")
            if response.status_code == 200:
                result = response.json().get('result')
                logger.info(f'wait_for_server response: {result}')
                if isinstance(result, dict) and result.get('status') == 'ok':
                    logger.info("Server is up and running.")
                    return True
        except (requests.exceptions.ConnectionError, json.JSONDecodeError) as e:
            logger.debug(f"Ping attempt failed: {e}")
        time.sleep(1)
    logger.error("Server did not start within the timeout period.")
    return False

def show_container_logs(container):
    try:
        logs = container.logs().decode('utf-8')
        logger.info(f"Container logs for {container.name}:\n{logs}")
    except docker.errors.DockerException as e:
        logger.error(f"Failed to retrieve logs for container {container.name}: {e}")

async def teardown_docker_container(plugin_id):
    container_name = CONTAINER_NAME_TEMPLATE.format(plugin_id=plugin_id)
    try:
        container = docker_client.containers.get(container_name)
        container.stop()
        container.remove()
        logger.info(f"Container {container_name} stopped and removed.")
    except docker.errors.NotFound:
        logger.warning(f"Container {container_name} not found during teardown.")
    except docker.errors.DockerException as e:
        logger.error(f"Error during teardown of container {container_name}: {e}")
