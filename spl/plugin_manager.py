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
import importlib
from functools import partial

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# plugin management variables
global_plugin_dir = '/tmp/my_plugins'
docker_plugin_dir = '/plugins'
server_script_name = 'server.py'
server_script_host = os.path.join(global_plugin_dir, server_script_name)
server_script_container = f"/app/{server_script_name}"

# docker and server configuration

DOCKER_IMAGE = "pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel"
CONTAINER_NAME_TEMPLATE = "secure_plugin_container_{plugin_id}"
HOST_PORT_BASE = 8000
CONTAINER_PORT = 8000

# security options
security_options = [
    "no-new-privileges:true",
]

# resource limits
mem_limit = "512m"
pids_limit = 100

# initialize docker client
docker_client = docker.from_env()

# Cache for last plugin
last_plugin_id = None
last_plugin_proxy = None

class PluginProxy:
    def __init__(self, host='localhost', port=HOST_PORT_BASE):
        self.url = f"http://{host}:{port}/execute"

    def serialize_data(self, data):
        try:
            return json.dumps(data)
        except (TypeError, OverflowError) as e:
            logger.error(f"serialization error: {e}")
            raise

    def deserialize_data(self, data):
        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            logger.error(f"deserialization error: {e}")
            raise

    def call_function(self, function_name, *args, **kwargs):
        payload = {
            'function': function_name,
            'args': args,
            'kwargs': kwargs
        }
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            return self.deserialize_data(response.text)
        except requests.exceptions.RequestException as e:
            logger.error(f"error calling function {function_name}: {e}")
            raise

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            return self.call_function(name, *args, **kwargs)
        return wrapper

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

    # Write the server script inside the plugin directory
    await write_server_script(plugin_id, plugin_package_dir)

    # Set up Docker and PluginProxy
    plugin_proxy = await setup_docker_container(plugin_id, plugin_package_dir)
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

async def write_server_script(plugin_id, plugin_package_dir):
    """
    Generate and write the server.py script within the plugin's directory.
    """
    server_script_content = f"""
import aiohttp
from aiohttp import web
import asyncio
import importlib
import sys
import os
import json
import traceback

# Adjust the plugin directory as needed
PLUGIN_DIR = '{docker_plugin_dir}'
if PLUGIN_DIR not in sys.path:
    sys.path.append(PLUGIN_DIR)

exported_plugin = None

async def handle(request):
    try:
        data = await request.json()
        func_name = data['function']
        args = data.get('args', [])
        kwargs = data.get('kwargs', {{}})
        
        if not exported_plugin:
            return web.json_response({{'error': 'Plugin not loaded'}}, status=500)
        
        func = getattr(exported_plugin, func_name, None)
        if not func:
            return web.json_response({{'error': f'Function {{func_name}} not found'}}, status=404)
        
        result = func(*args, **kwargs)
        return web.json_response({{'result': result}})
    except Exception as e:
        traceback.print_exc()
        return web.json_response({{'error': str(e)}}, status=500)

async def init_plugin():
    global exported_plugin
    # Import the plugin
    try:
        plugin_module = importlib.import_module('plugin_{plugin_id}')  # Plugin module name
        exported_plugin = getattr(plugin_module, 'exported_plugin')
        
        # Define a basic ping function for status check
        def __ping__():
            return {{'status': 'ok'}}

        # Attach the ping function to the plugin
        setattr(exported_plugin, '__ping__', __ping__)

        # Initialize the plugin environment if required
        if hasattr(exported_plugin, 'model_adapter'):
            exported_plugin.model_adapter.initialize_environment()
    except Exception as e:
        print(f"Failed to load plugin: {{e}}")

async def init_app():
    app = web.Application()
    app.add_routes([
        web.post('/execute', handle)
    ])
    await init_plugin()
    return app

if __name__ == '__main__':
    web.run_app(init_app(), host='0.0.0.0', port=8000)
""".strip()

    server_script_path = os.path.join(plugin_package_dir, server_script_name)
    async with aiofiles.open(server_script_path, mode='w') as f:
        await f.write(server_script_content)
    logger.info(f"Generated server script at {server_script_path}")

async def setup_docker_container(plugin_id, plugin_package_dir):
    container_name = CONTAINER_NAME_TEMPLATE.format(plugin_id=plugin_id)
    host_port = HOST_PORT_BASE + int(plugin_id)

    server_url = f"http://localhost:{host_port}/execute"
    tmp_dir = "/tmp"
    
    try:
        # Check if container exists
        container = docker_client.containers.get(container_name)
        if container.status != 'running':
            container.start()
            logger.info(f"Started existing container {container_name}")
    except docker.errors.NotFound:
        logger.info(f"Creating and starting container {container_name}")
        try:
            container = docker_client.containers.run(
                DOCKER_IMAGE,
                name=container_name,
                detach=True,
                security_opt=security_options,
                network_mode="host",
                mem_limit=mem_limit,
                pids_limit=pids_limit,
                volumes={
                    plugin_package_dir: {'bind': docker_plugin_dir, 'mode': 'rw'},  # Make plugin directory writable
                    tmp_dir: {'bind': tmp_dir, 'mode': 'rw'}  # Ensure tmp directory is writable
                },
                environment={
                    "TMPDIR": tmp_dir,
                    "PIP_NO_CACHE_DIR": "off",
                    "HOME": tmp_dir  # **Added this line to set HOME to /tmp**
                },
                command=f"/bin/bash -c 'ls {docker_plugin_dir} && cat {docker_plugin_dir}/requirements.txt && python -m pip install --upgrade pip && python -m pip install -r {docker_plugin_dir}/requirements.txt && python {docker_plugin_dir}/{server_script_name}'",
                user="nobody"  # Running as non-root user
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


def wait_for_server(url, timeout=30):
    logger.info(f"waiting for server at {url} to be ready...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.post(url, json={'function': '__ping__'})
            if response.status_code == 200 and 'result' in response.json():
                logger.info("server is up and running.")
                return True
        except requests.exceptions.ConnectionError:
            pass
        except json.JSONDecodeError:
            pass
        time.sleep(1)
    logger.error("server did not start within the timeout period.")
    return False

def show_container_logs(container):
    try:
        logs = container.logs().decode('utf-8')
        logger.info(f"container logs for {container.name}:\n{logs}")
    except docker.errors.DockerException as e:
        logger.error(f"failed to retrieve logs for container {container.name}: {e}")

async def teardown_docker_container(plugin_id):
    container_name = CONTAINER_NAME_TEMPLATE.format(plugin_id=plugin_id)
    try:
        container = docker_client.containers.get(container_name)
        container.stop()
        container.remove()
        logger.info(f"container {container_name} stopped and removed.")
    except docker.errors.NotFound:
        logger.warning(f"container {container_name} not found during teardown.")
    except docker.errors.DockerException as e:
        logger.error(f"error during teardown of container {container_name}: {e}")
