import aiofiles
import os
import sys
import shutil
import logging
import docker
import time
import asyncio
import json
from functools import partial
import hashlib
import tempfile
from .serialize import serialize_data, deserialize_data
import aiohttp  # Added for asynchronous HTTP requests

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Plugin management variables
global_plugin_dir = tempfile.mkdtemp()  # e.g., /tmp/my_plugins
plugin_package_name = 'plugin_code'
docker_plugin_dir = f'/app/{plugin_package_name}'  # Updated to match Dockerfile
server_script_name = 'server.py'
server_script_host = os.path.join(global_plugin_dir, server_script_name)
server_script_container = f"{docker_plugin_dir}/{server_script_name}"

# Docker and server configuration
DOCKER_IMAGE = "panthalia_plugin"  # Updated to custom image name
DOCKERFILE_PATH = "Dockerfile"
CONTAINER_NAME_TEMPLATE = "panthalia_plugin_{plugin_id}"
HOST_PORT_BASE = 8000

# Security options
security_options = [
    "no-new-privileges:true",
]

# Resource limits
mem_limit = "16g"
pids_limit = 100

# Initialize Docker client
docker_client = docker.from_env()

# Cache for last plugin
last_plugin_id = None
last_plugin_proxy = None

class PluginProxy:
    """
    Proxy class to interact with the remote plugin server.
    Allows direct async function calls and returns their results.
    """
    def __init__(self, host='localhost', port=HOST_PORT_BASE):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}/execute"
        self.session = aiohttp.ClientSession()  # Initialize aiohttp session

    async def call_remote(self, function, args=None, kwargs=None):
        payload = {
            'action': 'call_function',
            'function': function,
            'args': serialize_data(args) if args else [],
            'kwargs': serialize_data(kwargs) if kwargs else {}
        }

        try:
            headers = {'Content-Type': 'application/json'}
            async with self.session.post(self.base_url, json=payload, headers=headers) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.error(f"HTTP error {response.status}: {text}")
                    raise Exception(f"HTTP error {response.status}: {text}")

                response_obj = await response.json()
                result_serialized = response_obj.get('result')
                if result_serialized is None:
                    raise Exception("No 'result' in response.")

                result = deserialize_data(result_serialized)
                
                if isinstance(result, dict) and 'error' in result:
                    error_msg = result['error']
                    logger.error(f"Error during function '{function}': {error_msg}")
                    if error_msg == 'StopAsyncIteration':
                        raise StopAsyncIteration
                    else:
                        raise Exception(error_msg)

                return result
        except aiohttp.ClientError as e:
            logger.error(f"Request exception during function '{function}': {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error during function '{function}': {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during function '{function}': {e}")
            raise

    def __getattr__(self, name):
        """
        Allows direct async method calls as attributes.
        Example: await proxy.some_function(args) -> calls 'some_function' remotely.
        """
        async def method(*args, **kwargs):
            return await self.call_remote(name, args=args, kwargs=kwargs)
        return method

    async def close(self):
        """
        Close the aiohttp session when done.
        """
        await self.session.close()

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
        await last_plugin_proxy.close()  # Close the aiohttp session
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
        'serialize.py': 'serialize.py',
        'requirements.txt': 'requirements.txt',
        'plugin_server.py': 'server.py'
    }

    for local, global_target in resources.items():
        src = os.path.join(os.path.dirname(__file__), local)
        dst = os.path.join(plugin_package_dir, global_target)
        copy_if_missing(src, dst)

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
                ports={
                    f'{host_port}/tcp': host_port  # Map container's 8000 to host_port
                },
                mem_limit=mem_limit,
                pids_limit=pids_limit,
                volumes={
                    plugin_package_dir: {'bind': docker_plugin_dir, 'mode': 'rw'},
                    tmp_dir: {'bind': tmp_dir, 'mode': 'rw'}
                },
                environment={
                    "TMPDIR": tmp_dir,
                    "PIP_NO_CACHE_DIR": "off",
                    "HOME": tmp_dir,
                    "PLUGIN_ID": str(plugin_id),
                    "PORT": str(host_port),
                    "DOCKER_PLUGIN_DIR": docker_plugin_dir
                },
                user="nobody",
                device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
                extra_hosts={"host.docker.internal": "host-gateway"}
            )

            logger.info(f"Container {container_name} started with ID: {container.id}")
        except docker.errors.DockerException as e:
            logger.error(f"Failed to start Docker container: {e}")
            raise

    if not await wait_for_server(server_url):
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

async def wait_for_server(url, timeout=30):
    logger.info(f"Waiting for server at {url} to be ready...")
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < timeout:
            try:
                # Create the payload with a health_check action
                payload = {
                    'action': 'health_check'
                }
                headers = {'Content-Type': 'application/json'}

                async with session.post(url, json=payload, headers=headers) as response:
                    logger.debug(f"Server raw response: {await response.text()}")

                    if response.status == 200:
                        response_obj = await response.json()
                        result_serialized = response_obj.get('result')
                        if result_serialized is None:
                            logger.debug("No 'result' in response.")
                            continue
                        result = deserialize_data(result_serialized)
                        logger.debug(f'wait_for_server deserialized response: {result}')

                        if isinstance(result, dict) and result.get('status') == 'ok':
                            logger.info("Server is up and running.")
                            return True
            except (aiohttp.ClientError, json.JSONDecodeError, ValueError) as e:
                logger.debug(f"Ping attempt failed: {e}")
            await asyncio.sleep(1)

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
