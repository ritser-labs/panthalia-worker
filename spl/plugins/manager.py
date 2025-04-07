# spl/plugins/manager.py

import aiofiles
import os
import sys
import shutil
import logging
import docker
import time
import asyncio
import json
import hashlib
import tempfile
import aiohttp
import subprocess
import platform
import atexit
import paramiko
from stat import S_ISDIR

from spl.deploy.cloud_adapters.runpod import get_pod_ssh_ip_port_sync

from spl.worker.logging_config import logger

from ..util.constants import DEFAULT_DOCKER_IMAGE
from .serialize import serialize_data, deserialize_data

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if platform.system() == "Windows":
    DOCKER_ENGINE_URL = "npipe:////./pipe/docker_engine"
else:
    DOCKER_ENGINE_URL = "unix:///var/run/docker.sock"

logger.info(f"Using Docker engine at {DOCKER_ENGINE_URL}")

# -----------------------------------------------------------------------------
# GLOBALS
# -----------------------------------------------------------------------------

# We create a global temp directory for plugin code
global_plugin_dir = tempfile.mkdtemp(prefix="local_plugin_")
plugin_package_name = 'plugin_code'
docker_plugin_dir = f'/app/{plugin_package_name}'

# Docker container name pattern
CONTAINER_NAME_TEMPLATE = "panthalia_plugin_{job_id}{suffix}"
HOST_PORT_BASE = 8000

security_options = ["no-new-privileges:true"]
mem_limit = "16g"
pids_limit = 100

try:
    docker_client = docker.DockerClient(base_url=DOCKER_ENGINE_URL)
except docker.errors.DockerException as e:
    logger.error(f"Failed to connect to Docker engine. Ensure Docker is running: {e}")
    raise

# -----------------------------------------------------------------------------
# Track each job’s current RunPod or Docker ID
# -----------------------------------------------------------------------------
JOB_TO_POD_ID = {}         # job_id -> runpod ID string
JOB_TO_CONTAINER = {}      # job_id -> docker container name

# We also keep these sets in case you want to do final cleanup on shutdown
LAUNCHED_RUNPOD_IDS = set()
LAUNCHED_DOCKER_CONTAINERS = set()

# For job switching
LAST_JOB_ID = None

# -----------------------------------------------------------------------------
# Cleanup on process exit: kill everything we launched
# -----------------------------------------------------------------------------
def shutdown_runpod_instances():
    """Kill all runpod pods we explicitly launched."""
    from spl.deploy.cloud_adapters import runpod as runpod_adapter
    logger.info("Shutting down runpod pods from LAUNCHED_RUNPOD_IDS...")
    for pod_id in list(LAUNCHED_RUNPOD_IDS):
        try:
            runpod_adapter.terminate_pod(pod_id)
            logger.info(f"Terminated runpod pod {pod_id}")
        except Exception as e:
            logger.error(f"Failed terminating runpod pod {pod_id}: {e}")
        finally:
            LAUNCHED_RUNPOD_IDS.discard(pod_id)

def shutdown_docker_containers():
    """Kill all Docker containers we explicitly launched."""
    logger.info("Shutting down Docker containers from LAUNCHED_DOCKER_CONTAINERS...")
    for container_name in list(LAUNCHED_DOCKER_CONTAINERS):
        try:
            container = docker_client.containers.get(container_name)
            container.kill()
            container.remove(force=True)
            logger.info(f"Removed container {container_name}")
        except Exception as e:
            logger.error(f"Failed removing container {container_name}: {e}")
        finally:
            LAUNCHED_DOCKER_CONTAINERS.discard(container_name)

atexit.register(shutdown_runpod_instances)
atexit.register(shutdown_docker_containers)

# -----------------------------------------------------------------------------
# TEARDOWN HELPERS
# -----------------------------------------------------------------------------
def force_remove_container_by_name(container_name: str):
    """Helper to forcibly remove a Docker container by name."""
    try:
        container = docker_client.containers.get(container_name)
        container.kill()
        container.remove(force=True)
        logger.info(f"Container {container_name} killed and removed.")
    except docker.errors.NotFound:
        logger.warning(f"Container {container_name} not found.")
    except Exception as e:
        logger.error(f"Error removing container {container_name}: {e}")

async def teardown_old_runpod_instance(old_job_id):
    """
    Kills the single runpod pod for old_job_id, if we recorded it in JOB_TO_POD_ID.
    """
    from spl.deploy.cloud_adapters import runpod as cloud_adapter
    old_pod_id = JOB_TO_POD_ID.pop(old_job_id, None)
    if old_pod_id is None:
        # We didn't record any pod for that job
        return
    logger.info(f"Tearing down runpod instance {old_pod_id} for old job {old_job_id}")
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, cloud_adapter.terminate_pod, old_pod_id)
        logger.info(f"Terminated old runpod pod id={old_pod_id}")
        if old_pod_id in LAUNCHED_RUNPOD_IDS:
            LAUNCHED_RUNPOD_IDS.discard(old_pod_id)
    except Exception as e:
        logger.error(f"Failed to terminate runpod pod {old_pod_id}: {e}")

async def teardown_old_docker_container(old_job_id):
    """
    Kills the single Docker container for old_job_id, if we recorded it.
    """
    container_name = JOB_TO_CONTAINER.pop(old_job_id, None)
    if container_name is None:
        # Not recorded
        return
    logger.info(f"Tearing down docker container '{container_name}' for old job {old_job_id}.")
    try:
        force_remove_container_by_name(container_name)
        if container_name in LAUNCHED_DOCKER_CONTAINERS:
            LAUNCHED_DOCKER_CONTAINERS.discard(container_name)
    except Exception as e:
        logger.error(f"Failed to remove docker container {container_name}: {e}")


# -----------------------------------------------------------------------------
# PluginProxy: used to call into the plugin container
# -----------------------------------------------------------------------------
class PluginProxy:
    def __init__(self, host='localhost', port=8000):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}/execute"
        timeout = aiohttp.ClientTimeout(total=None)
        self.session = aiohttp.ClientSession(timeout=timeout)

    async def call_remote(self, function, args=None, kwargs=None):
        payload = {
            'action': 'call_function',
            'function': function,
            'args': serialize_data(args) if args else [],
            'kwargs': serialize_data(kwargs) if kwargs else {}
        }
        headers = {'Content-Type': 'application/json'}
        async with self.session.post(self.base_url, json=payload, headers=headers) as response:
            if response.status != 200:
                text = await response.text()
                logger.error(f"HTTP error {response.status}: {text}")
                raise Exception(f"HTTP error {response.status}: {text}")
            content_type = response.headers.get('Content-Type', '')
            if 'application/octet-stream' in content_type:
                data = await response.read()
                return data
            text = await response.text()
            response_obj = json.loads(text)
            result_serialized = response_obj.get('result')
            if result_serialized is None:
                if 'error' in response_obj:
                    raise Exception(response_obj['error'])
                raise Exception("No 'result' in response.")
            result = deserialize_data(result_serialized)
            if isinstance(result, dict) and 'error' in result:
                raise Exception(result['error'])
            return result

    def __getattr__(self, name):
        async def method(*args, **kwargs):
            return await self.call_remote(name, args=args, kwargs=kwargs)
        return method


# -----------------------------------------------------------------------------
# Wait for container’s plugin_server to come up
# -----------------------------------------------------------------------------
async def wait_for_server(url, timeout=300):
    logger.info(f"Waiting for server at {url} to be ready...")
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < timeout:
            try:
                payload = {'action': 'health_check'}
                headers = {'Content-Type': 'application/json'}
                async with session.post(url, json=payload, headers=headers) as response:
                    raw_text = await response.text()
                    logger.debug(f"Server raw response: {raw_text}")
                    if response.status == 200:
                        response_obj = json.loads(raw_text)
                        result_serialized = response_obj.get('result')
                        if result_serialized:
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


def get_log_file(name):
    logs_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    return os.path.join(logs_dir, f"{name}.log")


# -----------------------------------------------------------------------------
# Primary entrypoint: get_plugin()
# -----------------------------------------------------------------------------
async def get_plugin(
    job_id,
    db_adapter,
    forwarded_port=None,
    sot_id=False,
    cloud: bool = False,
    preserve_containers: bool = False,
    instance_salt: str = None
):
    """
    Acquire a plugin for the given job. If the job_id changed from LAST_JOB_ID,
    kill the old job's container/pod by ID (unless preserve_containers=True).
    
    For local Docker, we've removed the salting so the container name is
    consistent across processes with the same job_id. For runpod (cloud), we
    still allow salt to differentiate multiple pods if needed.
    """
    global LAST_JOB_ID

    # 1) If job changed, tear down old job
    if LAST_JOB_ID is not None and str(LAST_JOB_ID) != str(job_id) and not preserve_containers:
        if cloud:
            await teardown_old_runpod_instance(LAST_JOB_ID)
        else:
            await teardown_old_docker_container(LAST_JOB_ID)

    LAST_JOB_ID = job_id

    # 2) Load job record
    if not sot_id:
        job_record = await db_adapter.get_job(job_id)
    else:
        job_record = await db_adapter.get_job_sot(job_id)
    if not job_record:
        raise ValueError(f"No job found for job_id: {job_id}")

    # 3) Download plugin code
    plugin_id = job_record.plugin_id
    plugin_package_dir = os.path.join(global_plugin_dir, f'plugin_job_{job_id}')
    create_subdirectory(plugin_package_dir)

    plugin_file_name = f'plugin_{plugin_id}.py'
    plugin_path = os.path.join(plugin_package_dir, plugin_file_name)
    plugin_record = await fetch_and_write_plugin_code(plugin_id, db_adapter, plugin_path, sot_id)
    if not plugin_record:
        raise ValueError(f"No plugin found for plugin_id: {plugin_id}")

    setup_plugin_files(plugin_package_dir)
    init_file_path = os.path.join(plugin_package_dir, '__init__.py')
    if not os.path.exists(init_file_path):
        async with aiofiles.open(init_file_path, mode='w') as f:
            await f.write('# Init file for plugin package\n')
        logger.info(f"Created __init__.py at {init_file_path}")

    # 4) Figure out Docker image from the job’s subnet
    if not sot_id:
        subnet_obj = await db_adapter.get_subnet(plugin_record.subnet_id)
    else:
        subnet_obj = await db_adapter.get_subnet_sot(plugin_record.subnet_id, sot_id)
    if not subnet_obj:
        raise ValueError(f"No subnet found for subnet_id: {plugin_record.subnet_id}")

    docker_image = subnet_obj.docker_image
    logger.info(f"Using docker image '{docker_image}' from subnet {subnet_obj.id}")
    await ensure_docker_image(docker_image)

    # 5) If cloud mode => runpod, else => local Docker
    if cloud:
        from spl.deploy.cloud_adapters import runpod as cloud_adapter
        from spl.deploy.cloud_adapters.runpod import (
            launch_instance_and_record_logs,
            BASE_TEMPLATE_ID,
            get_public_ip_and_port,
            get_pod_ssh_ip_port,
        )

        # Check if we already launched a runpod for this job
        old_pod_id = JOB_TO_POD_ID.get(job_id)
        if old_pod_id:
            logger.info(f"Found existing runpod ID={old_pod_id} for job {job_id}; checking if it's running.")
            # We'll see if it still runs. If not, we create a new one.
            pods = await asyncio.get_event_loop().run_in_executor(None, cloud_adapter.get_pods)
            existing_pod = next((p for p in pods if p['id'] == old_pod_id and p['desiredStatus'] == 'RUNNING'), None)
            if existing_pod:
                # Reuse it
                instance_pod_id = old_pod_id
                logger.info(f"Reusing existing runpod pod id={instance_pod_id} for job {job_id}")
                try:
                    public_ip, public_port = await get_public_ip_and_port(instance_pod_id, private_port=8000)
                    server_url = f"http://{public_ip}:{public_port}/execute"
                    if not await wait_for_server(server_url):
                        logger.error("Existing cloud plugin server is not responding.")
                        raise Exception("Cloud plugin server failed.")
                    return PluginProxy(host=public_ip, port=public_port)
                except Exception as e:
                    logger.error(f"Failed reusing existing runpod {instance_pod_id}: {e}")
                    # We'll kill it and proceed to create a new one
                    await teardown_old_runpod_instance(job_id)

        # If we get here, no valid existing pod => launch new
        suffix_str = f"_{instance_salt}" if instance_salt else ""
        current_runpod_name = f"plugin_{job_id}{suffix_str}"
        GPU_TYPE = 'NVIDIA GeForce RTX 4090'
        DOCKER_CMD = "tail -f /dev/null"
        ports = "22/tcp,8000/tcp"
        env = {
            "ENABLE_SSH": "true",
            "PLUGIN_ID": str(plugin_id),
            "PORT": str(8000)
        }
        plugin_log_file = get_log_file(current_runpod_name)

        worker_instance, worker_helpers = await launch_instance_and_record_logs(
            name=current_runpod_name,
            gpu_type=GPU_TYPE,
            image=docker_image,
            gpu_count=1,
            ports=ports,
            log_file=plugin_log_file,
            env=env,
            template_id=BASE_TEMPLATE_ID,
            cmd=DOCKER_CMD
        )
        instance_pod_id = worker_instance['id']
        JOB_TO_POD_ID[job_id] = instance_pod_id
        LAUNCHED_RUNPOD_IDS.add(instance_pod_id)

        # SSH in, upload plugin code
        ssh_ip, ssh_port = await get_pod_ssh_ip_port(instance_pod_id, timeout=300)
        private_key_path = worker_helpers['private_key_path']

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ssh_ip, port=ssh_port, username="root",
                    key_filename=private_key_path, banner_timeout=200)

        upload_dir_recursively(
            ssh,
            plugin_package_dir,
            "/app/plugin_code",
            hostname=ssh_ip,
            port=ssh_port,
            username="root",
            key_filename=private_key_path
        )

        # Start the plugin server in background
        await start_plugin_server_via_ssh(ssh, host_port=8000, plugin_id=plugin_id)
        ssh.close()

        public_ip, public_port = await cloud_adapter.get_public_ip_and_port(instance_pod_id, private_port=8000)
        server_url = f"http://{public_ip}:{public_port}/execute"
        if not await wait_for_server(server_url):
            logger.error("New cloud plugin server failed to start.")
            await teardown_old_runpod_instance(job_id)  # kill it
            raise Exception("Cloud plugin server failed to start.")
        return PluginProxy(host=public_ip, port=public_port)

    else:
        # LOCAL DOCKER MODE -> no salting for container name
        host_port = get_port(job_id)  # e.g. 8000 + job_id mod 1000
        return await setup_docker_container(
            identifier=job_id,
            plugin_package_dir=plugin_package_dir,
            host_port=host_port,
            docker_image=docker_image,
            forwarded_port=forwarded_port,
            plugin_id=plugin_id
        )


def setup_dir():
    if not os.path.exists(global_plugin_dir):
        os.makedirs(global_plugin_dir)
        logger.info(f"Created plugin directory at {global_plugin_dir}")
    if global_plugin_dir not in sys.path:
        sys.path.append(global_plugin_dir)
        logger.info(f"Added {global_plugin_dir} to sys.path")

def create_subdirectory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created subdirectory at {path}")

def copy_if_missing(src, dst):
    if os.path.exists(src) and not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            logger.info(f"Copied directory from {src} to {dst}")
        else:
            shutil.copy(src, dst)
            logger.info(f"Copied file from {src} to {dst}")

async def fetch_and_write_plugin_code(plugin_id, db_adapter, plugin_path, sot_id=None):
    if not sot_id:
        plugin_record = await db_adapter.get_plugin(plugin_id)
    else:
        plugin_record = await db_adapter.get_plugin_sot(plugin_id, sot_id)
    if not plugin_record or not hasattr(plugin_record, 'code'):
        logger.error(f"No plugin code found for plugin_id: {plugin_id}")
        raise ValueError(f"No plugin code found for plugin_id: {plugin_id}")
    plugin_code = plugin_record.code
    async with aiofiles.open(plugin_path, mode='w') as f:
        await f.write(plugin_code)
    logger.info(f"Fetched and wrote plugin code to {plugin_path}")
    return plugin_record

def setup_plugin_files(plugin_package_dir):
    grandparent_dir = os.path.dirname(os.path.dirname(__file__))
    resources = {
        'adapters': 'adapters',
        'datasets': 'datasets',
        'tokenizer.py': 'tokenizer.py',
        'device.py': 'device.py',
        'common.py': 'common.py',
        'plugins/serialize.py': 'serialize.py',
        'requirements.txt': 'requirements.txt',
        'plugins/plugin_server.py': 'server.py',
        'util': 'util',
        'db/db_adapter_client.py': 'db/db_adapter_client.py',
        'models': 'models',
        'auth/key_auth.py': 'auth/key_auth.py',
        'auth/nonce_cache.py': 'auth/nonce_cache.py',
    }
    for local, global_target in resources.items():
        src = os.path.join(grandparent_dir, local)
        dst = os.path.join(plugin_package_dir, global_target)
        copy_if_missing(src, dst)

def get_port(identifier):
    """
    Convert job_id to a port. If it's numeric, we do 8000+job_id. Otherwise,
    we hash it. This ensures each job has a stable port (in local Docker mode).
    """
    try:
        return HOST_PORT_BASE + int(identifier)
    except ValueError:
        # hash fallback for non-integer job IDs
        hash_object = hashlib.sha256(str(identifier).encode())
        hash_int = int(hash_object.hexdigest(), 16)
        return HOST_PORT_BASE + (hash_int % 1000)

async def ensure_docker_image(docker_image):
    logger.info(f"Ensuring Docker image '{docker_image}' is available by pulling from Docker Hub.")
    try:
        try:
            docker_client.images.get(docker_image)
            logger.info(f"Docker image '{docker_image}' already exists locally.")
        except docker.errors.ImageNotFound:
            logger.info(f"Docker image '{docker_image}' not found locally. Pulling...")
            docker_client.images.pull(docker_image)
            logger.info(f"Pulled Docker image '{docker_image}' successfully.")
    except docker.errors.APIError as e:
        logger.error(f"Docker API error during image pull: {e}")
        raise

def is_gpu_available():
    try:
        subprocess.run(["nvidia-smi"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


# -----------------------------------------------------------------------------
# Docker logic
# -----------------------------------------------------------------------------
async def setup_docker_container(identifier,
                                 plugin_package_dir,
                                 host_port,
                                 docker_image,
                                 forwarded_port=None,
                                 plugin_id=None):
    """
    Sets up a local Docker container for the plugin server. 
    **Salting has been removed** for local mode, so the container name is just 
    `panthalia_plugin_{job_id}`, ensuring multiple local processes with 
    the same job_id share the same container.
    """
    # No salt suffix here for local deployment
    container_name = CONTAINER_NAME_TEMPLATE.format(job_id=identifier, suffix="")
    JOB_TO_CONTAINER[identifier] = container_name

    server_url = f"http://localhost:{host_port}/execute"
    gpu_available = is_gpu_available()
    device_requests = None
    if gpu_available:
        device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])]
        logger.info("GPU detected. Requesting GPU device in container.")
    else:
        logger.info("No GPU detected. Running container with CPU only.")

    port_bindings = {f'{host_port}/tcp': host_port}
    if forwarded_port:
        port_bindings[f'{forwarded_port}/tcp'] = forwarded_port
        logger.info(f"Forwarding container port {forwarded_port} to the host.")

    # Attempt to get if container already exists:
    container = None
    try:
        container = docker_client.containers.get(container_name)
        logger.info(f"Container {container_name} already exists with status={container.status}.")
    except docker.errors.NotFound:
        pass

    if container and container.status != 'running':
        container.start()
        logger.info(f"Started existing container {container_name}")
    elif not container:
        # create new container
        logger.info(f"Creating and starting container {container_name}")
        container = docker_client.containers.run(
            docker_image,
            name=container_name,
            detach=True,
            security_opt=security_options,
            ports=port_bindings,
            mem_limit=mem_limit,
            pids_limit=pids_limit,
            volumes={plugin_package_dir: {'bind': docker_plugin_dir, 'mode': 'ro'}},
            environment={
                "PIP_NO_CACHE_DIR": "off",
                "PLUGIN_ID": str(plugin_id),
                "PORT": str(host_port),
                "DOCKER_PLUGIN_DIR": docker_plugin_dir,
                "HOME": "/tmp"
            },
            user="nobody",
            device_requests=device_requests,
            extra_hosts={"host.docker.internal": "host-gateway"}
        )
        logger.info(f"Container {container_name} started with ID: {container.id}")

    LAUNCHED_DOCKER_CONTAINERS.add(container_name)

    if not await wait_for_server(server_url):
        show_container_logs(container)
        logger.error("Exiting due to server startup failure.")
        await teardown_old_docker_container(identifier)
        raise Exception("Docker container server failed to start.")

    plugin_proxy = PluginProxy(host='localhost', port=host_port)
    logger.info(
        f"Plugin proxy created for job_id '{identifier}' with PLUGIN_ID '{plugin_id}' at {server_url}"
    )
    return plugin_proxy


def reconnect_ssh(hostname, port, username, key_filename, pod_id=None):
    """If ephemeral port changed on runpod, re-resolve it. Otherwise just connect."""
    if pod_id:
        logger.info(f"[reconnect_ssh] Cloud mode. Re-resolving ephemeral port for pod_id={pod_id}.")
        new_hostname, new_port = get_pod_ssh_ip_port_sync(pod_id, private_port=22, timeout=300)
        logger.info(f"[reconnect_ssh] Using ephemeral IP={new_hostname}, port={new_port}.")
    else:
        new_hostname, new_port = hostname, port

    new_ssh = paramiko.SSHClient()
    new_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    new_ssh.connect(new_hostname, port=new_port, username=username, key_filename=key_filename, banner_timeout=200)
    transport = new_ssh.get_transport()
    if transport is not None:
        transport.set_keepalive(30)
    return new_ssh


def safe_sftp_put(ssh, local_file, remote_file,
                  hostname=None, port=None,
                  username='root', key_filename=None,
                  pod_id=None):
    """
    If SFTP fails, try reconnecting (especially for ephemeral RunPod ports).
    """
    logger.debug(f"[safe_sftp_put] Uploading {local_file} -> {remote_file}")
    try:
        sftp = ssh.open_sftp()
        sftp.put(local_file, remote_file)
        sftp.close()
    except paramiko.ssh_exception.SSHException as e:
        logger.warning(f"[safe_sftp_put] SSH error while uploading {local_file}: {e}")
        if not hostname or not port or not key_filename:
            logger.error("[safe_sftp_put] Missing connection info, cannot reconnect.")
            raise
        try:
            ssh.close()
            ssh = reconnect_ssh(hostname, port, username, key_filename, pod_id=pod_id)
            sftp = ssh.open_sftp()
            sftp.put(local_file, remote_file)
            sftp.close()
        except Exception as e2:
            logger.error(f"[safe_sftp_put] Reconnect failed: {e2}")
            raise

def upload_dir_recursively(ssh, local_dir, remote_dir,
                           hostname, port, username,
                           key_filename, pod_id=None):
    """Recursively upload local_dir -> remote_dir via SFTP, with reconnect logic."""
    sftp = ssh.open_sftp()
    try:
        sftp.stat(remote_dir)
    except IOError:
        sftp.mkdir(remote_dir)
    sftp.close()

    for root, dirs, files in os.walk(local_dir):
        rel_path = os.path.relpath(root, local_dir)
        remote_subdir = remote_dir
        if rel_path != ".":
            remote_subdir = os.path.join(remote_dir, rel_path)

        sftp = ssh.open_sftp()
        try:
            sftp.stat(remote_subdir)
        except IOError:
            sftp.mkdir(remote_subdir)
        sftp.close()

        for fname in files:
            local_file = os.path.join(root, fname)
            remote_file = os.path.join(remote_subdir, fname)
            safe_sftp_put(
                ssh,
                local_file=local_file,
                remote_file=remote_file,
                hostname=hostname,
                port=port,
                username=username,
                key_filename=key_filename,
                pod_id=pod_id
            )


async def start_plugin_server_via_ssh(ssh, host_port=8000, plugin_id=None):
    """
    SSH command to kill any old plugin_code.server, then start a new one.
    """
    kill_cmd = "pkill -f plugin_code.server || true"
    logger.info(f"Stopping any existing plugin_server: {kill_cmd}")
    ssh.exec_command(kill_cmd)
    if not plugin_id:
        plugin_id = os.environ.get('PLUGIN_ID', '')
    start_cmd = (
        f"mkdir -p /app/plugin_code && "
        f"cd /app && export PORT={host_port} && export PLUGIN_ID={plugin_id} "
        f"&& nohup /venv/bin/python -m plugin_code.server > /tmp/plugin_server.log 2>&1 &"
    )
    logger.info(f"Starting plugin_server in background: {start_cmd}")
    ssh.exec_command(start_cmd)
