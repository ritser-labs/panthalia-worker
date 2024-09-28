import time
import runpod
import threading
import paramiko
import logging
import os
from .runpod_config import RUNPOD_API_KEY
from ..util import is_port_open
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

runpod.api_key = RUNPOD_API_KEY
INPUT_JSON_PATH = '/tmp/input.json'

# Define a ThreadPoolExecutor for handling blocking calls
executor = ThreadPoolExecutor()

async def generate_ssh_key_pair():
    """
    Generates an SSH key pair using Paramiko and saves it locally if it doesn't already exist.

    Returns:
        str: The path to the private key file.
        str: The public key string.
    """
    private_key_path = os.path.expanduser("~/.ssh/id_rsa_runpod")

    # Check if the private key file already exists
    if os.path.exists(private_key_path):
        logging.info(f"SSH private key already exists at {private_key_path}, skipping generation.")

        # Load the existing key
        key = paramiko.RSAKey(filename=private_key_path)
        public_key_str = f"{key.get_name()} {key.get_base64()}"
        return private_key_path, public_key_str

    # Generate a new SSH key pair
    key = paramiko.RSAKey.generate(2048)
    public_key_str = f"{key.get_name()} {key.get_base64()}"

    # Save the private key
    with open(private_key_path, "w") as private_key_file:
        key.write_private_key(private_key_file)

    # Set appropriate permissions for the private key file
    os.chmod(private_key_path, 0o600)

    logging.info(f"Generated new SSH private key at {private_key_path}")
    return private_key_path, public_key_str

def generate_deploy_cpu_pod_mutation(
        name,
        image,
        instance_id,
        env={},
        ports=None,
        template_id=None,
        container_disk_in_gb=10,
        support_public_ip=True,
        cloud_type='SECURE'
    ):
    """
    Generates a GraphQL mutation string for deploying a CPU pod with detailed configurations.

    Args:
        name (str): The name of the pod.
        image (str): The Docker image to use.
        instance_id (str): The CPU instance ID.
        env (dict): Environment variables.
        ports (str or None): Ports to expose, e.g., "8888/http,666/tcp".
        template_id (str or None): Template ID for pod deployment.
        container_disk_in_gb (int): Disk size for the container.
        support_public_ip (bool): Whether to support a public IP.
        cloud_type (str): Cloud type, e.g., "SECURE".
        min_download (int): Minimum download bandwidth.
        min_upload (int): Minimum upload bandwidth.
        min_disk (int): Minimum disk size.
        min_memory_in_gb (int): Minimum memory in GB.
        min_vcpu_count (int): Minimum vCPU count.

    Returns:
        str: The GraphQL mutation string.
    """
    input_fields = []

    # Required Fields
    input_fields.append(f'name: "{name}"')
    if image is not None:
        input_fields.append(f'image: "{image}"')
    input_fields.append(f'instanceId: "{instance_id}"')
    input_fields.append(f'cloudType: {cloud_type}')
    if env is not None:
        env_string = ", ".join(
            [f'{{ key: "{key}", value: "{value}" }}' for key, value in env.items()])
        input_fields.append(f"env: [{env_string}]")
    if ports is not None:
        ports = ports.replace(' ', '')
        input_fields.append(f'ports: "{ports}"')
    if template_id is not None:
        input_fields.append(f'templateId: "{template_id}"')
    if container_disk_in_gb is not None:
        input_fields.append(f'containerDiskInGb: {container_disk_in_gb}')
    if support_public_ip is not None:
        input_fields.append(f'supportPublicIp: {str(support_public_ip).lower()}')

    input_string = ', '.join(input_fields)
    return f"""
    mutation {{
        deployCpuPod(input: {{
            {input_string}
        }}) {{
            id
            desiredStatus
            imageName
            env
            machineId
            machine {{
            podHostId
            }}
        }}
    }}
    """
    return mutation

def generate_pod_deploy_mutation(
        name,
        image,
        gpu_type_id,
        gpu_count,
        support_public_ip,
        ports,
        env,
        template_id,
        container_disk_in_gb,
        cloud_type,
        min_download,
        min_upload,
        min_disk,
        min_memory_in_gb,
        min_vcpu_count,
        instance_ids=None,
        start_ssh: bool = True,
        volume_mount_path='/runpod-volume',
        volume_in_gb=0
    ):
    """
    Generates a GraphQL mutation string for deploying a GPU pod with detailed configurations.

    Args:
        name (str): The name of the pod.
        image (str): The Docker image to use.
        gpu_type_id (str): The GPU type ID.
        gpu_count (int): Number of GPUs.
        support_public_ip (bool): Whether to support a public IP.
        ports (str or None): Ports to expose, e.g., "8888/http,666/tcp".
        env (dict): Environment variables.
        template_id (str or None): Template ID for pod deployment.
        container_disk_in_gb (int): Disk size for the container.
        cloud_type (str): Cloud type, e.g., "SECURE".
        min_download (int): Minimum download bandwidth.
        min_upload (int): Minimum upload bandwidth.
        min_disk (int): Minimum disk size.
        min_memory_in_gb (int): Minimum memory in GB.
        min_vcpu_count (int): Minimum vCPU count.
        instance_ids (list or None): List of CPU instance IDs (if applicable).

    Returns:
        str: The GraphQL mutation string.
    """
    input_fields = []

    # Required Fields
    input_fields.append(f'name: "{name}"')
    input_fields.append(f'imageName: "{image}"')
    input_fields.append(f'gpuTypeId: "{gpu_type_id}"')
    input_fields.append(f'cloudType: {cloud_type}')
    if start_ssh:
        input_fields.append('startSsh: true')
    input_fields.append(f'supportPublicIp: {str(support_public_ip).lower()}')
    input_fields.append(f'gpuCount: {gpu_count}')
    if volume_in_gb is not None:
        input_fields.append(f"volumeInGb: {volume_in_gb}")

    # Container Disk
    if container_disk_in_gb is not None:
        input_fields.append(f"containerDiskInGb: {container_disk_in_gb}")
    # Ports
    if ports:
        ports = ports.replace(" ", "")
        input_fields.append(f'ports: "{ports}"')
    if volume_mount_path is not None:
        input_fields.append(f'volumeMountPath: "{volume_mount_path}"')


    # Environment Variables
    if env is not None:
        env_string = ", ".join(
            [f'{{ key: "{key}", value: "{value}" }}' for key, value in env.items()])
        input_fields.append(f"env: [{env_string}]")

    # Template ID
    if template_id:
        input_fields.append(f'templateId: "{template_id}"')

    # Instance IDs (if any)
    if instance_ids:
        instance_ids_str = ", ".join([f'"{iid}"' for iid in instance_ids])
        input_fields.append(f'instanceIds: [{instance_ids_str}]')
    
    # Minimum Requirements
    if min_download is not None:
        input_fields.append(f"minDownload: {min_download}")
    if min_upload is not None:
        input_fields.append(f"minUpload: {min_upload}")
    if min_disk is not None:
        input_fields.append(f"minDisk: {min_disk}")
    if min_memory_in_gb is not None:
        input_fields.append(f"minMemoryInGb: {min_memory_in_gb}")
    if min_vcpu_count is not None:
        input_fields.append(f"minVcpuCount: {min_vcpu_count}")

    # Assemble the input string
    input_string = ", ".join(input_fields)

    mutation = f"""
    mutation {{
      podFindAndDeployOnDemand(
        input: {{
          {input_string}
        }}
      ) {{
        id
        desiredStatus
        imageName
        env
        machineId
        machine {{
          podHostId
        }}
      }}
    }}
    """
    return mutation

async def create_cpu_pod(
    name, image, instance_id, env={}, ports=None, template_id=None,
    container_disk_in_gb=10, support_public_ip=True, cloud_type='SECURE'
):
    """
    Creates a CPU pod by sending the deployCpuPod mutation directly.

    Args:
        name (str): The name of the pod.
        image (str): The Docker image to use.
        instance_id (str): The CPU instance ID.
        env (dict, optional): Environment variables. Defaults to {}.
        ports (str, optional): Ports to expose. Defaults to None.
        template_id (str, optional): Template ID for pod deployment. Defaults to None.
        container_disk_in_gb (int, optional): Disk size for the container. Defaults to 10.
        support_public_ip (bool, optional): Whether to support a public IP. Defaults to True.
        cloud_type (str, optional): Cloud type. Defaults to 'SECURE'.
        min_download (int, optional): Minimum download bandwidth. Defaults to 100.
        min_upload (int, optional): Minimum upload bandwidth. Defaults to 100.
        min_disk (int, optional): Minimum disk size. Defaults to 50.
        min_memory_in_gb (int, optional): Minimum memory in GB. Defaults to 8.
        min_vcpu_count (int, optional): Minimum vCPU count. Defaults to 2.

    Returns:
        dict: The newly created pod information.

    Raises:
        Exception: If the pod deployment fails.
    """
    private_key_path, public_key_str = await generate_ssh_key_pair()
    env = env.copy()  # Avoid mutating the original env
    env['PUBLIC_KEY'] = public_key_str  # Add public key to environment variables

    # Generate the GraphQL mutation for CPU pod
    mutation = generate_deploy_cpu_pod_mutation(
        name=name,
        image=image,
        instance_id=instance_id,
        env=env,
        ports=ports,
        template_id=template_id,
        container_disk_in_gb=container_disk_in_gb,
        support_public_ip=support_public_ip,
        cloud_type=cloud_type
    )

    logging.info(f"Deploying CPU pod '{name}' with mutation:\n{mutation}")

    try:
        # Send the mutation using runpod's GraphQL API
        raw_response = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: runpod.api.graphql.run_graphql_query(mutation)
        )

        # Check for errors in the response
        if 'errors' in raw_response:
            error_messages = "; ".join([error['message'] for error in raw_response['errors']])
            raise Exception(f"GraphQL Error: {error_messages}")

        # Extract the pod information from the response
        new_pod = raw_response["data"]["deployCpuPod"]
        pod_id = new_pod['id']
        logging.info(f"CPU Pod '{name}' created with ID: {pod_id}")

        return new_pod, private_key_path

    except Exception as e:
        logging.error(f"Failed to deploy CPU pod '{name}': {e}")
        # Optionally, terminate the pod if it was partially created
        if 'pod_id' in locals():
            try:
                runpod.terminate_pod(pod_id)
                logging.info(f"Terminated CPU pod '{name}' due to deployment failure.")
            except Exception as terminate_error:
                logging.error(f"Failed to terminate CPU pod '{name}': {terminate_error}")
        raise e

async def create_gpu_pod(
    name, image, gpu_type_id, gpu_count, env={}, ports=None, template_id=None,
    container_disk_in_gb=10, support_public_ip=True, cloud_type='SECURE',
    min_download=100, min_upload=100, min_disk=50, min_memory_in_gb=8, min_vcpu_count=2,
    volume_in_gb=0
):
    """
    Creates a GPU pod by sending the podFindAndDeployOnDemand mutation directly.

    Args:
        name (str): The name of the pod.
        image (str): The Docker image to use.
        gpu_type_id (str): The GPU type ID.
        gpu_count (int): Number of GPUs.
        env (dict, optional): Environment variables. Defaults to {}.
        ports (str, optional): Ports to expose. Defaults to None.
        template_id (str, optional): Template ID for pod deployment. Defaults to None.
        container_disk_in_gb (int, optional): Disk size for the container. Defaults to 10.
        support_public_ip (bool, optional): Whether to support a public IP. Defaults to True.
        cloud_type (str, optional): Cloud type. Defaults to 'COMMUNITY'.
        min_download (int, optional): Minimum download bandwidth. Defaults to 100.
        min_upload (int, optional): Minimum upload bandwidth. Defaults to 100.
        min_disk (int, optional): Minimum disk size. Defaults to 20.
        min_memory_in_gb (int, optional): Minimum memory in GB. Defaults to 8.
        min_vcpu_count (int, optional): Minimum vCPU count. Defaults to 2.

    Returns:
        dict: The newly created pod information.

    Raises:
        Exception: If the pod deployment fails.
    """
    private_key_path, public_key_str = await generate_ssh_key_pair()
    env = env.copy()  # Avoid mutating the original env
    env['PUBLIC_KEY'] = public_key_str  # Add public key to environment variables
    
    runpod.api.ctl_commands.get_gpu(gpu_type_id)
    
    if cloud_type not in ['ALL', 'SECURE', 'COMMUNITY']:
        raise ValueError(f"Invalid cloud type: {cloud_type}")
    '''
    sample_mutation = runpod.api.mutations.pods.generate_pod_deployment_mutation(
        name=name,
        image_name=image,
        gpu_type_id=gpu_type_id,
        gpu_count=gpu_count,
        support_public_ip=support_public_ip,
        ports=ports,
        env=env,
        template_id=template_id,
        container_disk_in_gb=container_disk_in_gb,
        cloud_type=cloud_type,
        volume_mount_path='/runpod-volume',
        volume_in_gb=volume_in_gb,
    )
    
    print(f'Sample mutation: {sample_mutation}')
    '''

    # Generate the GraphQL mutation for GPU pod
    mutation = generate_pod_deploy_mutation(
        name=name,
        image=image,
        gpu_type_id=gpu_type_id,
        gpu_count=gpu_count,
        support_public_ip=support_public_ip,
        ports=ports,
        env=env,
        template_id=template_id,
        container_disk_in_gb=container_disk_in_gb,
        cloud_type=cloud_type,
        min_download=min_download,
        min_upload=min_upload,
        min_disk=min_disk,
        min_memory_in_gb=min_memory_in_gb,
        min_vcpu_count=min_vcpu_count
    )

    logging.info(f"Deploying GPU pod '{name}' with mutation:\n{mutation}")

    try:
        # Send the mutation using runpod's GraphQL API
        raw_response = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: runpod.api.graphql.run_graphql_query(mutation)
        )

        # Check for errors in the response
        if 'errors' in raw_response:
            error_messages = "; ".join([error['message'] for error in raw_response['errors']])
            raise Exception(f"GraphQL Error: {error_messages}")

        # Extract the pod information from the response
        new_pod = raw_response["data"]["podFindAndDeployOnDemand"]
        pod_id = new_pod['id']
        logging.info(f"GPU Pod '{name}' created with ID: {pod_id}")

        return new_pod, private_key_path

    except Exception as e:
        logging.error(f"Failed to deploy GPU pod '{name}': {e}")
        # Optionally, terminate the pod if it was partially created
        if 'pod_id' in locals():
            try:
                runpod.terminate_pod(pod_id)
                logging.info(f"Terminated GPU pod '{name}' due to deployment failure.")
            except Exception as terminate_error:
                logging.error(f"Failed to terminate GPU pod '{name}': {terminate_error}")
        raise e

async def create_new_pod(
    name: str,
    image: str,
    gpu_type: Optional[str] = None,
    cpu_type: Optional[str] = None,
    gpu_count: int = 0,
    support_public_ip: bool = True,
    ports: str = '',
    env: dict = {},
    template_id: Optional[str] = None,
    container_disk_in_gb: int = 10,
    min_download: int = 100,      # Example default value
    min_upload: int = 100,        # Example default value
    min_disk: int = 50,           # Example default value
    min_memory_in_gb: int = 8,    # Example default value
    min_vcpu_count: int = 2,      # Example default value
    cloud_type: str = 'SECURE',
    volume_in_gb: int = 0
):
    """
    Creates a new pod by directly sending the appropriate GraphQL mutation to RunPod's API.

    This function determines whether to create a CPU or GPU pod based on the provided arguments
    and sends the corresponding mutation with all necessary parameters.

    Args:
        name (str): The name of the pod.
        image (str): The Docker image to use.
        gpu_type (str, optional): The GPU type ID. If None, a CPU pod is created.
        cpu_type (str, optional): The CPU instance ID for CPU pods.
        gpu_count (int, optional): Number of GPUs. Defaults to 0.
        support_public_ip (bool, optional): Whether to support a public IP. Defaults to True.
        ports (str, optional): Ports to expose, e.g., "8888/http,666/tcp". Defaults to ''.
        env (dict, optional): Environment variables. Defaults to {}.
        template_id (str, optional): Template ID for pod deployment. Defaults to None.
        container_disk_in_gb (int, optional): Disk size for the container. Defaults to 10.
        min_download (int, optional): Minimum download bandwidth. Defaults to 100.
        min_upload (int, optional): Minimum upload bandwidth. Defaults to 100.
        min_disk (int, optional): Minimum disk size. Defaults to 50.
        min_memory_in_gb (int, optional): Minimum memory in GB. Defaults to 8.
        min_vcpu_count (int, optional): Minimum vCPU count. Defaults to 2.
        cloud_type (str, optional): Cloud type, e.g., "SECURE". Defaults to 'SECURE'.

    Returns:
        tuple: A tuple containing the new pod information and the path to the private SSH key.

    Raises:
        ValueError: If required parameters for CPU pods are missing.
        Exception: If the pod deployment fails.
    """
    if gpu_type and gpu_count > 0:
        # GPU Pod
        logging.info(f"Creating GPU pod '{name}' with GPU type '{gpu_type}' and count {gpu_count}.")
        new_pod, private_key_path = await create_gpu_pod(
            name=name,
            image=image,
            gpu_type_id=gpu_type,
            gpu_count=gpu_count,
            env=env,
            ports=ports,
            template_id=template_id,
            container_disk_in_gb=container_disk_in_gb,
            support_public_ip=support_public_ip,
            cloud_type=cloud_type,
            min_download=min_download,
            min_upload=min_upload,
            min_disk=min_disk,
            min_memory_in_gb=min_memory_in_gb,
            min_vcpu_count=min_vcpu_count,
            volume_in_gb=volume_in_gb
        )
    else:
        # CPU Pod
        if not cpu_type:
            cpu_type = 'cpu3c-2-4'
        logging.info(f"Creating CPU pod '{name}' with CPU instance '{cpu_type}'.")
        new_pod, private_key_path = await create_cpu_pod(
            name=name,
            image=image,
            instance_id=cpu_type,
            env=env,
            ports=ports,
            template_id=template_id,
            container_disk_in_gb=container_disk_in_gb,
            support_public_ip=support_public_ip,
            cloud_type=cloud_type,
        )

    return new_pod, private_key_path

async def get_pod_ssh_ip_port(pod_id, timeout=300):
    """
    Returns the IP and port for SSH access to a pod by calling the generic function.

    Args:
        pod_id (str): The ID of the pod.
        timeout (int): Timeout in seconds to wait for the pod to be ready.

    Returns:
        tuple: A tuple containing the IP and port for SSH access.
    """
    return await get_public_ip_and_port(pod_id, private_port=22, timeout=timeout)

def generate_deploy_pod_mutation(
        name,
        image,
        gpu_type_id,
        gpu_count,
        support_public_ip,
        ports,
        env,
        template_id,
        container_disk_in_gb,
        cloud_type,
        min_download,
        min_upload,
        min_disk,
        min_memory_in_gb,
        min_vcpu_count,
        instance_ids=None
    ):
    """
    Generates a GraphQL mutation string for deploying a GPU pod with detailed configurations.

    Args:
        name (str): The name of the pod.
        image (str): The Docker image to use.
        gpu_type_id (str): The GPU type ID.
        gpu_count (int): Number of GPUs.
        support_public_ip (bool): Whether to support a public IP.
        ports (str or None): Ports to expose, e.g., "8888/http,666/tcp".
        env (dict): Environment variables.
        template_id (str or None): Template ID for pod deployment.
        container_disk_in_gb (int): Disk size for the container.
        cloud_type (str): Cloud type, e.g., "SECURE".
        min_download (int): Minimum download bandwidth.
        min_upload (int): Minimum upload bandwidth.
        min_disk (int): Minimum disk size.
        min_memory_in_gb (int): Minimum memory in GB.
        min_vcpu_count (int): Minimum vCPU count.
        instance_ids (list or None): List of CPU instance IDs (if applicable).

    Returns:
        str: The GraphQL mutation string.
    """
    input_fields = []

    # Required Fields
    input_fields.append(f'name: "{name}"')
    input_fields.append(f'imageName: "{image}"')
    input_fields.append(f'gpuTypeId: "{gpu_type_id}"')
    input_fields.append(f'gpuCount: {gpu_count}')
    input_fields.append(f'cloudType: "{cloud_type}"')
    input_fields.append(f'supportPublicIp: {str(support_public_ip).lower()}')

    # Environment Variables
    if env:
        env_string = ", ".join(
            [f'{{ key: "{key}", value: "{value}" }}' for key, value in env.items()]
        )
        input_fields.append(f"env: [{env_string}]")

    # Ports
    if ports:
        ports = ports.replace(" ", "")
        input_fields.append(f'ports: "{ports}"')

    # Template ID
    if template_id:
        input_fields.append(f'templateId: "{template_id}"')

    # Container Disk
    if container_disk_in_gb is not None:
        input_fields.append(f"containerDiskInGb: {container_disk_in_gb}")

    # Instance IDs (if any)
    if instance_ids:
        instance_ids_str = ", ".join([f'"{iid}"' for iid in instance_ids])
        input_fields.append(f'instanceIds: [{instance_ids_str}]')

    # Minimum Requirements
    input_fields.append(f"minDownload: {min_download}")
    input_fields.append(f"minUpload: {min_upload}")
    input_fields.append(f"minDisk: {min_disk}")
    input_fields.append(f"minMemoryInGb: {min_memory_in_gb}")
    input_fields.append(f"minVcpuCount: {min_vcpu_count}")

    # Assemble the input string
    input_string = ",\n          ".join(input_fields)

    mutation = f"""
    mutation {{
      podFindAndDeployOnDemand(
        input: {{
          {input_string}
        }}
      ) {{
        id
        desiredStatus
        imageName
        env {{
          key
          value
        }}
        machineId
        machine {{
          podHostId
        }}
      }}
    }}
    """
    return mutation

async def get_public_ip_and_port(pod_id, private_port, timeout=300):
    """
    Generic function to get the public IP and port for a given private port of a pod.

    Args:
        pod_id (str): The ID of the pod.
        private_port (int): The private port for which the public IP and port is needed.
        timeout (int): Timeout in seconds to wait for the pod to be ready.

    Returns:
        tuple: A tuple containing (public IP, public port) or raises TimeoutError.
    """
    start_time = time.time()
    pod_ip = None
    pod_port = None

    while time.time() - start_time < timeout and (pod_ip is None or pod_port is None):
        # Run the blocking pod retrieval in a thread
        pod = await asyncio.get_event_loop().run_in_executor(executor, lambda: runpod.get_pod(pod_id))
        desired_status = pod.get('desiredStatus', None)
        runtime = pod.get('runtime', None)

        if desired_status == 'RUNNING' and runtime and 'ports' in runtime and runtime['ports'] is not None:
            for port in runtime['ports']:
                if port['privatePort'] == private_port:
                    pod_ip = port['ip']
                    pod_port = int(port['publicPort'])
                    logging.info(f"Pod {pod_id} has IP {pod_ip} and port {pod_port} for private port {private_port}")
                    break

        await asyncio.sleep(1)

    if desired_status != 'RUNNING':
        raise TimeoutError(f"Pod {pod_id} did not reach 'RUNNING' state within {timeout} seconds.")

    if runtime is None:
        raise TimeoutError(f"Pod {pod_id} did not report runtime data within {timeout} seconds.")

    if pod_ip is None or pod_port is None:
        raise TimeoutError(f"Pod {pod_id} did not provide the IP and port for private port {private_port} within {timeout} seconds.")

    return pod_ip, pod_port

def terminate_all_pods():
    """
    Terminates all pods that are currently running.
    """
    pods = runpod.get_pods()
    for pod in pods:
        pod_id = pod['id']
        runpod.terminate_pod(pod_id)
    logging.info("All pods have been terminated.")

async def stream_handler(stream, log_file):
    """
    Handles streaming of logs to a file in real-time asynchronously.

    Args:
        stream (paramiko.ChannelFile): The stream to read from (stdout or stderr).
        log_file (file): The file object to write logs to.
    """
    while True:
        line = await asyncio.get_event_loop().run_in_executor(executor, stream.readline)
        if not line:
            break
        log_file.write(line)
        log_file.flush()  # Ensure logs are written immediately

async def copy_file_from_remote(ssh, remote_path, local_path, interval=0.1, copy_mode='stream'):
    """
    Asynchronous function to copy a file from a remote server to the local system using Paramiko over SSH.
    It can either stream incremental updates (for logs) or copy the entire file (for images) atomically.

    Args:
        ssh (paramiko.SSHClient): The active SSH connection.
        remote_path (str): The path to the file on the remote server.
        local_path (str): The path to the file on the local system.
        interval (float): The time in seconds between each copy attempt.
        copy_mode (str): The mode of copying: 'stream' for incremental updates (logs) or 'full' for entire file (images).
    """
    sftp = ssh.open_sftp()
    last_size = 0  # Track the last known size of the remote file (only for streaming mode)

    try:
        while True:
            try:
                # Get the current file size
                file_stat = await asyncio.get_event_loop().run_in_executor(executor, sftp.stat, remote_path)
                current_size = file_stat.st_size

                if copy_mode == 'stream':
                    # Stream updates: Copy only new content
                    if current_size > last_size:
                        # Open the remote file and seek to the last position
                        with sftp.file(remote_path, 'r') as remote_file:
                            remote_file.seek(last_size)

                            # Read the new content from the remote file
                            new_content = await asyncio.get_event_loop().run_in_executor(executor, remote_file.read, current_size - last_size)

                            # Check if any new content was read
                            if new_content:
                                # Decode the content from bytes to a string (assuming it's text data, such as logs)
                                new_content_str = new_content.decode('utf-8')

                                # Append the new content to the local file
                                with open(local_path, 'a') as local_file:
                                    local_file.write(new_content_str)

                                # Update the last known size of the file
                                last_size = current_size

                    else:
                        # No new updates in the remote file
                        pass

                elif copy_mode == 'full':
                    # Full file copy: Copy the entire file atomically
                    logging.info(f"Copying the entire file {remote_path} to {local_path} atomically...")

                    # Generate a temporary file name on the remote side
                    remote_temp_path = remote_path + ".tmp"

                    # Copy the remote file to a temporary file on the remote side
                    copy_command = f"cp {remote_path} {remote_temp_path}"
                    stdin, stdout, stderr = await async_exec_command(ssh, copy_command)
                    exit_status = await asyncio.get_event_loop().run_in_executor(executor, stdout.channel.recv_exit_status)

                    if exit_status == 0:
                        # Generate a temporary file name on the local side
                        local_temp_path = local_path + ".tmp"

                        # Perform SFTP get operation to download the temp file to a local temp file
                        await asyncio.get_event_loop().run_in_executor(executor, sftp.get, remote_temp_path, local_temp_path)

                        # Perform the atomic rename operation locally
                        await asyncio.get_event_loop().run_in_executor(executor, os.rename, local_temp_path, local_path)

                        # Clean up the temporary remote file
                        await asyncio.get_event_loop().run_in_executor(executor, sftp.remove, remote_temp_path)
                        logging.info(f"Remote temp file {remote_temp_path} deleted.")

                    else:
                        logging.error(f"Failed to create remote copy: {stderr.read().decode()}")

                else:
                    logging.error(f"Invalid copy mode: {copy_mode}")
                    break

            except FileNotFoundError:
                logging.info(f"File {remote_path} not found. Retrying...")
            except Exception as e:
                logging.error(f"Error checking or copying file: {e}")

            await asyncio.sleep(interval)

    finally:
        # Close the SFTP connection to clean up resources
        await asyncio.get_event_loop().run_in_executor(executor, sftp.close)

async def async_exec_command(ssh, command):
    """
    Run the SSH command asynchronously using an executor and log the exit status and outputs.
    
    Args:
        ssh (paramiko.SSHClient): The active SSH connection.
        command (str): The command to execute.

    Returns:
        tuple: stdin, stdout, stderr with command outputs.
    """
    logging.info(f"Executing SSH command: {command}")
    loop = asyncio.get_event_loop()
    
    # Execute the command
    stdin, stdout, stderr = await loop.run_in_executor(None, ssh.exec_command, command)

    # Wait for the command to complete and get the exit status
    exit_status = await loop.run_in_executor(None, stdout.channel.recv_exit_status)

    # Read the outputs from stdout and stderr
    stdout_output = await loop.run_in_executor(None, stdout.read)
    stderr_output = await loop.run_in_executor(None, stderr.read)

    # Log the results
    if exit_status == 0:
        logging.info(f"Command succeeded with exit status {exit_status}")
        logging.info(f"STDOUT: {stdout_output.decode().strip()}")
    else:
        logging.error(f"Command failed with exit status {exit_status}")
        logging.error(f"STDERR: {stderr_output.decode().strip()}")

    return stdin, stdout, stderr

async def launch_instance_and_record_logs(
    name,
    image=None,
    gpu_type=None,
    cpu_type=None,
    gpu_count=0,
    ports='',
    log_file="instance_logs.txt",
    timeout=300,
    env={},
    cmd='tail -f /var/log/syslog',
    template_id=None,
    container_disk_in_gb=10,
    input_jsons=[],
    min_download=800,
    min_upload=800,
    min_disk=None,
    min_memory_in_gb=1,
    min_vcpu_count=1,
    volume_in_gb=0,
    cloud_type='SECURE'
):
    """
    Launches a new instance, waits for it to be ready, SSH into it, and records the logs.

    Args:
        name (str): The name of the pod.
        image (str): The Docker image to be used.
        gpu_type (str): The type of GPU required.
        cpu_type (str): The type of CPU required.
        gpu_count (int): The number of GPUs required.
        ports (str): A string representing ports to be exposed, e.g., "8888/http,666/tcp".
        log_file (str): The file where logs will be recorded.
        timeout (int): Timeout in seconds to wait for the pod to be ready.
        env (dict): Environment variables to set on the remote instance.
        cmd (str): Command to execute on the remote instance.
        template_id (str): Template ID for pod deployment.
        container_disk_in_gb (int): Disk size for the container.
        input_jsons (list): List of JSON objects to upload to the remote instance.

    Returns:
        tuple: The new pod information and helper objects.
    """

    if ports:
        ports += ",22/tcp"  # Ensure SSH port is exposed
    else:
        ports = '22/tcp'

    new_pod, private_key_path = await create_new_pod(
        name=name,
        image=image,
        gpu_type=gpu_type,
        cpu_type=cpu_type,
        gpu_count=gpu_count,
        support_public_ip=True,
        ports=ports,
        env={},  # we will use our env vars for the cmd not initial env vars
        template_id=template_id,
        container_disk_in_gb=container_disk_in_gb,
        min_download=min_download,
        min_upload=min_upload,
        min_disk=min_disk,
        min_memory_in_gb=min_memory_in_gb,
        min_vcpu_count=min_vcpu_count,
        volume_in_gb=volume_in_gb,
        cloud_type=cloud_type
    )
    pod_id = new_pod['id']
    logging.info(f'Pod {name} created with id {pod_id}')
    ssh = None
    try:
        # Step 2: Get the public IP and SSH port
        ssh_ip, ssh_port = await get_pod_ssh_ip_port(pod_id, timeout=timeout)

        while not is_port_open(ssh_ip, ssh_port):
            await asyncio.sleep(1)

        # Step 3: SSH into the pod
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Use the generated key file
        ssh.connect(ssh_ip, port=ssh_port, username="root", key_filename=private_key_path, banner_timeout=200)

        # Prepare environment variables to be set in the remote session
        env_export = " ".join([f'{key}="{value}"' for key, value in env.items()])

        # Create a temporary file on the remote host
        sftp = ssh.open_sftp()
        for i, input_json in enumerate(input_jsons):
            with sftp.file(INPUT_JSON_PATH + f'_{i}', 'w') as remote_file:
                json.dump(input_json, remote_file)
        remote_path = '/run_service.sh'
        with sftp.file(remote_path, 'w') as remote_file:
            remote_file.write(f'export {env_export}\n{cmd}')
        sftp.chmod(remote_path, 0o755)  # Make it executable

        remote_log_path = f"/panthalia.log"
        
        setup_cmd = f'chmod +x {remote_path} && apt-get update && apt-get install -y tmux'
        
        _, _, stderr = await async_exec_command(ssh, setup_cmd)
        
        script_cmd = f'tmux new-session -d -s mysession "/bin/bash {remote_path} > {remote_log_path} 2>&1"'
        
        stdin, stdout, stderr = await async_exec_command(ssh, script_cmd)

        # Function to check if SSH session is still alive
        def is_ssh_session_alive():
            """
            Checks if the SSH session is still alive.

            Returns:
                bool: True if the session is alive, False otherwise.
            """
            transport = ssh.get_transport()  # Get the transport associated with this session
            return transport is not None and transport.is_active()

        # Add the is_ssh_session_alive function to pod_helpers
        pod_helpers = {
            'private_key_path': private_key_path,
            'log': open(log_file, "a"),  # Append mode to preserve logs
            'is_ssh_session_alive': is_ssh_session_alive
        }

        # Step 6: Create a new task to copy the log file from remote to local every second
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        local_log_path = os.path.join(logs_dir, f"{name}.log")

        pod_helpers['sftp_task'] = asyncio.create_task(copy_file_from_remote(ssh, remote_log_path, local_log_path))

        remote_loss_path = f"/app/spl/loss_plot.png"
        local_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "loss_plot.png")
        if name == 'master':
            pod_helpers['loss_task'] = asyncio.create_task(copy_file_from_remote(ssh, remote_loss_path, local_file_path, interval=5, copy_mode='full'))
        pod_helpers['sftp'] = sftp
        pod_helpers['ssh'] = ssh

        return new_pod, pod_helpers

    except Exception as e:
        runpod.terminate_pod(pod_id)
        logging.error(f"Error launching instance {name}: {e}")
        raise e

async def reconnect_and_initialize_existing_pod(pod_id, name, private_key_path, log_file="instance_logs.txt", remote_log_path="/panthalia.log", remote_loss_path="/app/spl/loss_plot.png"):
    """
    Reconnects to an existing pod and sets up necessary threads to handle file copying and SSH connection management.
    
    Args:
        pod_id (str): The ID of the pod.
        name (str): The name of the pod (used for logging purposes).
        private_key_path (str): The path to the private key for SSH connection.
        log_file (str): Path to the log file where local logs will be stored.
        remote_log_path (str): Path to the remote log file on the instance.
        remote_loss_path (str): Path to the remote loss file for master instance.
    
    Returns:
        dict: A dictionary of helper objects, including the SSH and SFTP clients and threads for log copying.
    """
    logging.info(f"Reconnecting to pod {pod_id}...")
    
    # Get the public IP and SSH port of the pod
    ssh_ip, ssh_port = await get_pod_ssh_ip_port(pod_id)
    
    while not is_port_open(ssh_ip, ssh_port):
        logging.info(f"Waiting for port {ssh_port} on {ssh_ip} to open...")
        await asyncio.sleep(1)
    
    # Connect to the pod via SSH
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ssh_ip, port=ssh_port, username="root", key_filename=private_key_path, banner_timeout=200)
    
    # Function to check if SSH session is still alive
    def is_ssh_session_alive():
        transport = ssh.get_transport()
        return transport is not None and transport.is_active()

    # Set up the SFTP client
    sftp = ssh.open_sftp()

    # Initialize the helper structure
    pod_helpers = {
        'private_key_path': private_key_path,
        'log': open(log_file, "a"),  # Append mode for logs
        'is_ssh_session_alive': is_ssh_session_alive
    }
    
    # Start the task to copy logs from the remote pod to the local machine
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    local_log_path = os.path.join(logs_dir, f"{name}.log")
    
    # Set up file copying tasks
    pod_helpers['sftp_task'] = asyncio.create_task(copy_file_from_remote(ssh, remote_log_path, local_log_path))

    # If the pod is the master, set up additional tasks for copying the loss file
    local_loss_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "loss_plot.png")
    if name == 'master':
        pod_helpers['loss_task'] = asyncio.create_task(copy_file_from_remote(ssh, remote_loss_path, local_loss_file_path, interval=5, copy_mode='full'))
    
    pod_helpers['sftp'] = sftp
    pod_helpers['ssh'] = ssh
    
    logging.info(f"Reconnected and initialized pod {pod_id} successfully.")
    
    return pod_helpers
