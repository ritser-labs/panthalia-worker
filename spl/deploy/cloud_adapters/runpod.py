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
    input_fields = []
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

async def create_cpu_pod(
    name, image, instance_id, env={}, ports=None, template_id=None, container_disk_in_gb=10, support_public_ip=True, cloud_type='SECURE'
):
    mutation = generate_deploy_cpu_pod_mutation(
        name, image, instance_id, env, ports, template_id, container_disk_in_gb, support_public_ip, cloud_type
    )

    # Run the blocking API call in a thread
    raw_response = await asyncio.get_event_loop().run_in_executor(executor, lambda: runpod.api.graphql.run_graphql_query(mutation))
    cleaned_response = raw_response["data"]["deployCpuPod"]
    return cleaned_response

async def create_new_pod(
    name, image, gpu_type, gpu_count, support_public_ip, ports, env={}, template_id=None, container_disk_in_gb=10
):
    private_key_path, public_key_str = await generate_ssh_key_pair()
    env['PUBLIC_KEY'] = public_key_str  # Add public key here
    kwargs = {
        'support_public_ip': support_public_ip,
        'ports': ports,
        'env': env,
        'template_id': template_id,
        'cloud_type': 'SECURE',
        'container_disk_in_gb': container_disk_in_gb
    }
    if gpu_type is None or gpu_count == 0:
        new_pod = await create_cpu_pod(name, image, instance_id='cpu3c-2-4', **kwargs)
    else:
        if image is None:
            raise ValueError("Image must be provided for GPU instances.")
        kwargs['gpu_type_id'] = gpu_type
        kwargs['gpu_count'] = gpu_count
        # Run the blocking pod creation in a thread
        new_pod = await asyncio.get_event_loop().run_in_executor(executor, lambda: runpod.create_pod(name, image, **kwargs))
    return new_pod, private_key_path

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

                    break  # Exit the loop after the first full copy

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
    Run the SSH command asynchronously using an executor.
    Args:
        ssh (paramiko.SSHClient): The active SSH connection.
        command (str): The command to execute.

    Returns:
        tuple: stdin, stdout, stderr as with paramiko exec_command.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, ssh.exec_command, command)

async def launch_instance_and_record_logs(
    name,
    image=None,
    gpu_type=None,
    gpu_count=0,
    ports='',
    log_file="instance_logs.txt",
    timeout=300,
    env={},
    cmd='tail -f /var/log/syslog',
    template_id=None,
    container_disk_in_gb=10,
    input_jsons=[]
):
    """
    Launches a new instance, waits for it to be ready, SSH into it, and records the logs.

    Args:
        name (str): The name of the pod.
        image (str): The Docker image to be used.
        gpu_type (str): The type of GPU required.
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
        name,
        image,
        gpu_type,
        gpu_count,
        support_public_ip=True,
        ports=ports,
        env={},  # we will use our env vars for the cmd not initial env vars
        template_id=template_id,
        container_disk_in_gb=container_disk_in_gb
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
        ssh.connect(ssh_ip, port=ssh_port, username="root", key_filename=private_key_path)

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
        # Execute the temporary script file using bash with nohup
        full_command = f'nohup /bin/bash {remote_path} > {remote_log_path} 2>&1 &'
        stdin, stdout, stderr = await async_exec_command(ssh, full_command)

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
    ssh.connect(ssh_ip, port=ssh_port, username="root", key_filename=private_key_path)
    
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
