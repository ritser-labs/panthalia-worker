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

async def copy_file_from_remote(ssh, remote_path, local_path, interval=0.1):
    """
    Copies a file from the remote server to the local system every `interval` seconds, 
    ensuring that the file exists and is not of zero size before copying. A copy of the file
    is made on the remote server to avoid issues if the original file is modified during the transfer.

    Args:
        ssh (paramiko.SSHClient): The active SSH connection.
        remote_path (str): The path to the file on the remote server.
        local_path (str): The path to the file on the local system.
        interval (int): The time in seconds between each copy attempt.
    """
    sftp = ssh.open_sftp()
    try:
        while True:
            try:
                # Check if the file exists and is not zero size
                file_stat = sftp.stat(remote_path)
                if file_stat.st_size > 0:
                    # Create a temporary copy of the file on the remote system using SSH command
                    temp_remote_path = remote_path + ".bak"  # Add a .bak suffix for the backup copy
                    copy_command = f"cp {remote_path} {temp_remote_path}"
                    
                    stdin, stdout, stderr = ssh.exec_command(copy_command)
                    exit_status = stdout.channel.recv_exit_status()
                    
                    if exit_status == 0:

                        # Copy the temporary file to the local system using SFTP
                        sftp.get(temp_remote_path, local_path)

                        # Optionally remove the temporary remote copy after successful download
                        sftp.remove(temp_remote_path)
                    else:
                        logging.error(f"Failed to create copy on the remote server: {stderr.read().decode()}")

                else:
                    logging.info(f"File {remote_path} is zero size. Waiting for update...")
            except FileNotFoundError:
                pass
            except Exception as e:
                logging.error(f"Error checking or copying file: {e}")
                
            await asyncio.sleep(interval)
    except Exception as e:
        logging.error(f"Error copying file: {e}")
    finally:
        sftp.close()

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
        pod_helpers['log_task'] = asyncio.create_task(copy_file_from_remote(ssh, remote_loss_path, local_file_path))
        pod_helpers['sftp'] = sftp
        pod_helpers['ssh'] = ssh

        return new_pod, pod_helpers

    except Exception as e:
        runpod.terminate_pod(pod_id)
        logging.error(f"Error launching instance {name}: {e}")
        raise e
