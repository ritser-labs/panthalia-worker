import time
import runpod
import threading
import paramiko
import logging
import os
from .runpod_config import RUNPOD_API_KEY
from ..util import is_port_open
import json

runpod.api_key = RUNPOD_API_KEY
INPUT_JSON_PATH = '/tmp/input.json'

def generate_ssh_key_pair():
    """
    Generates an SSH key pair using Paramiko and saves it locally.
    
    Returns:
        str: The path to the private key file.
        str: The public key string.
    """
    key = paramiko.RSAKey.generate(2048)
    private_key_path = os.path.expanduser("~/.ssh/id_rsa_runpod")
    public_key_str = f"{key.get_name()} {key.get_base64()}"

    # Save the private key
    with open(private_key_path, "w") as private_key_file:
        key.write_private_key(private_key_file)

    # Set appropriate permissions for the private key file
    os.chmod(private_key_path, 0o600)

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

def create_cpu_pod(
    name, image, instance_id, env={}, ports=None, template_id=None, container_disk_in_gb=10, support_public_ip=True, cloud_type='SECURE'
):
    mutation = generate_deploy_cpu_pod_mutation(
        name, image, instance_id, env, ports, template_id, container_disk_in_gb, support_public_ip, cloud_type
    )
    raw_response = runpod.api.graphql.run_graphql_query(mutation)
    cleaned_response = raw_response["data"]["deployCpuPod"]
    return cleaned_response

def create_new_pod(
    name, image, gpu_type, gpu_count, support_public_ip, ports, env={}, template_id=None, container_disk_in_gb=10
):
    private_key_path, public_key_str = generate_ssh_key_pair()
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
        new_pod = create_cpu_pod(name, image, instance_id='cpu3c-2-4', **kwargs)
    else:
        if image is None:
            raise ValueError("Image must be provided for GPU instances.")
        kwargs['gpu_type_id'] = gpu_type
        kwargs['gpu_count'] = gpu_count
        new_pod = runpod.create_pod(name, image, **kwargs)
    return new_pod, private_key_path

def get_public_ip_and_port(pod_id, private_port, timeout=300):
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
        pod = runpod.get_pod(pod_id)
        desired_status = pod.get('desiredStatus', None)
        runtime = pod.get('runtime', None)

        if desired_status == 'RUNNING' and runtime and 'ports' in runtime and runtime['ports'] is not None:
            for port in runtime['ports']:
                if port['privatePort'] == private_port:
                    pod_ip = port['ip']
                    pod_port = int(port['publicPort'])
                    logging.info(f"Pod {pod_id} has IP {pod_ip} and port {pod_port} for private port {private_port}")
                    break

        time.sleep(1)

    if desired_status != 'RUNNING':
        raise TimeoutError(f"Pod {pod_id} did not reach 'RUNNING' state within {timeout} seconds.")

    if runtime is None:
        raise TimeoutError(f"Pod {pod_id} did not report runtime data within {timeout} seconds.")

    if pod_ip is None or pod_port is None:
        raise TimeoutError(f"Pod {pod_id} did not provide the IP and port for private port {private_port} within {timeout} seconds.")

    return pod_ip, pod_port

def get_pod_ssh_ip_port(pod_id, timeout=300):
    """
    Returns the IP and port for SSH access to a pod by calling the generic function.

    Args:
        pod_id (str): The ID of the pod.
        timeout (int): Timeout in seconds to wait for the pod to be ready.

    Returns:
        tuple: A tuple containing the IP and port for SSH access.
    """
    return get_public_ip_and_port(pod_id, private_port=22, timeout=timeout)

def terminate_all_pods():
    """
    Terminates all pods that are currently running.
    """
    pods = runpod.get_pods()
    for pod in pods:
        pod_id = pod['id']
        runpod.terminate_pod(pod_id)
    logging.info("All pods have been terminated.")

def stream_handler(stream, log_file):
    """
    Handles streaming of logs to a file in real-time.

    Args:
        stream (paramiko.ChannelFile): The stream to read from (stdout or stderr).
        log_file (file): The file object to write logs to.
    """
    for line in iter(stream.readline, ""):
        log_file.write(line)
        log_file.flush()  # Ensure logs are written immediately

def copy_file_from_remote(ssh, remote_path, local_path, interval=1):
    """
    Copies a file from the remote server to the local system every `interval` seconds, 
    ensuring that the file exists and is not of zero size before copying.

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
                    # If the file exists and has content, copy it to the local system
                    sftp.get(remote_path, local_path)
                    logging.info(f"Copied {remote_path} to {local_path}")
                else:
                    logging.info(f"File {remote_path} is zero size. Waiting for update...")
            except FileNotFoundError:
                # If the file doesn't exist, log the event
                pass
            except Exception as e:
                logging.error(f"Error checking file: {e}")
                
            time.sleep(interval)
    except Exception as e:
        logging.error(f"Error copying file: {e}")
    finally:
        sftp.close()

def launch_instance_and_record_logs(
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
        ports (list): A list of ports to be exposed, example format - "8888/http,666/tcp"
        log_file (str): The file where logs will be recorded.
        timeout (int): Timeout in seconds to wait for the pod to be ready.

    Returns:
        None
    """

    if ports:
        ports += ",22/tcp"  # Ensure SSH port is exposed
    else:
        ports = '22/tcp'

    new_pod, private_key_path = create_new_pod(
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
        ssh_ip, ssh_port = get_pod_ssh_ip_port(pod_id, timeout=timeout)

        while not is_port_open(ssh_ip, ssh_port):
            time.sleep(1)

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
        remote_path = '/tmp/temp_script.sh'
        with sftp.file(remote_path, 'w') as remote_file:
            remote_file.write(f'export {env_export}\n{cmd}')
        sftp.chmod(remote_path, 0o755)  # Make it executable

        # Execute the temporary script file using bash
        stdin, stdout, stderr = ssh.exec_command(f'/bin/bash {remote_path}')

        # Step 5: Write logs to file
        pod_helpers = {}
        pod_helpers['log'] = open(log_file, "w")
        # Create separate threads for handling stdout and stderr streams
        pod_helpers['stdout_thread'] = threading.Thread(target=stream_handler, args=(stdout, pod_helpers['log']))
        pod_helpers['stderr_thread'] = threading.Thread(target=stream_handler, args=(stderr, pod_helpers['log']))
        
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
        pod_helpers['is_ssh_session_alive'] = is_ssh_session_alive

        # Start both threads
        pod_helpers['stdout_thread'].start()
        pod_helpers['stderr_thread'].start()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        parent_dir = os.path.dirname(parent_dir)

        # Step 6: Create a new thread to copy the file from remote to local every second
        remote_file_path = "/app/spl/loss_plot.png"  # Path to the file on the remote system

        # Define the local path relative to the script
        local_file_path = os.path.join(parent_dir, "loss_plot.png")  # Saves in the same directory as the script

        pod_helpers['sftp_thread'] = threading.Thread(target=copy_file_from_remote, args=(ssh, remote_file_path, local_file_path))
        pod_helpers['sftp_thread'].start()

        return new_pod, pod_helpers

    except Exception as e:
        runpod.terminate_pod(pod_id)
        raise e
    finally:
        print(f"Logs have been recorded in {log_file}")

