import time
from runpod import create_pod, get_pod, get_pods, terminate_pod

def create_new_pod(
    name, image, gpu_type, gpu_count, support_public_ip, ports
):
    kwargs = {
        'gpu_count': gpu_count,
        'support_public_ip': support_public_ip,
        'ports': ports
    }
    # Assuming additional code here to create a pod with the above kwargs
    new_pod = create_pod(name, image, gpu_type, **kwargs)
    return new_pod

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
        pod = get_pod(pod_id)
        desired_status = pod.get('desiredStatus', None)
        runtime = pod.get('runtime', None)

        if desired_status == 'RUNNING' and runtime and 'ports' in runtime:
            for port in runtime['ports']:
                if port['privatePort'] == private_port:
                    pod_ip = port['ip']
                    pod_port = int(port['publicPort'])
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
    pods = get_pods()
    for pod in pods:
        pod_id = pod.id
        terminate_pod(pod_id)

