# spl/master/deploy.py

import subprocess
import os
import asyncio
import logging
from eth_account import Account

import paramiko  # new for SSH
import socket

from ..models import ServiceType
from ..common import generate_wallets, SOT_PRIVATE_PORT, wait_for_health
from .config import args
from .main_logic import Master

logger = logging.getLogger(__name__)

DB_PERM_ID = 1

################################################################
# Parse the user:pass@host[:port] endpoints
################################################################
def parse_ssh_endpoint(endpoint_str: str):
    """
    Parses a string like "user:pass@somehost" or "alice:secret@mycloud.com:2202"
    into a dict: {
       'username': 'alice',
       'password': 'secret',
       'hostname': 'mycloud.com',
       'port': 2202
    }
    If port not specified => default to 22.
    """
    # Example: "alice:secret@mycloud.com:2202"
    # 1) Split on '@'
    if '@' not in endpoint_str:
        raise ValueError(f"Invalid endpoint (no '@'): {endpoint_str}")

    userpass_part, hostport_part = endpoint_str.split('@', 1)
    # userpass_part = "alice:secret"
    # hostport_part = "mycloud.com:2202"

    if ':' not in userpass_part:
        raise ValueError(f"Invalid endpoint user:pass portion: {userpass_part}")
    user, password = userpass_part.split(':', 1)

    # check if there's a port
    port = 22
    hostname = hostport_part
    if ':' in hostport_part:
        # e.g. "mycloud.com:2202"
        parts = hostport_part.split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid host:port format in {hostport_part}")
        hostname = parts[0]
        try:
            port = int(parts[1])
        except ValueError:
            raise ValueError(f"Port must be integer in host:port => {hostport_part}")

    return {
        'username': user,
        'password': password,
        'hostname': hostname,
        'port': port
    }

################################################################
# We'll store each endpoint as a dict with user/pass/host/port
################################################################
_CLOUD_ENDPOINTS = []
if args.cloud_ssh_endpoints.strip():
    raw_items = [s.strip() for s in args.cloud_ssh_endpoints.split(",") if s.strip()]
    for item in raw_items:
        parsed = parse_ssh_endpoint(item)
        _CLOUD_ENDPOINTS.append(parsed)

################################################################
# In-memory 'slots' tracking for concurrency in cloud deployment
################################################################
_CLOUD_SLOTS = [None for _ in range(args.max_concurrent_jobs)]

def get_cloud_slot_for_job(job_id: int) -> int:
    """
    In a real system, you'd have logic in jobs.py to pick a free slot
    or store the job->slot assignment in the DB. 
    For demonstration, we simply pick the first free slot.
    """
    for i in range(args.max_concurrent_jobs):
        if _CLOUD_SLOTS[i] is None:
            return i
    raise RuntimeError("No free cloud slot available (should not happen if concurrency is enforced).")

def ensure_slot_dict(slot_index: int):
    """
    Returns a dict structure for the given slot, creating it if needed.
    Each slot dict => {
       'endpoint': {...}  # one from _CLOUD_ENDPOINTS
       'ssh_client_sot': paramiko.SSHClient or None
       'ssh_client_worker': paramiko.SSHClient or None
       'job_id': ...
    }
    We'll pick the endpoint as slot_index % len(_CLOUD_ENDPOINTS].
    """
    if slot_index < 0 or slot_index >= args.max_concurrent_jobs:
        raise ValueError(f"Invalid slot index {slot_index}")

    if _CLOUD_ENDPOINTS == []:
        raise ValueError("No cloud SSH endpoints provided via --cloud_ssh_endpoints")

    if _CLOUD_SLOTS[slot_index] is None:
        endpoint_index = slot_index % len(_CLOUD_ENDPOINTS)
        endpoint = _CLOUD_ENDPOINTS[endpoint_index]
        logger.info(f"[ensure_slot_dict] Creating new slot={slot_index} for endpoint={endpoint}")
        _CLOUD_SLOTS[slot_index] = {
            'endpoint': endpoint,
            'ssh_client_sot': None,
            'ssh_client_worker': None,
            'job_id': None
        }
    return _CLOUD_SLOTS[slot_index]

def open_ssh_connection(endpoint: dict) -> paramiko.SSHClient:
    """
    Create and return an SSHClient connected to the endpoint's host/port with the given user/password.
    endpoint = {
       'username': ...,
       'password': ...,
       'hostname': ...,
       'port': ...
    }
    """
    username = endpoint['username']
    password = endpoint['password']
    hostname = endpoint['hostname']
    port = endpoint['port']

    logger.info(f"Opening SSH connection to {username}@{hostname}:{port} ...")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(
            hostname=hostname,
            port=port,
            username=username,
            password=password,
            timeout=15
        )
    except (paramiko.SSHException, socket.error) as e:
        logger.error(f"SSH connect failed to {hostname}:{port} => {e}")
        raise

    return client

async def run_remote_process(ssh_client: paramiko.SSHClient, command: list, cwd: str):
    """
    Launch a remote process over SSH. We'll do a naive 'nohup' approach so it won't block.
    """
    cmd_str = " ".join(command)
    logger.info(f"[run_remote_process] remote: (cd {cwd}; nohup {cmd_str} > /dev/null 2>&1 & )")

    # We do a 'nohup ... &' so the command keeps running after we close the channel
    full_command = f"cd {cwd} && nohup {cmd_str} > /dev/null 2>&1 &"
    stdin, stdout, stderr = ssh_client.exec_command(full_command)
    # We won't block on it. The process stays running on the remote side.
    # If you want to store the remote PID, parse `echo $!`. For now, we rely on SSH session close to kill.

    # Optionally read anything from stdout/stderr, but since we did > /dev/null, there's none
    return

################################################################
# The main SOT + Worker launching
################################################################
async def launch_sot(db_adapter, job, deploy_type, db_url):
    logging.info(f"launch_sot for job={job.id}")
    sot_wallet = generate_wallets(1)[0]
    sot_id = await db_adapter.create_sot(job.id, sot_wallet['address'], None)

    if deploy_type == 'local':
        # EXACTLY as in your original code
        sot_url = f"http://localhost:{SOT_PRIVATE_PORT}"
        sot_log_path = os.path.join(os.path.dirname(__file__), "..", "logs", "sot.log")
        os.makedirs(os.path.dirname(sot_log_path), exist_ok=True)
        sot_log = open(sot_log_path, 'w')
        package_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        sot_process = subprocess.Popen(
            [
                'python', '-m', 'spl.sot',
                '--sot_id', str(sot_id),
                '--db_url', db_url,
                '--private_key', sot_wallet['private_key'],
            ],
            stdout=sot_log, stderr=sot_log, cwd=package_root_dir
        )
        instance_pid = sot_process.pid
        await db_adapter.create_instance(
            name="sot",
            service_type=ServiceType.Sot.name,
            job_id=job.id,
            private_key=None,
            pod_id=None,
            process_id=instance_pid
        )
        sot_url_final = sot_url

        logging.info(f"[LOCAL] SOT started, pid={instance_pid}")
        if not await wait_for_health(sot_url_final):
            logging.error("SOT did not become healthy in time. Terminating.")
            sot_process.terminate()
            os._exit(1)

        await db_adapter.update_sot(sot_id, sot_url_final)
        await db_adapter.update_job_sot_url(job.id, sot_url_final)
        sot_db = await db_adapter.get_sot_by_job_id(job.id)

        # create perm for SOT
        sot_perm_id = sot_db.perm
        private_key_address = Account.from_key(args.private_key).address
        await db_adapter.create_perm(private_key_address, sot_perm_id)
        return sot_db, sot_url_final

    elif deploy_type == 'cloud':
        # find a free slot
        slot_index = get_cloud_slot_for_job(job.id)
        slot_info = ensure_slot_dict(slot_index)
        slot_info['job_id'] = job.id

        if slot_info['ssh_client_sot'] is None:
            slot_info['ssh_client_sot'] = open_ssh_connection(slot_info['endpoint'])

        # Adjust path for remote environment
        # Suppose you have your code in /home/ubuntu/spl or something similar:
        # We'll do a placeholder: "/opt/spl"
        package_root_dir = "/opt/spl"

        command = [
            'python', '-m', 'spl.sot',
            '--sot_id', str(sot_id),
            '--db_url', db_url,
            '--private_key', sot_wallet['private_key'],
        ]
        await run_remote_process(slot_info['ssh_client_sot'], command, package_root_dir)

        # We'll guess the SOT is listening on SOT_PRIVATE_PORT on that remote host:
        remote_host = slot_info['endpoint']['hostname']
        sot_url_final = f"http://{remote_host}:{SOT_PRIVATE_PORT}"

        await db_adapter.create_instance(
            name=f"sot_cloud_{slot_index}",
            service_type=ServiceType.Sot.name,
            job_id=job.id,
            private_key=None,
            pod_id=None,
            process_id=None
        )
        logging.info(f"[CLOUD] SOT launched on {remote_host}, job={job.id}, slot={slot_index}")

        # If you want health-check logic => implement something or skip
        # For demonstration, we skip.

        await db_adapter.update_sot(sot_id, sot_url_final)
        await db_adapter.update_job_sot_url(job.id, sot_url_final)
        sot_db = await db_adapter.get_sot_by_job_id(job.id)
        # create perm
        sot_perm_id = sot_db.perm
        private_key_address = Account.from_key(args.private_key).address
        await db_adapter.create_perm(private_key_address, sot_perm_id)
        return sot_db, sot_url_final

    else:
        raise NotImplementedError(f"Unknown deploy_type={deploy_type}")

async def launch_worker(db_adapter, job, deploy_type, subnet, db_url: str, sot_url: str, worker_key: str):
    worker_name = f"worker_{worker_key[-6:]}"

    if deploy_type == 'local':
        package_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        command = [
            'python', '-m', 'spl.worker',
            '--cli',
            '--subnet_id', str(subnet.id),
            '--db_url', db_url,
            '--private_key', worker_key,
        ]
        if args.torch_compile:
            command.append('--torch_compile')
        if args.detailed_logs:
            command.append('--detailed_logs')

        log_dir = os.path.join(package_root_dir, "spl", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f"{worker_name}.log")
        log_file = open(log_file_path, 'w')

        worker_process = subprocess.Popen(command, stdout=log_file, stderr=log_file, cwd=package_root_dir)
        instance_pid = worker_process.pid

        await db_adapter.create_instance(
            name=worker_name,
            service_type=ServiceType.Worker.name,
            job_id=job.id,
            private_key=None,
            pod_id=None,
            process_id=instance_pid
        )
        logging.info(f"[LOCAL] Worker started pid={instance_pid}, job={job.id}, name={worker_name}")

    elif deploy_type == 'cloud':
        # same approach
        slot_index = get_cloud_slot_for_job(job.id)
        slot_info = ensure_slot_dict(slot_index)

        if slot_info['ssh_client_worker'] is None:
            slot_info['ssh_client_worker'] = open_ssh_connection(slot_info['endpoint'])

        package_root_dir = "/opt/spl"

        command = [
            'python', '-m', 'spl.worker',
            '--cli',
            '--subnet_id', str(subnet.id),
            '--db_url', db_url,
            '--private_key', worker_key,
        ]
        if args.torch_compile:
            command.append('--torch_compile')
        if args.detailed_logs:
            command.append('--detailed_logs')

        await run_remote_process(slot_info['ssh_client_worker'], command, package_root_dir)

        await db_adapter.create_instance(
            name=worker_name,
            service_type=ServiceType.Worker.name,
            job_id=job.id,
            private_key=None,
            pod_id=None,
            process_id=None
        )
        logging.info(f"[CLOUD] Worker started for job={job.id}, slot={slot_index}, name={worker_name}")

    else:
        raise NotImplementedError(f"Unknown deploy_type={deploy_type}")

async def launch_workers(db_adapter, job, deploy_type, subnet, db_url: str, sot_url: str):
    for i in range(args.num_workers):
        key_result = await db_adapter.admin_create_account_key(job.user_id)
        worker_key = key_result['private_key']
        await launch_worker(db_adapter, job, deploy_type, subnet, db_url, sot_url, worker_key)

async def run_master_task(
    sot_url: str,
    job_id: int,
    subnet_id: int,
    db_adapter,
    max_iterations: float,
    detailed_logs: bool,
    max_concurrent_iterations: int = 4
):
    """
    The Master logic from your original code. After finishing,
    if we're in cloud mode, we'll close the SSH connections in that slot.
    """
    master_obj = Master(
        sot_url=sot_url,
        job_id=job_id,
        subnet_id=subnet_id,
        db_adapter=db_adapter,
        max_iterations=max_iterations,
        detailed_logs=detailed_logs,
        max_concurrent_iterations=max_concurrent_iterations
    )
    try:
        await master_obj.initialize()
        await master_obj.run_main()
        logger.info(f"[run_master_task] Master for job {job_id} ended normally.")
    except asyncio.CancelledError:
        logger.warning(f"[run_master_task] Master for job {job_id} canceled.")
    except Exception as e:
        logger.error(f"[run_master_task] Master for job {job_id} crashed: {e}", exc_info=True)
    finally:
        master_obj.done = True
        logger.info(f"[run_master_task] Master for job {job_id} fully finished.")

        # If we're in cloud mode, free the SSH slot so SOT/worker processes are killed
        if args.deploy_type == 'cloud':
            # Find the slot with job_id
            for i in range(args.max_concurrent_jobs):
                slot_dict = _CLOUD_SLOTS[i]
                if slot_dict and slot_dict.get('job_id') == job_id:
                    ssh_sot = slot_dict.get('ssh_client_sot')
                    ssh_worker = slot_dict.get('ssh_client_worker')
                    if ssh_sot:
                        ssh_sot.close()
                    if ssh_worker:
                        ssh_worker.close()
                    _CLOUD_SLOTS[i] = None
                    logger.info(f"[CLOUD CLEANUP] Freed slot={i} for job={job_id}")
                    break
