"""
spl/master/deploy.py

- If deploy_type == 'local': we spawn local subprocesses for SOT or Worker
  AND we create matching 'Instance' rows in the DB with the same name
  as the log filename. That way the local curses UI can match them.
- If deploy_type == 'cloud': we pick a free instance from DB (slot_type SOT or WORKER),
  reserve it for the job, parse connection_info for SSH, run remote processes.
"""

import os
import logging
import subprocess
import asyncio
import paramiko
import json

from eth_account import Account

from ..models.enums import SlotType, ServiceType
from ..common import generate_wallets, SOT_PRIVATE_PORT, wait_for_health
from .config import args
from .main_logic import Master
from ..db.db_adapter_client import DBAdapterClient

logger = logging.getLogger(__name__)


###############################################################################
# Helpers for Cloud Deploy
###############################################################################
def parse_connection_info(connection_info: str) -> dict:
    """Parses JSON from instance.connection_info into a dict with host, port, user, password, remote_dir, etc."""
    if not connection_info:
        return {}
    try:
        return json.loads(connection_info)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in connection_info: {connection_info}")
        return {}

def open_ssh_connection(host: str, port: int, user: str, password: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        port=port,
        username=user,
        password=password,
        timeout=15
    )
    return client

async def run_remote_process(ssh_client: paramiko.SSHClient, command: list, remote_dir: str):
    cmd_str = " ".join(command)
    logger.info(f"Remote command: cd {remote_dir}; nohup {cmd_str} > /dev/null 2>&1 &")
    full_cmd = f"cd {remote_dir} && nohup {cmd_str} > /dev/null 2>&1 &"
    ssh_client.exec_command(full_cmd)

async def pick_free_instance_for_job(db_adapter: DBAdapterClient, slot_type: SlotType, job_id: int):
    """Pick the first free instance from DB for the given slot_type, reserve it for job_id, return instance dict."""
    free_list = await db_adapter.get_free_instances_by_slot_type(slot_type)
    if not free_list:
        logger.warning(f"No free Instances found for slot_type={slot_type.name}")
        return None
    candidate = free_list[0]
    success = await db_adapter.reserve_instance(candidate['id'], job_id)
    if not success:
        logger.warning(f"Failed to reserve instance {candidate['id']} for job {job_id}")
        return None
    logger.info(f"Reserved instance {candidate['id']} for job {job_id} (slot_type={slot_type.name})")
    return candidate


###############################################################################
# Launch SOT
###############################################################################
async def launch_sot(db_adapter: DBAdapterClient, job, db_url: str):
    """
    - Creates a SOT row in DB
    - If local => spawns local subproc + logs to `sot_local_<jobid>.log`, 
      also creates a DB Instance with the same name for the curses UI.
    - If cloud => picks a free SOT Instance from DB, does SSH, sets URL
    Returns (sot_db_obj, sot_url).
    """
    logger.info(f"Launching SOT for job={job.id}")
    sot_wallet = generate_wallets(1)[0]
    sot_id = await db_adapter.create_sot(job.id, sot_wallet['address'], None)

    if args.deploy_type == 'local':
        # Local subproc
        sot_name = f"sot_local_{job.id}"  # We'll use this for both the .log file & instance name
        sot_url = f"http://localhost:{SOT_PRIVATE_PORT}"
        logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # Use a unique log file matching the instance name
        sot_log_file = os.path.join(logs_dir, f"{sot_name}.log")
        sot_log = open(sot_log_file, 'w')

        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        proc = subprocess.Popen(
            [
                'python', '-m', 'spl.sot',
                '--sot_id', str(sot_id),
                '--db_url', db_url,
                '--private_key', sot_wallet['private_key']
            ],
            stdout=sot_log, stderr=sot_log, cwd=package_root
        )
        pid_val = proc.pid
        logger.info(f"[LOCAL] SOT started with pid={pid_val} for job={job.id}")

        if not await wait_for_health(sot_url):
            logger.error("SOT not healthy => kill process.")
            proc.terminate()
            raise RuntimeError("SOT health check timed out")

        final_url = sot_url

        # Create a DB Instance row so local curses UI can see it
        instance_id = await db_adapter.create_instance(
            name=sot_name,
            service_type=ServiceType.Sot.name,
            job_id=job.id,
            private_key=sot_wallet['private_key'],
            pod_id="",
            process_id=pid_val
        )
        logger.info(f"[LOCAL] SOT instance row created with id={instance_id} (name={sot_name})")

    else:
        # Cloud => pick SOT instance from DB, parse connection_info, SSH
        sot_instance = await pick_free_instance_for_job(db_adapter, SlotType.SOT, job.id)
        if not sot_instance:
            raise RuntimeError("No free SOT instance for cloud deploy.")
        cinfo_str = sot_instance.get('connection_info', '')
        cinfo = parse_connection_info(cinfo_str)

        host = cinfo.get('host', '127.0.0.1')
        port = cinfo.get('port', 22)
        user = cinfo.get('user', 'ubuntu')
        password = cinfo.get('password', 'Secret')
        remote_dir = cinfo.get('remote_dir', '/opt/spl')

        ssh_client = open_ssh_connection(host, port, user, password)
        command = [
            'python', '-m', 'spl.sot',
            '--sot_id', str(sot_id),
            '--db_url', db_url,
            '--private_key', sot_wallet['private_key']
        ]
        await run_remote_process(ssh_client, command, remote_dir)

        final_url = f"http://{host}:{SOT_PRIVATE_PORT}"

    # Update SOT info in DB
    await db_adapter.update_sot(sot_id, final_url)
    await db_adapter.update_job_sot_url(job.id, final_url)

    # create perm
    sot_db_obj = await db_adapter.get_sot_by_job_id(job.id)
    perm_id = sot_db_obj.perm
    master_addr = Account.from_key(args.private_key).address
    await db_adapter.create_perm(master_addr, perm_id)

    return sot_db_obj, final_url


###############################################################################
# Launch Workers
###############################################################################
async def launch_workers(db_adapter: DBAdapterClient, job, db_url: str, num_workers: int):
    """
    If local => spawns local procs, logs to `worker_<jobid>_<idx>.log`,
      and creates a DB Instance for each.
    If cloud => picks free WORKER Instances from DB and does SSH.
    """
    for i in range(num_workers):
        w_index = i + 1
        logger.info(f"Launching worker {w_index}/{num_workers} for job={job.id}")

        # generate or fetch a worker private_key
        w_key_resp = await db_adapter.admin_create_account_key(job.user_id)
        w_priv = w_key_resp['private_key']

        if args.deploy_type == 'local':
            # local subproc
            logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
            os.makedirs(logs_dir, exist_ok=True)

            # Name the log file & instance consistently
            instance_name = f"worker_{job.id}_{w_index}"
            w_log_file = os.path.join(logs_dir, f"{instance_name}.log")
            log_handle = open(w_log_file, 'w')

            package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            command = [
                'python', '-m', 'spl.worker',
                '--cli',
                '--subnet_id', str(job.subnet_id),
                '--db_url', db_url,
                '--private_key', w_priv
            ]
            if args.torch_compile:
                command.append('--torch_compile')
            if args.detailed_logs:
                command.append('--detailed_logs')

            proc = subprocess.Popen(command, stdout=log_handle, stderr=log_handle, cwd=package_root)
            pid_val = proc.pid
            logger.info(f"[LOCAL] Worker started pid={pid_val} for job={job.id}")

            # Create DB Instance row so it shows in local curses UI
            instance_id = await db_adapter.create_instance(
                name=instance_name,
                service_type=ServiceType.Worker.name,
                job_id=job.id,
                private_key=w_priv,
                pod_id="",
                process_id=pid_val
            )
            logger.info(f"[LOCAL] Worker instance row created with id={instance_id} (name={instance_name})")

        else:
            # cloud => pick a worker instance from DB
            worker_instance = await pick_free_instance_for_job(db_adapter, SlotType.WORKER, job.id)
            if not worker_instance:
                raise RuntimeError("No free WORKER instance for cloud deploy.")
            cinfo_str = worker_instance.get('connection_info', '')
            cinfo = parse_connection_info(cinfo_str)

            host = cinfo.get('host', '127.0.0.1')
            port = cinfo.get('port', 22)
            user = cinfo.get('user', 'ubuntu')
            password = cinfo.get('password', 'Secret')
            remote_dir = cinfo.get('remote_dir', '/opt/spl')

            ssh_client = open_ssh_connection(host, port, user, password)
            command = [
                'python', '-m', 'spl.worker',
                '--cli',
                '--subnet_id', str(job.subnet_id),
                '--db_url', db_url,
                '--private_key', w_priv
            ]
            if args.torch_compile:
                command.append('--torch_compile')
            if args.detailed_logs:
                command.append('--detailed_logs')

            await run_remote_process(ssh_client, command, remote_dir)
            logger.info(f"[CLOUD] Worker launched for job={job.id}, index={w_index}")


###############################################################################
# Run Master Task
###############################################################################
async def run_master_task(db_adapter: DBAdapterClient, job_id: int, max_iters: int = 999999):
    """
    Creates Master object from main_logic, runs it to completion.
    """
    job_obj = await db_adapter.get_job(job_id)
    if not job_obj:
        raise RuntimeError(f"Job {job_id} not found")

    sot_url = job_obj.sot_url
    subnet_id = job_obj.subnet_id
    master = Master(
        sot_url=sot_url,
        job_id=job_id,
        subnet_id=subnet_id,
        db_adapter=db_adapter,
        max_iterations=float(max_iters),
        detailed_logs=args.detailed_logs,
        max_concurrent_iterations=4
    )
    await master.initialize()
    await master.run_main()
    logger.info(f"Master finished for job {job_id}")
