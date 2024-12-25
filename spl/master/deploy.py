# spl/master/deploy.py
import subprocess
import os
import asyncio
import logging
from eth_account import Account
from ..models import ServiceType
from ..common import generate_wallets, SOT_PRIVATE_PORT, wait_for_health
from .config import args
from .main_logic import Master

logger = logging.getLogger(__name__)

DB_PERM_ID = 1

async def launch_sot(db_adapter, job, deploy_type, db_url):
    logging.info(f"launch_sot")
    sot_wallet = generate_wallets(1)[0]
    sot_id = await db_adapter.create_sot(job.id, None)
    
    if deploy_type == 'local':
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
        instance_private_key = None
        instance_pod_id = None
        instance_pid = sot_process.pid
    else:
        # Handle cloud deployment if needed
        raise NotImplementedError("Cloud deployment not implemented in this refactoring.")

    await db_adapter.create_instance(
        "sot",
        ServiceType.Sot.name,
        job.id,
        instance_private_key,
        instance_pod_id,
        instance_pid
    )
    logging.info(f"SOT service started")
    if not await wait_for_health(sot_url):
        logging.error("Error: SOT service did not become available within the timeout period.")
        sot_process.terminate()
        exit(1)
    await db_adapter.update_sot(sot_id, sot_url)
    await db_adapter.update_job_sot_url(job.id, sot_url)
    sot_db = await db_adapter.get_sot(job.id)
    sot_perm_id = sot_db.perm
    private_key_address = Account.from_key(args.private_key).address
    await db_adapter.create_perm(private_key_address, sot_perm_id)
    await db_adapter.create_perm(sot_wallet['address'], DB_PERM_ID)
    return sot_db, sot_url

async def launch_worker(db_adapter, job, deploy_type, subnet, db_url: str, sot_url: str, worker_key: str):
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
        worker_name = f'worker_{worker_key[-6:]}'
        log_file_path = os.path.join(log_dir, f"{worker_name}.log")
        log_file = open(log_file_path, 'w')
        worker_process = subprocess.Popen(command, stdout=log_file, stderr=log_file, cwd=package_root_dir)
        instance_private_key = None
        instance_pod_id = None
        instance_pid = worker_process.pid
    else:
        # Handle cloud deployment if needed
        raise NotImplementedError("Cloud deployment not implemented in this refactoring.")

    await db_adapter.create_instance(
        name=worker_name,
        service_type=ServiceType.Worker.name,
        job_id=job.id,
        private_key=instance_private_key,
        pod_id=instance_pod_id,
        process_id=instance_pid
    )
    logging.info(f"Started worker process for tasks with command: {' '.join(command)}")

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
    Create a Master object, run .run_main() until complete or canceled.
    If the job toggles inactive but still has tasks, .run_main() will keep
    them going until resolved.
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
