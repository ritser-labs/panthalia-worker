# spl/master/jobs.py
import asyncio
import logging
from ..db.db_adapter_client import DBAdapterClient
from ..models import ServiceType
from .deploy import launch_sot, launch_workers, run_master_task
from .config import args

logger = logging.getLogger(__name__)

async def check_for_new_jobs(
    private_key: str,
    db_url: str,
    detailed_logs: bool,
    num_workers: int,
    deploy_type: str,
    num_master_wallets: int
):
    jobs_processing = {}
    db_adapter = DBAdapterClient(db_url, private_key)
    
    logger.info(f"Checking for new jobs")
    while True:
        new_jobs = await db_adapter.get_jobs_without_instances()
        for job in new_jobs:
            if job.id in jobs_processing:
                continue

            logger.info(f"Starting new job: {job.id}")
            subnet = await db_adapter.get_subnet(job.subnet_id)
            
            await db_adapter.admin_deposit_account(job.user_id, 999999999)
            # SOT
            logging.info(f"Starting SOT service")
            sot_db, sot_url = await launch_sot(
                db_adapter, job, deploy_type, db_url)
            # Workers
            logging.info(f"Starting worker processes")
            await launch_workers(
                db_adapter, job, deploy_type, subnet,
                db_url, sot_url
            )

            # Master
            logging.info(f"Starting master process")
            
            master_args = [
                sot_url,
                job.id,
                job.subnet_id,
                db_adapter,
                float('inf'),
                detailed_logs,
            ]

            # run master in a non-blocking way
            logger.info(f'Starting master process for job {job.id}')
            task = asyncio.create_task(run_master_task(*master_args))
            jobs_processing[job.id] = task

        # clean up finished jobs
        completed_jobs = [job_id for job_id, task in jobs_processing.items() if task.done()]
        for job_id in completed_jobs:
            jobs_processing.pop(job_id)

        await asyncio.sleep(1)
