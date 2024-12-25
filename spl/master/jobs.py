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
    """
    Production code that polls /get_jobs_in_progress every 2 seconds, returning:
      - all active jobs, plus
      - any inactive jobs that still have unfinished tasks.
    
    For each job returned, we ensure we have a Master running if not already. If a job
    is no longer in the list, that means it's fully resolved, so we stop it.
    
    This ensures that a job toggled inactive but with partial tasks in flight
    will STILL be processed until tasks resolve, so we never lose progress.
    """

    db_adapter = DBAdapterClient(db_url, private_key)
    # job_id -> asyncio.Task (the Master process)
    jobs_processing = {}

    while True:
        # 1) fetch the in-progress jobs from DB
        #    (active or inactive-with-unresolved-tasks)
        jobs_in_progress = await db_adapter.get_jobs_in_progress()
        if jobs_in_progress is None:
            logger.error("Failed to fetch jobs_in_progress; retrying in 2s.")
            await asyncio.sleep(2)
            continue

        # Convert them to a set of IDs
        in_progress_ids = set(j.id for j in jobs_in_progress)

        # 2) Start any new jobs we don’t have running yet
        for job_obj in jobs_in_progress:
            if job_obj.id not in jobs_processing:
                await db_adapter.admin_deposit_account(job_obj.user_id, 999999999) # TODO: remove this line
                logger.info(f"[check_for_new_jobs] Starting job {job_obj.id}. (Active or partial tasks remain)")
                
                # Launch SOT + workers
                sot_db, sot_url = await launch_sot(db_adapter, job_obj, deploy_type, db_url)
                await launch_workers(db_adapter, job_obj, deploy_type, sot_db, db_url, sot_url)
                
                # Start the Master
                master_task = asyncio.create_task(
                    run_master_task(
                        sot_url=sot_url,
                        job_id=job_obj.id,
                        subnet_id=job_obj.subnet_id,
                        db_adapter=db_adapter,
                        max_iterations=float('inf'),
                        detailed_logs=detailed_logs
                    )
                )
                jobs_processing[job_obj.id] = master_task

        # 3) Any job we’re running that’s NOT in the new list => done => stop it
        #    i.e. no tasks remain AND it’s inactive => Master can be canceled.
        finished = []
        for job_id, master_task in jobs_processing.items():
            if job_id not in in_progress_ids:
                # means that job is fully resolved & absent from get_jobs_in_progress
                logger.info(f"[check_for_new_jobs] Job {job_id} is fully resolved => stopping Master.")
                master_task.cancel()
                finished.append(job_id)
            elif master_task.done():
                finished.append(job_id)

        # Cleanup
        for job_id in finished:
            # optionally tear down container, etc.
            jobs_processing.pop(job_id, None)

        await asyncio.sleep(2)
