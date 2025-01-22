# spl/master/jobs.py
import asyncio
import logging
import socket
import os
import time

from ..db.db_adapter_client import DBAdapterClient
from ..models import ServiceType
from .deploy import launch_sot, launch_workers, run_master_task
from .config import args

logger = logging.getLogger(__name__)

# For the multi-master scenario, define a unique ID for this Master instance
MASTER_ID = f"{socket.gethostname()}_{os.getpid()}"
TIMEOUT_SECONDS = 30 * 60  # 30 minutes

async def handle_newly_assigned_job(
    db_adapter,
    job_obj,
    deploy_type,
    db_url,
    master_id,
    jobs_processing,
    deposit_amount=999999999,
    unqueue_if_needed=False
):
    """
    DRY helper to:
      - Optionally unqueue the job (if unqueue_if_needed=True) and mark assigned_master_id=master_id.
      - Admin-deposit (for demonstration).
      - Launch the SOT + workers.
      - Create and store the local Master task in jobs_processing.

    :param db_adapter: DBAdapterClient instance
    :param job_obj: The Job object (with .id, .user_id, etc.)
    :param deploy_type: "local" or "cloud" (currently only "local" is implemented)
    :param db_url: The DB connection string
    :param master_id: e.g. MASTER_ID to store in job.assigned_master_id
    :param jobs_processing: dict of job_id => asyncio.Task for run_master_task
    :param deposit_amount: int – how much to deposit (if needed)
    :param unqueue_if_needed: bool – if True, we call update_job_queue_status(...) to unqueue + assign.
    """

    # 1) Possibly unqueue / assign to master if we want that
    if unqueue_if_needed:
        logger.info(f"[handle_newly_assigned_job] Assigning job {job_obj.id} to {master_id}.")
        success = await db_adapter.update_job_queue_status(
            job_id=job_obj.id,
            new_queued=False,        # remove from queue
            assigned_master_id=master_id
        )
        if not success:
            logger.warning(f"[handle_newly_assigned_job] Failed to assign job {job_obj.id} to {master_id}.")
            return  # skip if we can't assign

    # 2) Admin deposit
    #    NOTE: In your actual code, the deposit might be conditional, 
    #    but here we replicate the same logic as the queued-jobs block
    await db_adapter.admin_deposit_account(job_obj.user_id, deposit_amount)

    # 3) Launch SOT + workers
    sot_db, sot_url = await launch_sot(db_adapter, job_obj, db_url)
    await launch_workers(db_adapter, job_obj, db_url, args.num_workers)

    # 4) Create the local Master task for run_master_task
    master_task = asyncio.create_task(
        run_master_task(
            job_id=job_obj.id,
            db_adapter=db_adapter,
            max_iters=float('inf')
        )
    )
    jobs_processing[job_obj.id] = master_task


async def check_for_new_jobs(
    private_key: str,
    db_url: str,
    detailed_logs: bool,
    num_workers: int,
    deploy_type: str,
    num_master_wallets: int
):
    """
    This function does the following:
      1) Auto-queue any active, unassigned (and unqueued) jobs so they can be assigned.
      2) Enforces concurrency limit (args.max_concurrent_jobs).
      3) If we have capacity, we pop from the "queue" in DB (jobs that are queued == True and unassigned).
      4) We also check existing assigned jobs for timeouts or completion.
      5) If job times out (no tasks created in 30 min) => we set it inactive.
      6) Log how many jobs are active in total.

    'jobs_processing' is a local dict of job_id => asyncio.Task for run_master_task.
    The actual assignment is also in the DB so multiple Master processes don't step on each other.
    """

    db_adapter = DBAdapterClient(db_url, private_key)
    jobs_processing = {}

    while True:
        # ------------------------------------------------------
        # 1) Auto-queue any active, unassigned & unqueued jobs
        # ------------------------------------------------------
        unassigned_unqueued_active_jobs = await db_adapter.get_unassigned_unqueued_active_jobs()
        for job_obj in unassigned_unqueued_active_jobs:
            logger.info(f"[check_for_new_jobs] Auto-queuing active job {job_obj.id} since it's unassigned & unqueued.")
            await db_adapter.update_job_queue_status(
                job_id=job_obj.id,
                new_queued=True,
                assigned_master_id=None
            )

        # Remove any local tasks if they're done
        done_list = []
        for job_id, master_task in jobs_processing.items():
            if master_task.done():
                done_list.append(job_id)
        for d in done_list:
            jobs_processing.pop(d, None)

        # get all jobs assigned to this Master
        assigned_jobs_in_db = await db_adapter.get_jobs_assigned_to_master(MASTER_ID)

        # Ensure assigned jobs from DB are also tracked locally
        # (Now replicating the same steps as the queued-jobs block via helper)
        for assigned_job in assigned_jobs_in_db:
            # If job is active but missing from local tracking => handle
            if assigned_job.active and assigned_job.id not in jobs_processing:
                logger.info(
                    f"[check_for_new_jobs] Found assigned job {assigned_job.id} in DB but missing locally =>"
                    " launching SOT/workers + Master task."
                )
                await handle_newly_assigned_job(
                    db_adapter=db_adapter,
                    job_obj=assigned_job,
                    deploy_type=deploy_type,
                    db_url=db_url,
                    master_id=MASTER_ID,
                    jobs_processing=jobs_processing,
                    deposit_amount=999999999,  # replicate the same deposit logic
                    unqueue_if_needed=False     # because it's already assigned
                )

        current_count = len(jobs_processing)
        capacity = args.max_concurrent_jobs - current_count

        # ------------------------------------------------------
        # 2) If we have capacity, fetch some queued jobs from DB
        # ------------------------------------------------------
        if capacity > 0:
            queued_jobs = await db_adapter.get_unassigned_queued_jobs()
            if queued_jobs:
                for job_obj in queued_jobs:
                    if capacity <= 0:
                        break
                    # Reuse the same DRY function
                    await handle_newly_assigned_job(
                        db_adapter=db_adapter,
                        job_obj=job_obj,
                        deploy_type=deploy_type,
                        db_url=db_url,
                        master_id=MASTER_ID,
                        jobs_processing=jobs_processing,
                        deposit_amount=999999999,
                        unqueue_if_needed=True
                    )
                    capacity -= 1

        # 3) check assigned active jobs for inactivity timeouts
        assigned_active_jobs = await db_adapter.get_jobs_assigned_to_master(MASTER_ID)
        if assigned_active_jobs is None:
            assigned_active_jobs = []
        for job_obj in assigned_active_jobs:
            if not job_obj.active:
                # job might be toggled inactive => cancel locally
                if job_obj.id in jobs_processing:
                    logger.info(f"[check_for_new_jobs] job {job_obj.id} was inactivated => stopping local Master.")
                    jobs_processing[job_obj.id].cancel()
                    jobs_processing.pop(job_obj.id, None)
                continue

            # check inactivity (30 min)
            job_state = await db_adapter.get_master_state_for_job(job_obj.id)
            last_t = job_state.get("last_task_creation_time", None)
            if last_t is not None and (time.time() - last_t > TIMEOUT_SECONDS):
                logger.info(f"[check_for_new_jobs] Job {job_obj.id} timed out => marking inactive.")
                await db_adapter.update_job_active(job_obj.id, False)
                if job_obj.id in jobs_processing:
                    jobs_processing[job_obj.id].cancel()
                    jobs_processing.pop(job_obj.id, None)

        # 4) Clean up any local jobs that ended
        finished = []
        for job_id, master_task in jobs_processing.items():
            if master_task.done():
                finished.append(job_id)
        for job_id in finished:
            logger.info(f"[check_for_new_jobs] Local Master sees job {job_id} ended => removing from local dict.")
            jobs_processing.pop(job_id, None)

        # 5) Log how many are "active" overall
        jobs_in_progress = await db_adapter.get_jobs_in_progress()
        if jobs_in_progress:
            active_jobs_count = sum(1 for j in jobs_in_progress if j.active)
            logger.info(f"[check_for_new_jobs] Currently {active_jobs_count} active job(s) in total.")
        else:
            logger.info("[check_for_new_jobs] No jobs_in_progress found at the moment.")

        # small sleep
        await asyncio.sleep(2)
