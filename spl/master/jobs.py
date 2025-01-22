# file: spl/master/jobs.py

import asyncio
import logging
import socket
import os
import time
import signal  # <-- ADDED for local process termination

from ..db.db_adapter_client import DBAdapterClient
from ..models import ServiceType
from .deploy import launch_sot, launch_workers, run_master_task
from .config import args

logger = logging.getLogger(__name__)

# For the multi-master scenario, define a unique ID for this Master instance
MASTER_ID = f"{socket.gethostname()}_{os.getpid()}"
TIMEOUT_SECONDS = 30 * 60  # 30 minutes

async def terminate_local_process(db_adapter: DBAdapterClient, instance_id: int):
    """
    For local deployments only: kills the subprocess given by instance.process_id (if present).
    This frees local resources after a job is done or inactive.
    """
    inst = await db_adapter.get_instance_by_id(instance_id)  # or get_instance(...) if you have such a method
    if not inst:
        return

    pid = inst.process_id
    if pid and pid > 0:
        try:
            logger.info(f"[terminate_local_process] Killing local process pid={pid} for instance {instance_id}")
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            logger.warning(f"[terminate_local_process] No process found with pid={pid}, ignoring.")
        except Exception as e:
            logger.error(f"[terminate_local_process] Error killing pid={pid}: {e}")

async def release_instances_for_job(db_adapter: DBAdapterClient, job_id: int):
    """
    Resets job_id=None for every Instance that was reserved by this job_id,
    thereby freeing those slots for future jobs.
    Also kills local processes (if any) for local mode by calling terminate_local_process(...).
    """
    allocated_instances = await db_adapter.get_instances_by_job(job_id)
    if not allocated_instances:
        return

    freed_count = 0
    for inst in allocated_instances:
        instance_id = inst.id if hasattr(inst, 'id') else inst['id']
        # Kill local processes if it's a local deployment (only if process_id is set)
        if inst.process_id and inst.process_id > 0:
            await terminate_local_process(db_adapter, instance_id)

        # Now free the instance by setting job_id=None
        await db_adapter.update_instance(instance_id, job_id=None)
        freed_count += 1

    logger.info(f"[release_instances_for_job] Freed {freed_count} instance(s) for job {job_id}")


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
      - Optionally unqueue the job (job.queued=False, assigned_master_id=master_id)
      - Admin-deposit for demonstration
      - Launch the SOT + workers
      - Start the local Master logic as an asyncio.Task
    """
    if unqueue_if_needed:
        logger.info(f"[handle_newly_assigned_job] Assigning job {job_obj.id} to {master_id}")
        success = await db_adapter.update_job_queue_status(
            job_id=job_obj.id,
            new_queued=False,
            assigned_master_id=master_id
        )
        if not success:
            logger.warning(f"[handle_newly_assigned_job] Failed to assign job {job_obj.id} to {master_id}")
            return

    # For demonstration, deposit enough so the job owner can place tasks/bids
    await db_adapter.admin_deposit_account(job_obj.user_id, deposit_amount)

    # 1) Launch SOT
    sot_db_obj, sot_url = await launch_sot(db_adapter, job_obj, db_url)
    # 2) Launch Workers
    await launch_workers(db_adapter, job_obj, db_url, args.num_workers)

    # 3) Start local Master logic in the background
    master_task = asyncio.create_task(
        run_master_task(job_id=job_obj.id, db_adapter=db_adapter, max_iters=float('inf'))
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
    The main loop that:
      1) Auto-queues any unassigned+active jobs
      2) Assigns or re-assigns queued jobs to this Master if capacity
      3) Checks inactivity timeouts
      4) If a job becomes inactive => we free its instances
    """
    db_adapter = DBAdapterClient(db_url, private_key)
    jobs_processing = {}

    while True:
        # 1) Auto-queue any active, unassigned & unqueued jobs
        unassigned_unqueued_active_jobs = await db_adapter.get_unassigned_unqueued_active_jobs()
        for job_obj in unassigned_unqueued_active_jobs:
            logger.info(f"[check_for_new_jobs] Auto-queuing active job {job_obj.id} since it's unassigned & unqueued.")
            await db_adapter.update_job_queue_status(job_obj.id, new_queued=True, assigned_master_id=None)

        # Remove any local tasks that are done
        done_list = []
        for job_id, master_task in jobs_processing.items():
            if master_task.done():
                done_list.append(job_id)
        for d in done_list:
            jobs_processing.pop(d, None)

        # Get all jobs assigned to this Master
        assigned_jobs_in_db = await db_adapter.get_jobs_assigned_to_master(MASTER_ID)
        if not assigned_jobs_in_db:
            assigned_jobs_in_db = []

        # Ensure assigned DB jobs are tracked locally
        for assigned_job in assigned_jobs_in_db:
            # If job is active but missing locally => spin up SOT/workers + Master
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
                    deposit_amount=999999999,
                    unqueue_if_needed=False
                )

        current_count = len(jobs_processing)
        capacity = args.max_concurrent_jobs - current_count

        # 2) If we have capacity, fetch some queued jobs from DB
        if capacity > 0:
            queued_jobs = await db_adapter.get_unassigned_queued_jobs()
            if queued_jobs:
                for job_obj in queued_jobs:
                    if capacity <= 0:
                        break
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

        # 3) Check assigned active jobs for inactivity or if they became inactive
        assigned_active_jobs = await db_adapter.get_jobs_assigned_to_master(MASTER_ID)
        if assigned_active_jobs is None:
            assigned_active_jobs = []

        for job_obj in assigned_active_jobs:
            if not job_obj.active:
                # job is now inactive => stop local tasks, free Instances
                if job_obj.id in jobs_processing:
                    logger.info(f"[check_for_new_jobs] job {job_obj.id} was inactivated => stopping local Master.")
                    jobs_processing[job_obj.id].cancel()
                    jobs_processing.pop(job_obj.id, None)

                # FREE the Instances for that job
                await release_instances_for_job(db_adapter, job_obj.id)
                # done checking this job
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
                # FREE the Instances since it's now inactive
                await release_instances_for_job(db_adapter, job_obj.id)

        # 4) Clean up any local tasks that ended
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
            logger.info(f"[check_for_new_jobs] Currently {active_jobs_count} active job(s) total.")
        else:
            logger.info("[check_for_new_jobs] No jobs_in_progress found at the moment.")

        # small pause
        await asyncio.sleep(2)
