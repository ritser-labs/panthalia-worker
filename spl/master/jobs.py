import asyncio
import logging
import socket
import os
import time
import signal

from ..db.db_adapter_client import DBAdapterClient
from ..models import ServiceType
from .deploy import launch_sot, launch_workers, run_master_task
from .config import args
from .main_logic import Master

logger = logging.getLogger(__name__)

# For the multi-master scenario, define a unique ID for this Master instance
MASTER_ID = f"{socket.gethostname()}_{os.getpid()}"
TIMEOUT_SECONDS = 30 * 60  # 30 minutes

async def terminate_local_process(db_adapter: DBAdapterClient, instance_id: int):
    """
    For local deployments only: kills the subprocess given by instance.process_id (if present).
    This frees local resources after a job is done or inactive.
    """
    inst = await db_adapter.get_instance(instance_id)  # or get_instance(...) if you have such a method
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
    thereby freeing those slots for future jobs. Also kills local processes (if any)
    if we're in local mode.
    """
    allocated_instances = await db_adapter.get_instances_by_job(job_id)
    if not allocated_instances:
        return

    freed_count = 0
    for inst in allocated_instances:
        instance_id = inst.id
        # Kill local processes if it's a local deployment
        if inst.process_id and inst.process_id > 0:
            await terminate_local_process(db_adapter, instance_id)

        # Free the instance
        await db_adapter.update_instance({
            'instance_id': instance_id,
            'job_id': None
        })
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
    Helper to set job.queued=False, assigned_master_id=..., deposit funds, launch SOT + workers,
    then start the local Master logic as an asyncio.Task.
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

    # 1) Launch SOT
    sot_db_obj, sot_url = await launch_sot(db_adapter, job_obj, db_url)
    # 2) Launch Workers
    await launch_workers(db_adapter, job_obj, db_url, args.num_workers)

    # 3) Start local Master logic
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
      1) Auto-queues any unassigned+active jobs.
      2) Assigns or re-assigns queued jobs to this Master if capacity remains.
      3) Manages inactivity/timeouts.
      4) Once a job becomes inactive, let its Master finish gracefully. Only
         after the Master is truly done do we free local processes.
    """
    db_adapter = DBAdapterClient(db_url, private_key)

    # A dict mapping job_id -> an asyncio.Task representing this Master’s local run
    jobs_processing = {}

    while True:
        # (A) Auto-queue any active, unassigned, unqueued jobs:
        unassigned_unqueued_active_jobs = await db_adapter.get_unassigned_unqueued_active_jobs()
        for job_obj in unassigned_unqueued_active_jobs or []:
            logger.info(f"[check_for_new_jobs] Auto-queuing active job {job_obj.id}.")
            await db_adapter.update_job_queue_status(job_obj.id, new_queued=True, assigned_master_id=None)

        # Remove any local tasks that finished:
        done_jobs = []
        for j_id, master_task in jobs_processing.items():
            if master_task.done():
                done_jobs.append(j_id)
        for dj in done_jobs:
            jobs_processing.pop(dj, None)

        # Fetch all jobs assigned to this Master:
        assigned_jobs_in_db = await db_adapter.get_jobs_assigned_to_master(MASTER_ID) or []

        # For any assigned job that *we* have no local task for, spin up SOT/Workers + Master
        for assigned_job in assigned_jobs_in_db:
            if assigned_job.active and assigned_job.id not in jobs_processing:
                logger.info(f"[check_for_new_jobs] Found assigned job {assigned_job.id}, launching Master tasks.")
                # Launch SOT + workers => run Master
                # Re-use your code that does so in e.g. handle_newly_assigned_job
                await handle_newly_assigned_job(
                    db_adapter=db_adapter,
                    job_obj=assigned_job,
                    deploy_type=deploy_type,
                    db_url=db_url,
                    master_id=MASTER_ID,
                    jobs_processing=jobs_processing,
                    unqueue_if_needed=False
                )

        # Check capacity vs. queued jobs
        current_count = len(jobs_processing)
        capacity = args.max_concurrent_jobs - current_count
        if capacity > 0:
            queued_jobs = await db_adapter.get_unassigned_queued_jobs() or []
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
                    unqueue_if_needed=True
                )
                capacity -= 1

        # Now handle assigned jobs (some possibly inactive). 
        assigned_active_jobs = await db_adapter.get_jobs_assigned_to_master(MASTER_ID) or []
        for job_obj in assigned_active_jobs:
            master_task = jobs_processing.get(job_obj.id, None)

            if job_obj.active:
                #
                # Possibly check for inactivity/timeouts:
                #
                job_state = await db_adapter.get_master_state_for_job(job_obj.id)
                last_t = job_state.get("last_task_creation_time", None)
                if last_t is not None and (time.time() - last_t > TIMEOUT_SECONDS):
                    logger.info(f"[check_for_new_jobs] job {job_obj.id} timed out => marking inactive.")
                    await db_adapter.update_job_active(job_obj.id, False)

            else:
                # job is inactive => let the Master finish on its own
                if master_task:
                    if master_task.done():
                        # The Master has finished => we can free local processes
                        logger.info(f"[check_for_new_jobs] job {job_obj.id} is inactive & Master is done => freeing instances.")
                        # Optionally remove unmatched orders here (if desired):
                        unmatched_orders = await db_adapter.get_unmatched_orders_for_job(job_obj.id)
                        for order in unmatched_orders:
                            await db_adapter.delete_order(order.id)

                        await release_instances_for_job(db_adapter, job_obj.id)
                        jobs_processing.pop(job_obj.id, None)

                    else:
                        # The Master is still running => do nothing; let it complete
                        pass
                else:
                    # If no local task => no reason to keep instance(s). Possibly free them.
                    # This scenario could happen if we never started the Master for it.
                    unmatched_orders = await db_adapter.get_unmatched_orders_for_job(job_obj.id)
                    for order in unmatched_orders:
                        await db_adapter.delete_order(order.id)
                    await release_instances_for_job(db_adapter, job_obj.id)

        # Optional debugging:
        jobs_in_progress = await db_adapter.get_jobs_in_progress() or []
        if jobs_in_progress:
            active_count = sum(1 for j in jobs_in_progress if j.active)
            logger.debug(f"[check_for_new_jobs] Currently {active_count} active job(s).")
        else:
            logger.debug("[check_for_new_jobs] No jobs_in_progress found.")

        # final short sleep
        await asyncio.sleep(2)
