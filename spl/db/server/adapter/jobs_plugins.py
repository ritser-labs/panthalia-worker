# spl/db/server/adapter/jobs_plugins.py

import logging
from sqlalchemy import select, update
from ....models import Job, Plugin, Subnet, Task, TaskStatus
from sqlalchemy.orm import joinedload

logger = logging.getLogger(__name__)

class DBAdapterJobsPluginsMixin:
    async def create_job(self, name: str, plugin_id: int, subnet_id: int, sot_url: str, iteration: int, initial_state_url: str):
        async with self.get_async_session() as session:
            new_job = Job(
                name=name,
                plugin_id=plugin_id,
                subnet_id=subnet_id,
                user_id=self.get_user_id(),
                sot_url=sot_url,
                iteration=iteration,
                done=False,
                initial_state_url=initial_state_url
            )
            session.add(new_job)
            await session.commit()
            await session.refresh(new_job)
            return new_job.id

    async def create_subnet(self, dispute_period: int, solve_period: int, stake_multiplier: float):
        async with self.get_async_session() as session:
            new_subnet = Subnet(
                dispute_period=dispute_period,
                solve_period=solve_period,
                stake_multiplier=stake_multiplier
            )
            session.add(new_subnet)
            await session.commit()
            await session.refresh(new_subnet)
            return new_subnet.id

    async def create_plugin(self, name: str, code: str):
        async with self.get_async_session() as session:
            new_plugin = Plugin(
                name=name,
                code=code
            )
            session.add(new_plugin)
            await session.commit()
            await session.refresh(new_plugin)
            return new_plugin.id

    async def get_plugin(self, plugin_id: int):
        async with self.get_async_session() as session:
            stmt = select(Plugin).filter_by(id=plugin_id)
            result = await session.execute(stmt)
            plugin = result.scalar_one_or_none()
            return plugin

    async def get_subnet_using_address(self, address: str):
        async with self.get_async_session() as session:
            stmt = select(Subnet).filter_by(address=address)
            result = await session.execute(stmt)
            subnet = result.scalar_one_or_none()
            return subnet

    async def get_subnet(self, subnet_id: int):
        async with self.get_async_session() as session:
            stmt = select(Subnet).filter_by(id=subnet_id)
            result = await session.execute(stmt)
            subnet = result.scalar_one_or_none()
            return subnet

    async def get_jobs_without_instances(self):
        from ....models import Instance
        async with self.get_async_session() as session:
            stmt = (
                select(Job)
                .outerjoin(Instance, Job.id == Instance.job_id)
                .filter(Instance.id == None)
            )
            result = await session.execute(stmt)
            jobs_without_instances = result.scalars().all()
            return jobs_without_instances

    async def get_plugins(self):
        async with self.get_async_session() as session:
            stmt = select(Plugin)
            result = await session.execute(stmt)
            plugins = result.scalars().all()
            return plugins

    async def get_master_job_state(self, job_id: int) -> dict:
        """
        Returns the master_state_json for the specified job. 
        If job not found or no master_state_json, returns {}.
        """
        async with self.get_async_session() as session:
            job = await session.get(Job, job_id)
            if not job:
                logger.warning(f"[get_master_job_state] job_id={job_id} not found.")
                return {}
            return job.master_state_json or {}

    async def update_master_job_state(self, job_id: int, new_state: dict) -> bool:
        """
        Overwrites the master_state_json for the specified job.
        Returns True on success, False if the job does not exist.
        """
        async with self.get_async_session() as session:
            job = await session.get(Job, job_id)
            if not job:
                logger.warning(f"[update_master_job_state] job_id={job_id} not found.")
                return False

            job.master_state_json = new_state
            await session.commit()
            return {'success': True}

    async def get_sot_job_state(self, job_id: int) -> dict:
        """
        Returns the sot_state_json for the specified job.
        If job not found or no sot_state_json, returns {}.
        """
        async with self.get_async_session() as session:
            job = await session.get(Job, job_id)
            if not job:
                logger.warning(f"[get_sot_job_state] job_id={job_id} not found.")
                return {}
            return job.sot_state_json or {}

    async def update_sot_job_state(self, job_id: int, new_state: dict) -> bool:
        """
        Overwrites the sot_state_json for the specified job.
        Returns True on success, False if the job does not exist.
        """
        async with self.get_async_session() as session:
            job = await session.get(Job, job_id)
            if not job:
                logger.warning(f"[update_sot_job_state] job_id={job_id} not found.")
                return False

            job.sot_state_json = new_state
            await session.commit()
            return {'success': True}

    async def update_job_active(self, job_id: int, new_active: bool):
        """
        Overwrite job.active. If new_active=True, also set job.queued=True
        and assigned_master_id=None so the Master will pick it up.
        """
        async with self.get_async_session() as session:
            job = await session.get(Job, job_id)
            if not job:
                logger.warning(f"[update_job_active] job_id={job_id} not found.")
                return

            job.active = new_active
            if new_active:
                # <--- This is the crucial fix so Master sees the job
                job.queued = True
                job.assigned_master_id = None

            await session.commit()

        # If we set the job to inactive, also set any SOT for that job to inactive:
        if new_active is False:
            sot = await self.get_sot_by_job_id(job_id)
            if sot and sot.active:  # Only if it's currently active
                await self.update_sot_active(sot.id, False)

    async def get_jobs_in_progress(self):
        """
        Returns all jobs that are EITHER:
          - active == True, OR
          - have any task that is not resolved (i.e. statuses not in [ResolvedCorrect, ResolvedIncorrect]).
        This ensures we keep finishing tasks even if the owner toggles job.active==False.
        """
        async with self.get_async_session() as session:
            # We'll define which statuses are considered 'unresolved'
            unresolved_statuses = [
                TaskStatus.SelectingSolver,
                TaskStatus.SolverSelected,
                TaskStatus.Checking,
                TaskStatus.SanityCheckPending
            ]
            
            stmt = (
                select(Job)
                .outerjoin(Task, Task.job_id == Job.id)
                .options(joinedload(Job.tasks))  # optional, if you want tasks preloaded
                .where(
                    (Job.active == True) |
                    (Task.status.in_(unresolved_statuses))
                )
                .distinct()
            )
            result = await session.execute(stmt)
            jobs = result.scalars().unique().all()
            return jobs

    async def update_job_queue_status(self, job_id: int, new_queued: bool, assigned_master_id: str | None) -> bool:
        """
        Set job.queued = new_queued, job.assigned_master_id = assigned_master_id
        (or clear it if assigned_master_id is None).
        """
        async with self.get_async_session() as session:
            job = await session.get(Job, job_id)
            if not job:
                logger.warning(f"[update_job_queue_status] job_id={job_id} not found.")
                return False

            job.queued = new_queued
            job.assigned_master_id = assigned_master_id  # may be None
            await session.commit()
            return True

    async def get_unassigned_queued_jobs(self):
        """
        Return all jobs where queued==True and assigned_master_id==None
        """
        async with self.get_async_session() as session:
            stmt = (
                select(Job)
                .where(Job.queued == True)
                .where(Job.assigned_master_id == None)
            )
            result = await session.execute(stmt)
            jobs = result.scalars().all()
            return jobs

    async def get_jobs_assigned_to_master(self, master_id: str):
        """
        Return all jobs assigned to master_id, regardless of queued or not.
        Usually we only care about those that are still active or partially done.
        """
        async with self.get_async_session() as session:
            stmt = (
                select(Job)
                .where(Job.assigned_master_id == master_id)
            )
            result = await session.execute(stmt)
            jobs = result.scalars().all()
            return jobs

    # The following placeholders assume you have update_sot_active or get_sot_by_job_id, etc.
    async def update_sot_active(self, sot_id: int, new_active: bool):
        """
        If you have a separate 'active' field for SOT, you'd do it here.
        For now, we just log a placeholder.
        """
        # Example placeholder
        logger.info(f"[update_sot_active] Setting SOT {sot_id} active={new_active} (Not fully implemented)")

    async def get_sot_by_job_id(self, job_id: int):
        async with self.get_async_session() as session:
            from ....models import Sot
            stmt = select(Sot).filter_by(job_id=job_id)
            result = await session.execute(stmt)
            sot = result.scalar_one_or_none()
            return sot

    async def get_unassigned_unqueued_active_jobs(self):
        """
        Return all jobs with:
          - active == True
          - assigned_master_id IS NULL
          - queued == False
        We'll use this to auto-set them queued=True so the Master can pick them up.
        """
        async with self.get_async_session() as session:
            stmt = (
                select(Job)
                .where(
                    Job.active == True,
                    Job.queued == False,
                    Job.assigned_master_id.is_(None)
                )
            )
            result = await session.execute(stmt)
            jobs = result.scalars().all()
            return jobs
