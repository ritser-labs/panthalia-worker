# spl/db/server/adapter/jobs_plugins.py

import logging
from sqlalchemy import select, update
from ....models import Job, Plugin, Subnet, Task, TaskStatus
from sqlalchemy.orm import joinedload


logger = logging.getLogger(__name__)

class DBAdapterJobsPluginsMixin:
    async def create_job(self, name: str, plugin_id: int, subnet_id: int, sot_url: str, iteration: int):
        async with self.get_async_session() as session:
            new_job = Job(
                name=name,
                plugin_id=plugin_id,
                subnet_id=subnet_id,
                user_id=self.get_user_id(),
                sot_url=sot_url,
                iteration=iteration,
                done=False
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

    ####################################################
    # *** CHANGED ***: New debug logs in get_job_state
    ####################################################
    async def get_job_state(self, job_id: int) -> dict:
        """
        Return the dictionary stored in Job.state_json
        """
        logger.debug(f"[jobs_plugins] get_job_state called with job_id={job_id}")
        async with self.get_async_session() as session:
            stmt = select(Job).where(Job.id == job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if not job:
                logger.debug(f"[jobs_plugins] get_job_state => no job found for job_id={job_id}, returning {{}}")
                return {}
            if not job.state_json:
                logger.debug(f"[jobs_plugins] get_job_state => job found but state_json is None, returning {{}}")
                return {}
            logger.debug(f"[jobs_plugins] get_job_state => returning: {job.state_json}")
            return job.state_json

    ####################################################
    # *** CHANGED ***: New debug logs in update_job_state
    ####################################################
    async def update_job_state(self, job_id: int, new_state: dict):
        """
        Overwrite Job.state_json with new_state
        """
        logger.debug(f"[jobs_plugins] update_job_state called with job_id={job_id}, new_state={new_state}")
        async with self.get_async_session() as session:
            stmt = update(Job).where(Job.id == job_id).values(state_json=new_state)
            result = await session.execute(stmt)
            await session.commit()
            rows_updated = result.rowcount
            logger.debug(f"[jobs_plugins] update_job_state => rows_updated={rows_updated}")
            logger.debug(f"[jobs_plugins] update_job_state => done for job_id={job_id}")

    async def update_job_active(self, job_id: int, new_active: bool):
        async with self.get_async_session() as session:
            stmt = (
                update(Job)
                .where(Job.id == job_id)
                .values(active=new_active)
            )
            await session.execute(stmt)
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
            
            # We do a LEFT JOIN on Task. If job is active OR there's at least one unresolved task => included
            # Note: Using a subquery or a distinct approach is typical. Example below is fairly direct:
            
            stmt = (
                select(Job)
                .outerjoin(Task, Task.job_id == Job.id)
                .options(joinedload(Job.tasks))  # optional, if you want tasks preloaded
                .where(
                    # job.active==True OR (some tasks with 'unresolved' statuses)
                    (Job.active == True) |
                    (Task.status.in_(unresolved_statuses))
                )
                .distinct()
            )
            result = await session.execute(stmt)
            jobs = result.scalars().unique().all()
            return jobs