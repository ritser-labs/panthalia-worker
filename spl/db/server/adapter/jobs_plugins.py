# spl/db/server/adapter/jobs_plugins.py

import logging
from sqlalchemy import select, update, desc
from ....models import Job, Plugin, Subnet, Task, TaskStatus, Sot, PluginReviewStatus
from sqlalchemy.orm import joinedload

logger = logging.getLogger(__name__)

MIN_REPLICATE_PROB = 0.0
MAX_REPLICATE_PROB = 0.4

class DBAdapterJobsPluginsMixin:
    async def create_job(self, name: str, plugin_id: int, sot_url: str,
                         iteration: int, limit_price: int, initial_state_url: str='', 
                         replicate_prob: float=0.1):
        async with self.get_async_session() as session:
            if replicate_prob < MIN_REPLICATE_PROB or replicate_prob > MAX_REPLICATE_PROB:
                logger.warning(f"[create_job] replicate_prob={replicate_prob} out of range.")
                return None
            plugin = await session.get(Plugin, plugin_id)
            new_job = Job(
                name=name,
                plugin_id=plugin_id,
                subnet_id=plugin.subnet_id,
                user_id=self.get_user_id(),
                sot_url=sot_url,
                iteration=iteration,
                done=False,
                initial_state_url=initial_state_url,
                replicate_prob=replicate_prob,
                limit_price=limit_price
            )
            session.add(new_job)
            await session.commit()
            await session.refresh(new_job)
            return new_job.id

    async def update_job_limit_price(self, job_id: int, new_limit_price: int):
        async with self.get_async_session() as session:
            job = await session.get(Job, job_id)
            if not job:
                logger.warning(f"[update_job_limit_price] job_id={job_id} not found.")
                return {"error": "Job not found"}
            if job.user_id != self.get_user_id():
                logger.warning(f"[update_job_limit_price] job_id={job_id} does not belong to user {self.get_user_id()}.")
                return {"error": "Not authorized"}
            if not job.active:
                logger.warning(f"[update_job_limit_price] job_id={job_id} is not active.")
                return {"error": "Job is not active"}
            job.limit_price = new_limit_price
            await session.commit()
            return {"success": True}

    async def create_subnet(self, dispute_period: int, solve_period: int, stake_multiplier: float, target_price: float=1, description: str=''):
        async with self.get_async_session() as session:
            new_subnet = Subnet(
                dispute_period=dispute_period,
                solve_period=solve_period,
                stake_multiplier=stake_multiplier,
                target_price=target_price,
                description=description
            )
            session.add(new_subnet)
            await session.commit()
            await session.refresh(new_subnet)
            return new_subnet.id
    
    async def set_subnet_target_price(self, subnet_id: int, target_price: float):
        async with self.get_async_session() as session:
            stmt = (
                update(Subnet)
                .where(Subnet.id == subnet_id)
                .values(target_price=target_price)
            )
            await session.execute(stmt)
            await session.commit()

    async def create_plugin(self, name: str, code: str, subnet_id: int):
        async with self.get_async_session() as session:
            new_plugin = Plugin(
                name=name,
                code=code,
                subnet_id=subnet_id,
                user_id=self.get_user_id()
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
    
    async def get_subnet_of_job(self, job_id: int):
        async with self.get_async_session() as session:
            stmt = (
                select(Job)
                .options(joinedload(Job.subnet))  # Eagerly load the subnet relationship
                .filter_by(id=job_id)
            )
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            return job.subnet if job else None

    async def get_subnet_of_plugin(self, plugin_id: int):
        async with self.get_async_session() as session:
            stmt = (
                select(Plugin)
                .options(joinedload(Plugin.subnet))  # Eagerly load the subnet relationship
                .filter_by(id=plugin_id)
            )
            result = await session.execute(stmt)
            plugin = result.scalar_one_or_none()
            return plugin.subnet if plugin else None

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

    async def get_plugins(self, offset: int = 0, limit: int = 20):
        async with self.get_async_session() as session:
            stmt = (
                select(Plugin)
                .where(Plugin.user_id == self.get_user_id())  # filter to only the current user's plugins
                .offset(offset)
                .limit(limit)
            )
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

    async def update_job_active(self, job_id: int, active: bool):
        """
        Overwrite job.active. If active=True, also set job.queued=True
        and assigned_master_id=None so the Master will pick it up.
        """
        async with self.get_async_session() as session:
            job = await session.get(Job, job_id)
            if not job:
                logger.warning(f"[update_job_active] job_id={job_id} not found.")
                return

            job.active = active
            if active:
                # <--- This is the crucial fix so Master sees the job
                job.queued = True
                job.assigned_master_id = None

            await session.commit()

        # If we set the job to inactive, also set any SOT for that job to inactive:
        if active is False:
            sot = await self.get_sot_by_job_id(job_id)
            if sot and sot.active:  # Only if it's currently active
                await self.update_sot_active(sot.id, False)
    
    async def stop_job(self, job_id: int):
        async with self.get_async_session() as session:
            user_id = self.get_user_id()
            job = await session.get(Job, job_id)
            if not job:
                logger.warning(f"[stop_job] job_id={job_id} not found.")
                return
            if job.user_id != user_id:
                logger.warning(f"[stop_job] job_id={job_id} does not belong to user_id={user_id}.")
                return
            
            job.active = False
            await session.commit()
            
            # Also stop the SOT if it exists
            sot = await self.get_sot_by_job_id(job_id)
            if sot and sot.active:
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
                TaskStatus.SolutionSubmitted
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

    async def update_sot_active(self, sot_id: int, new_active: bool):
        async with self.get_async_session() as session:
            stmt = (
                update(Sot)
                .where(Sot.id == sot_id)
                .values(active=new_active)
            )
            await session.execute(stmt)
            await session.commit()

        self.logger.info(f"[update_sot_active] SOT {sot_id} set active={new_active}")


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

    async def update_plugin_review_status(self, plugin_id: int, review_status: PluginReviewStatus):
        async with self.get_async_session() as session:
            stmt = (
                update(Plugin)
                .where(Plugin.id == plugin_id)
                .values(review_status=review_status)
            )
            await session.execute(stmt)
            await session.commit()
            return plugin_id
    
    async def list_plugins(self, offset: int = 0, limit: int = 20, review_status: str = 'all'):
        """
        Returns a paginated list of Plugin records.
        
        :param offset: Number of records to skip.
        :param limit: Maximum number of records to return.
        :param review_status: (Optional) a string value, one of 'all', 'unreviewed', 'approved', or 'rejected'
                              (case insensitive). If provided, only plugins with that status are returned.
        :return: A list of Plugin objects.
        :raises ValueError: if an invalid review_status is provided.
        """
        stmt = select(Plugin)
        if review_status != 'all':
            try:
                # Convert the provided string (e.g. "approved") to the corresponding enum value.
                status_enum = PluginReviewStatus(review_status.lower())
            except Exception as e:
                raise ValueError("Invalid review_status filter; allowed values are 'all', 'unreviewed', 'approved', 'rejected'.")
            stmt = stmt.where(Plugin.review_status == status_enum)
        stmt = stmt.offset(offset).limit(limit)
        async with self.get_async_session() as session:
            result = await session.execute(stmt)
            plugins = result.scalars().all()
            return plugins

    async def get_jobs_for_user(self, offset: int = 0, limit: int = 20):
        async with self.get_async_session() as session:
            stmt = (
                select(Job)
                .where(Job.user_id == self.get_user_id())
                .order_by(desc(Job.submitted_at))  # order by submission time; adjust as needed
                .offset(offset)
                .limit(limit)
            )
            result = await session.execute(stmt)
            jobs = result.scalars().all()
            # Return jobs as dictionaries so the route can jsonify them directly.
            return [job.as_dict() for job in jobs]
    
    async def get_subnets(self, offset: int = 0, limit: int = 20):
        async with self.get_async_session() as session:
            stmt = select(Subnet).offset(offset).limit(limit)
            result = await session.execute(stmt)
            subnets = result.scalars().all()
            # Convert each Subnet ORM object to a dict using its as_dict() method.
            return [s.as_dict() for s in subnets]
