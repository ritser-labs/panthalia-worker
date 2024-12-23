# spl/db/server/adapter/jobs_plugins.py

from sqlalchemy import select, update
from ....models import AsyncSessionLocal, Job, Plugin, Subnet

class DBAdapterJobsPluginsMixin:
    async def create_job(self, name: str, plugin_id: int, subnet_id: int, sot_url: str, iteration: int):
        async with AsyncSessionLocal() as session:
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
        async with AsyncSessionLocal() as session:
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
        async with AsyncSessionLocal() as session:
            new_plugin = Plugin(
                name=name,
                code=code
            )
            session.add(new_plugin)
            await session.commit()
            await session.refresh(new_plugin)
            return new_plugin.id

    async def get_plugin(self, plugin_id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Plugin).filter_by(id=plugin_id)
            result = await session.execute(stmt)
            plugin = result.scalar_one_or_none()
            return plugin

    async def get_subnet_using_address(self, address: str):
        async with AsyncSessionLocal() as session:
            stmt = select(Subnet).filter_by(address=address)
            result = await session.execute(stmt)
            subnet = result.scalar_one_or_none()
            return subnet

    async def get_subnet(self, subnet_id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Subnet).filter_by(id=subnet_id)
            result = await session.execute(stmt)
            subnet = result.scalar_one_or_none()
            return subnet

    async def get_jobs_without_instances(self):
        from ....models import Instance
        async with AsyncSessionLocal() as session:
            stmt = (
                select(Job)
                .outerjoin(Instance, Job.id == Instance.job_id)
                .filter(Instance.id == None)
            )
            result = await session.execute(stmt)
            jobs_without_instances = result.scalars().all()
            return jobs_without_instances

    async def get_plugins(self):
        async with AsyncSessionLocal() as session:
            stmt = select(Plugin)
            result = await session.execute(stmt)
            plugins = result.scalars().all()
            return plugins

    ####################################################
    # NEW: Generic GET/UPDATE state_json for a given job
    ####################################################

    async def get_job_state(self, job_id: int) -> dict:
        """
        Return the dictionary stored in Job.state_json
        """
        async with AsyncSessionLocal() as session:
            stmt = select(Job).where(Job.id == job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if not job:
                return {}
            if not job.state_json:
                return {}
            return job.state_json

    async def update_job_state(self, job_id: int, new_state: dict):
        """
        Overwrite Job.state_json with new_state
        """
        async with AsyncSessionLocal() as session:
            stmt = update(Job).where(Job.id == job_id).values(state_json=new_state)
            await session.execute(stmt)
            await session.commit()
