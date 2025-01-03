from sqlalchemy import select, update
from ....models import Instance, ServiceType
from ....db.init import AsyncSessionLocal


class DBAdapterInstancesMixin:
    async def create_instance(self, name: str, service_type: ServiceType, job_id: int | None, private_key: str | None, pod_id: str | None, process_id: int | None):
        async with AsyncSessionLocal() as session:
            new_instance = Instance(
                name=name,
                service_type=service_type,
                job_id=job_id,
                private_key=private_key,
                pod_id=pod_id,
                process_id=process_id
            )
            session.add(new_instance)
            await session.commit()
            await session.refresh(new_instance)
            return new_instance.id

    async def get_instance_by_service_type(self, service_type: ServiceType, job_id: int | None = None):
        async with AsyncSessionLocal() as session:
            stmt = select(Instance).filter_by(service_type=service_type)
            if job_id is not None:
                stmt = stmt.filter_by(job_id=job_id)
            result = await session.execute(stmt)
            instance = result.scalars().first()
            return instance

    async def get_instances_by_job(self, job_id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Instance).filter_by(job_id=job_id)
            result = await session.execute(stmt)
            instances = result.scalars().all()
            return instances

    async def get_all_instances(self):
        async with AsyncSessionLocal() as session:
            stmt = select(Instance)
            result = await session.execute(stmt)
            instances = result.scalars().all()
            return instances

    async def update_instance(self, instance_id: int, **kwargs):
        async with AsyncSessionLocal() as session:
            stmt = update(Instance).where(Instance.id == instance_id).values(**kwargs)
            await session.execute(stmt)
            await session.commit()
            return True
