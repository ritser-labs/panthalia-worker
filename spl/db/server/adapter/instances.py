# spl/db/server/adapter/instances.py

from sqlalchemy import select, update
from ....models import Instance, ServiceType
from ....models.enums import SlotType
from typing import List, Optional

class DBAdapterInstancesMixin:

    async def create_instance(
        self,
        name: str,
        service_type: ServiceType,
        job_id: int | None,
        private_key: str | None,
        pod_id: str | None,
        process_id: int | None,
        gpu_enabled: bool = False,
        slot_type: SlotType | None = None,
        connection_info: str | None = None   # <--- new param
    ):
        async with self.get_async_session() as session:
            new_instance = Instance(
                name=name,
                service_type=service_type,
                job_id=job_id,
                private_key=private_key,
                pod_id=pod_id,
                process_id=process_id,
                gpu_enabled=gpu_enabled,
                slot_type=slot_type,
                connection_info=connection_info  # store new field
            )
            session.add(new_instance)
            await session.commit()
            await session.refresh(new_instance)
            return new_instance.id
    
    async def delete_instance(self, instance_id: int) -> bool:
        async with self.get_async_session() as session:
            stmt = select(Instance).where(Instance.id == instance_id)
            res = await session.execute(stmt)
            inst = res.scalar_one_or_none()
            if not inst:
                return False
            await session.delete(inst)
            await session.commit()
            return True

    async def get_instance_by_service_type(self, service_type: ServiceType, job_id: int | None = None):
        async with self.get_async_session() as session:
            stmt = select(Instance).filter_by(service_type=service_type)
            if job_id is not None:
                stmt = stmt.filter_by(job_id=job_id)
            result = await session.execute(stmt)
            instance = result.scalars().first()
            return instance

    async def get_instances_by_job(self, job_id: int):
        async with self.get_async_session() as session:
            stmt = select(Instance).filter_by(job_id=job_id)
            result = await session.execute(stmt)
            instances = result.scalars().all()
            return instances
    
    async def get_instance(self, instance_id: int):
        async with self.get_async_session() as session:
            stmt = select(Instance).filter_by(id=instance_id)
            result = await session.execute(stmt)
            instance = result.scalar_one_or_none()
            return instance

    async def get_all_instances(self):
        async with self.get_async_session() as session:
            stmt = select(Instance)
            result = await session.execute(stmt)
            instances = result.scalars().all()
            return instances

    async def update_instance(self, data: dict) -> bool:
        """
        data is the parsed JSON. If a key is present, we update that column 
        (including `None`). If absent, we leave the column untouched.
        """
        # For clarity, instance_id MUST appear in `data`; 
        # everything else is optional
        instance_id = data["instance_id"]

        updates = {}
        if "job_id" in data:
            # If user wants to set job_id=7 or job_id=None:
            updates["job_id"] = data["job_id"]

        if "pod_id" in data:
            updates["pod_id"] = data["pod_id"]

        if "process_id" in data:
            updates["process_id"] = data["process_id"]

        if "gpu_enabled" in data:
            updates["gpu_enabled"] = data["gpu_enabled"]

        if "slot_type" in data:
            # Example of validating a non-null value
            if data["slot_type"] is not None:
                updates["slot_type"] = SlotType[data["slot_type"]]
            else:
                updates["slot_type"] = None  # explicitly null

        if "connection_info" in data:
            updates["connection_info"] = data["connection_info"]

        if not updates:
            # no changes => just return success
            return True

        async with self.get_async_session() as session:
            stmt = (
                update(Instance)
                .where(Instance.id == instance_id)
                .values(**updates)
            )
            await session.execute(stmt)
            await session.commit()
            return True

    ######################################################################
    # NEW METHODS
    ######################################################################
    async def get_free_instances_by_slot_type(self, slot_type: SlotType) -> List[Instance]:
        """
        Returns all Instances with the given slot_type and job_id == None
        (meaning not currently reserved for any job).
        """
        async with self.get_async_session() as session:
            stmt = (
                select(Instance)
                .where(Instance.slot_type == slot_type)
                .where(Instance.job_id == None)
            )
            res = await session.execute(stmt)
            free_list = res.scalars().all()
            return free_list

    async def reserve_instance(self, instance_id: int, job_id: int) -> bool:
        """
        Sets job_id on a free Instance so that it’s "reserved" for that job.
        If the instance is already taken (job_id != None), returns False.
        """
        async with self.get_async_session() as session:
            # ensure instance is free
            stmt = select(Instance).where(Instance.id == instance_id)
            res = await session.execute(stmt)
            inst = res.scalar_one_or_none()
            if not inst:
                return False
            if inst.job_id is not None:
                # already reserved
                return False

            # set job_id
            stmt2 = (
                update(Instance)
                .where(Instance.id == instance_id, Instance.job_id == None)
                .values(job_id=job_id)
            )
            await session.execute(stmt2)
            await session.commit()
            return True
