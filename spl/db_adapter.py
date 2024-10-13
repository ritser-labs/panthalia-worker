from .models import (
    AsyncSessionLocal, Job, Task, TaskStatus, Plugin, StateUpdate, Subnet,
    Perms, Sot, PermDescription, PermType)
from sqlalchemy import select, update
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession
import logging
import json


class DBAdapter:
    def __init__(self):
        pass

    async def get_job(self, job_id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Job).filter_by(id=job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if job:
                logging.debug(f"Retrieved Job: {job}")
            else:
                logging.error(f"Job with ID {job_id} not found.")
            return job

    async def update_job_iteration(self, job_id: int, new_iteration: int):
        async with AsyncSessionLocal() as session:
            stmt = update(Job).where(Job.id == job_id).values(iteration=new_iteration)
            await session.execute(stmt)
            await session.commit()
            logging.debug(f"Updated Job {job_id} to iteration {new_iteration}")

    async def mark_job_as_done(self, job_id: int):
        async with AsyncSessionLocal() as session:
            stmt = update(Job).where(Job.id == job_id).values(done=True)
            await session.execute(stmt)
            await session.commit()
            logging.info(f"Marked Job {job_id} as done.")

    async def create_task(self, job_id: int, subnet_task_id: int, job_iteration: int, status: TaskStatus):
        async with AsyncSessionLocal() as session:
            new_task = Task(
                job_id=job_id,
                subnet_task_id=subnet_task_id,
                job_iteration=job_iteration,
                status=status
            )
            session.add(new_task)
            await session.commit()
            logging.debug(f"Created Task {subnet_task_id} for Job {job_id}, Iteration {job_iteration} with status {status}.")
            return new_task.id
    
    async def create_job(self, name: str, plugin_id: int, subnet_id: int, sot_url: str, iteration: int):
        async with AsyncSessionLocal() as session:
            new_job = Job(
                name=name,
                plugin_id=plugin_id,
                subnet_id=subnet_id,
                sot_url=sot_url,
                iteration=iteration
            )
            session.add(new_job)
            await session.commit()
            logging.debug(f"Created Job {name} with Plugin {plugin_id}, Subnet {subnet_id}, SOT URL {sot_url}, Iteration {iteration}.")
            return new_job.id

    async def create_subnet(self, address: str, rpc_url: str):
        async with AsyncSessionLocal() as session:
            new_subnet = Subnet(
                address=address,
                rpc_url=rpc_url
            )
            session.add(new_subnet)
            await session.commit()
            logging.debug(f"Created Subnet {address} with RPC URL {rpc_url}.")
            return new_subnet.id
    
    async def create_plugin(self, name: str, code: str):
        async with AsyncSessionLocal() as session:
            new_plugin = Plugin(
                name=name,
                code=code
            )
            session.add(new_plugin)
            await session.commit()
            logging.debug(f"Created Plugin {name} with code {code}.")
            return new_plugin.id

    async def update_task_status(self, subnet_task_id: int, status: TaskStatus, result=None):
        async with AsyncSessionLocal() as session:
            stmt = update(Task).where(Task.subnet_task_id == subnet_task_id).values(status=status, result=result)
            await session.execute(stmt)
            await session.commit()
            logging.debug(f"Updated Task {subnet_task_id} to status {status} with result {result}.")
    
    async def create_state_update(self, job_id: int, state_iteration: int):
        async with AsyncSessionLocal() as session:
            new_state_update = StateUpdate(
                job_id=job_id,
                state_iteration=state_iteration
            )
            session.add(new_state_update)
            await session.commit()
            logging.debug(f"Created State Update for Job {job_id}, Iteration {state_iteration}.")
    
    async def get_plugin_code(self, plugin_id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Plugin).filter_by(id=plugin_id)
            result = await session.execute(stmt)
            plugin = result.scalar_one_or_none()
            if plugin:
                logging.debug(f"Retrieved Plugin: {plugin}")
            else:
                logging.error(f"Plugin with ID {plugin_id} not found.")
            return plugin.code
    
    async def get_subnet_using_address(self, address: str):
        async with AsyncSessionLocal() as session:
            stmt = select(Subnet).filter_by(address=address)
            result = await session.execute(stmt)
            subnet = result.scalar_one_or_none()
            if subnet:
                logging.debug(f"Retrieved Subnet: {subnet}")
            else:
                logging.error(f"Subnet with address {address} not found.")
            return subnet
    
    async def get_task(self, subnet_task_id: int, subnet_id: int):
        async with AsyncSessionLocal() as session:
            # Eager load Job with joinedload
            stmt = select(Task).options(joinedload(Task.job)).join(Task.job).join(Job.subnet).filter(
                Task.subnet_task_id == subnet_task_id, 
                Job.subnet_id == subnet_id
            )
            result = await session.execute(stmt)
            task = result.scalar_one_or_none()
            if task:
                logging.debug(f"Retrieved Task: {task}")
            else:
                logging.error(f"Task with ID {subnet_task_id} not found or does not match Subnet ID {subnet_id}.")
            return task
    
    async def has_perm(self, address: str, perm: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Perms).filter_by(address=address, perm=perm)

            result = await session.execute(stmt)
            perm = result.scalar_one_or_none()
            return perm
    async def set_last_nonce(self, address: str, perm: int, last_nonce: str):
        async with AsyncSessionLocal() as session:
            stmt = update(Perms).where(
                Perms.address == address,
                Perms.perm == perm
            ).values(last_nonce=last_nonce)
            await session.execute(stmt)
            await session.commit()
            logging.debug(f"Updated last nonce for address {address} to {last_nonce}")
    
    async def get_sot(self, id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Sot).filter_by(id=id)
            result = await session.execute(stmt)
            sot = result.scalar_one_or_none()
            if sot:
                logging.debug(f"Retrieved SOT: {sot}")
            else:
                logging.error(f"SOT with ID {id} not found.")
            return sot
    
    async def create_perm(self, address: str, perm: int):
        async with AsyncSessionLocal() as session:
            new_perm = Perms(
                address=address,
                perm=perm
            )
            session.add(new_perm)
            await session.commit()
            logging.debug(f"Created Perm for address {address} with perm {perm}.")
            return new_perm.id
    
    async def create_perm_description(self, perm_type: PermType):
        async with AsyncSessionLocal() as session:
            new_perm_description = PermDescription(
                perm_type=perm_type
            )
            session.add(new_perm_description)
            await session.commit()
            perm_id = new_perm_description.id
            logging.debug(f"Created Perm Description with type {perm_type} and id {perm_id}.")
            return perm_id
    
    async def create_sot(self, job_id: int, url: str):
        async with AsyncSessionLocal() as session:
            perm = await self.create_perm_description(perm_type=PermType.ModifySot)
            new_sot = Sot(
                job_id=job_id,
                perm=perm,
                url=url
            )
            session.add(new_sot)
            await session.commit()
            logging.debug(f"Created SOT for Job {job_id} with perm {perm}.")
            return new_sot.id
    
    async def get_sot(self, job_id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Sot).filter_by(job_id=job_id)
            result = await session.execute(stmt)
            sot = result.scalar_one_or_none()
            if sot:
                logging.debug(f"Retrieved SOT: {sot}")
            else:
                logging.error(f"SOT for Job {job_id} not found.")
            return sot

db_adapter = DBAdapter()