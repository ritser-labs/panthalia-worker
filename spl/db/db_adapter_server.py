# db_adapter_server.py

from ..models import (
    AsyncSessionLocal, Job, Task, TaskStatus, Plugin, StateUpdate, Subnet,
    Perm, Sot, PermDescription, PermType, Base, init_db
)
from sqlalchemy import select, update, desc, func
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
from typing import Dict
import logging
import json
import asyncio

logger = logging.getLogger(__name__)

class DBAdapterServer:
    def __init__(self):
        asyncio.run(init_db())

    async def get_job(self, job_id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Job).filter_by(id=job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if job:
                logger.debug(f"Retrieved Job: {job}")
            else:
                logger.error(f"Job with ID {job_id} not found.")
            return job

    async def update_job_iteration(self, job_id: int, new_iteration: int):
        async with AsyncSessionLocal() as session:
            stmt = update(Job).where(Job.id == job_id).values(iteration=new_iteration)
            await session.execute(stmt)
            await session.commit()
            logger.debug(f"Updated Job {job_id} to iteration {new_iteration}")

    async def mark_job_as_done(self, job_id: int):
        async with AsyncSessionLocal() as session:
            stmt = update(Job).where(Job.id == job_id).values(done=True)
            await session.execute(stmt)
            await session.commit()
            logger.info(f"Marked Job {job_id} as done.")

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
            logger.debug(f"Created Task {subnet_task_id} for Job {job_id}, Iteration {job_iteration} with status {status}.")
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
            logger.debug(f"Created Job {name} with Plugin {plugin_id}, Subnet {subnet_id}, SOT URL {sot_url}, Iteration {iteration}.")
            return new_job.id

    async def create_subnet(self, address: str, rpc_url: str):
        async with AsyncSessionLocal() as session:
            new_subnet = Subnet(
                address=address,
                rpc_url=rpc_url
            )
            session.add(new_subnet)
            await session.commit()
            logger.debug(f"Created Subnet {address} with RPC URL {rpc_url}.")
            return new_subnet.id

    async def create_plugin(self, name: str, code: str):
        async with AsyncSessionLocal() as session:
            new_plugin = Plugin(
                name=name,
                code=code
            )
            session.add(new_plugin)
            await session.commit()
            logger.debug(f"Created Plugin {name} with code {code}.")
            return new_plugin.id
    
    async def update_time_solved(
        self,
        subnet_task_id: int,
        job_id: int,
        time_solved: int
    ):
        time_solved = datetime.utcfromtimestamp(time_solved)
        async with AsyncSessionLocal() as session:
            stmt = update(Task).where(
                Task.subnet_task_id == subnet_task_id
                and Task.job_id == job_id
            ).values(time_solved=time_solved)
            await session.execute(stmt)
            await session.commit()
            logger.debug(f"Updated Task {subnet_task_id} to time solved {time_solved}.")
    
    async def update_time_solver_selected(
        self,
        subnet_task_id: int,
        job_id: int,
        time_solver_selected: int
    ):
        time_solver_selected = datetime.utcfromtimestamp(time_solver_selected)
        async with AsyncSessionLocal() as session:
            stmt = update(Task).where(
                Task.subnet_task_id == subnet_task_id
                and Task.job_id == job_id
            ).values(time_solver_selected=time_solver_selected)
            await session.execute(stmt)
            await session.commit()
            logger.debug(f"Updated Task {subnet_task_id} to time solver selected {time_solver_selected}.")

    async def update_task_status(
        self,
        subnet_task_id: int,
        job_id: int,
        status: TaskStatus,
        result=None,
        solver_address=None
    ):
        async with AsyncSessionLocal() as session:
            stmt = update(Task).where(
                Task.subnet_task_id == subnet_task_id
                and Task.job_id == job_id
            ).values(
                status=status, result=result, solver_address=solver_address)
            await session.execute(stmt)
            await session.commit()
            logger.debug(f"Updated Task {subnet_task_id} to status {status} with result {result}.")

    async def create_state_update(self, job_id: int, data: Dict):
        async with AsyncSessionLocal() as session:
            new_state_update = StateUpdate(
                job_id=job_id,
                data=data
            )
            session.add(new_state_update)
            await session.commit()
            logger.debug(f"Created State Update for Job {job_id}.")
            return new_state_update.id

    async def get_plugin(self, plugin_id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Plugin).filter_by(id=plugin_id)
            result = await session.execute(stmt)
            plugin = result.scalar_one_or_none()
            if plugin:
                logger.debug(f"Retrieved Plugin: {plugin}")
            else:
                logger.error(f"Plugin with ID {plugin_id} not found.")
            return plugin

    async def get_subnet_using_address(self, address: str):
        async with AsyncSessionLocal() as session:
            stmt = select(Subnet).filter_by(address=address)
            result = await session.execute(stmt)
            subnet = result.scalar_one_or_none()
            if subnet:
                logger.debug(f"Retrieved Subnet: {subnet}")
            else:
                logger.error(f"Subnet with address {address} not found.")
            return subnet

    async def get_task(self, subnet_task_id: int, subnet_id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Task).options(joinedload(Task.job)).join(Task.job).join(Job.subnet).filter(
                Task.subnet_task_id == subnet_task_id, 
                Job.subnet_id == subnet_id
            )
            result = await session.execute(stmt)
            task = result.scalar_one_or_none()
            if task:
                logger.debug(f"Retrieved Task: {task}")
            else:
                logger.error(f"Task with ID {subnet_task_id} not found or does not match Subnet ID {subnet_id}.")
            return task

    async def get_tasks_with_pagination_for_job(self, job_id: int, offset: int = 0, limit: int = 20):
        """
        Retrieve tasks for a specific job with pagination, ordered by the earliest created.
        :param job_id: The ID of the job to retrieve tasks for.
        :param offset: The starting point for pagination.
        :param limit: The number of tasks to retrieve.
        :return: A list of tasks.
        """
        async with AsyncSessionLocal() as session:
            # Select tasks for a specific job, ordered by creation time, applying offset and limit
            stmt = select(Task).filter_by(job_id=job_id).order_by(Task.submitted_at.asc()).offset(offset).limit(limit)
            result = await session.execute(stmt)
            tasks = result.scalars().all()
            logger.debug(f"Retrieved {len(tasks)} tasks for job {job_id} with offset {offset} and limit {limit}.")
            return tasks
        
    async def get_task_count_for_job(self, job_id: int):
        """
        Get the total number of tasks for a specific job.
        :param job_id: The ID of the job.
        :return: The total number of tasks for the job.
        """
        async with AsyncSessionLocal() as session:
            stmt = select(func.count(Task.id)).filter_by(job_id=job_id)
            result = await session.execute(stmt)
            task_count = result.scalar_one()
            logger.debug(f"Job {job_id} has {task_count} tasks.")
            return task_count

    async def get_task_count_by_status_for_job(self, job_id: int, statuses: list[TaskStatus]):
        """
        Get the number of tasks for a specific job with a list of statuses.
        :param job_id: The ID of the job.
        :param statuses: A list of TaskStatus values to count tasks for.
        :return: The number of tasks with the given statuses for the job.
        """
        async with AsyncSessionLocal() as session:
            stmt = select(func.count(Task.id)).filter(
                Task.job_id == job_id,
                Task.status.in_(statuses)
            )
            result = await session.execute(stmt)
            task_status_count = result.scalar_one()
            logger.debug(f"Job {job_id} has {task_status_count} tasks with statuses {statuses}.")
            return task_status_count

    async def get_perm(self, address: str, perm: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Perm).filter_by(address=address, perm=perm)
            result = await session.execute(stmt)
            perm_obj = result.scalar_one_or_none()
            return perm_obj

    async def set_last_nonce(self, address: str, perm: int, last_nonce: str):
        async with AsyncSessionLocal() as session:
            stmt = update(Perm).where(
                Perm.address == address,
                Perm.perm == perm
            ).values(last_nonce=last_nonce)
            await session.execute(stmt)
            await session.commit()
            logger.debug(f"Updated last nonce for address {address} to {last_nonce}")

    async def get_sot(self, id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Sot).filter_by(id=id)
            result = await session.execute(stmt)
            sot = result.scalar_one_or_none()
            if sot:
                logger.debug(f"Retrieved SOT: {sot}")
            else:
                logger.error(f"SOT with ID {id} not found.")
            return sot

    async def create_perm(self, address: str, perm: int):
        async with AsyncSessionLocal() as session:
            new_perm = Perm(
                address=address,
                perm=perm
            )
            session.add(new_perm)
            await session.commit()
            logger.debug(f"Created Perm for address {address} with perm {perm}.")
            return new_perm.id

    async def create_perm_description(self, perm_type: PermType):
        async with AsyncSessionLocal() as session:
            new_perm_description = PermDescription(
                perm_type=perm_type
            )
            session.add(new_perm_description)
            await session.commit()
            perm_id = new_perm_description.id
            logger.debug(f"Created Perm Description with type {perm_type} and id {perm_id}.")
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
            logger.debug(f"Created SOT for Job {job_id} with perm {perm}.")
            return new_sot.id

    async def get_sot_by_job_id(self, job_id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Sot).filter_by(job_id=job_id)
            result = await session.execute(stmt)
            sot = result.scalar_one_or_none()
            if sot:
                logger.debug(f"Retrieved SOT: {sot}")
            else:
                logger.error(f"SOT for Job {job_id} not found.")
            return sot
    
    async def get_total_state_updates_for_job(self, job_id: int):
        """
        Get the total number of state updates for a specific job.
        :param job_id: The ID of the job.
        :return: The total number of state updates for the job.
        """
        async with AsyncSessionLocal() as session:
            stmt = select(func.count(StateUpdate.id)).filter_by(job_id=job_id)
            result = await session.execute(stmt)
            total_state_updates = result.scalar_one()
            logger.debug(f"Job {job_id} has {total_state_updates} state updates.")
            return total_state_updates
    
    async def get_last_task_with_status(self, job_id: int, statuses: list[TaskStatus]):
        """
        Get the last task of a job that has one of the specified TaskStatus values.
        :param job_id: The ID of the job.
        :param statuses: A list of TaskStatus values to filter tasks by.
        :return: The last task with one of the specified statuses or None if no task is found.
        """
        async with AsyncSessionLocal() as session:
            stmt = select(Task).filter(
                Task.job_id == job_id,
                Task.status.in_(statuses)
            ).order_by(desc(Task.submitted_at)).limit(1)
            result = await session.execute(stmt)
            task = result.scalar_one_or_none()
            if task:
                logger.debug(f"Retrieved the last task with one of the statuses {statuses} for Job {job_id}.")
            else:
                logger.error(f"No task found with statuses {statuses} for Job {job_id}.")
            return task


# Instantiate the server adapter
db_adapter_server = DBAdapterServer()
