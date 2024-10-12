from .models import AsyncSessionLocal, Job, Task, TaskStatus, Plugin, StateUpdate, Subnet
from sqlalchemy import select, update
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

    async def create_task(self, job_id: int, task_id: int, job_iteration: int, status: TaskStatus):
        async with AsyncSessionLocal() as session:
            new_task = Task(
                job_id=job_id,
                task_id=task_id,
                job_iteration=job_iteration,
                status=status
            )
            session.add(new_task)
            await session.commit()
            logging.debug(f"Created Task {task_id} for Job {job_id}, Iteration {job_iteration} with status {status}.")
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

    async def update_task_status(self, task_id: int, status: TaskStatus, result=None):
        async with AsyncSessionLocal() as session:
            stmt = update(Task).where(Task.task_id == task_id).values(status=status, result=result)
            await session.execute(stmt)
            await session.commit()
            logging.debug(f"Updated Task {task_id} to status {status} with result {result}.")
    
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
    
    async def get_task(self, task_id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Task).filter_by(task_id=task_id)
            result = await session.execute(stmt)
            task = result.scalar_one_or_none()
            if task:
                logging.debug(f"Retrieved Task: {task}")
            else:
                logging.error(f"Task with ID {task_id} not found.")
            return task.job
    
db_adapter = DBAdapter()