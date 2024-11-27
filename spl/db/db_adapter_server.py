# db_adapter_server.py

from ..models import (
    AsyncSessionLocal, Job, Task, TaskStatus, Plugin, StateUpdate, Subnet,
    Perm, Sot, PermDescription, PermType, Base, init_db, ServiceType,
    Instance, Order, Account, OrderType, AccountTransaction, AccountTxnType
)
from ..auth.view import get_user_id
from sqlalchemy import select, update, desc, func
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone
from typing import Dict, Optional
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


    async def create_task(self, job_id: int, job_iteration: int, status: TaskStatus, params: str):
        async with AsyncSessionLocal() as session:
            stmt = select(Job).filter_by(id=job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if not job or job.user_id != get_user_id():
                raise PermissionError("You do not have access to create a task for this job.")

            new_task = Task(
                job_id=job_id,
                job_iteration=job_iteration,
                status=status,
                params=params
            )
            session.add(new_task)
            await session.commit()
            logger.debug(f"Created Task for Job {job_id}, Iteration {job_iteration} with status {status}.")
            return new_task.id
    
    # Add this helper function in DBAdapterServer
    async def handle_account_transaction(self, session, account_id: int, amount: float, transaction_type: AccountTxnType):
        # Retrieve the account object
        stmt = select(Account).where(Account.id == account_id)
        result = await session.execute(stmt)
        account = result.scalar_one_or_none()

        if not account:
            raise ValueError(f"Account with ID {account_id} not found.")

        # Check if there is enough available balance for deduction
        if transaction_type == AccountTxnType.Withdrawal and account.available < amount:
            raise ValueError(f"Insufficient balance in account {account_id}.")

        # Deduct or add balance
        if transaction_type == AccountTxnType.Withdrawal:
            account.available -= amount
        elif transaction_type == AccountTxnType.Deposit:
            account.available += amount

        # Create a AccountTransaction
        new_transaction = AccountTransaction(
            account_id=account_id,
            user_id=account.user_id,
            amount=amount,
            transaction_type=transaction_type,
            timestamp=datetime.utcnow()
        )
        session.add(new_transaction)
        session.add(account)  # Update the account object

        return new_transaction.id

    # Modify the create_order method
    async def create_order(self, task_id: int, subnet_id: int, order_type: OrderType, price: float):
        async with AsyncSessionLocal() as session:
            user_id = await get_user_id()
            account_stmt = select(Account).where(Account.user_id == user_id)
            account_result = await session.execute(account_stmt)
            account = account_result.scalar_one_or_none()
            if not account or account.user_id != get_user_id():
                raise PermissionError("You do not have access to create an order with this account.")

            if order_type == OrderType.Bid:
                if task_id is None:
                    raise ValueError(f"Task ID must be provided for Bid orders.")
                task_stmt = select(Task).where(Task.id == task_id)
                task_result = await session.execute(task_stmt)
                task = task_result.scalar_one_or_none()
                if not task or task.job.user_id != user_id:
                    raise PermissionError("You do not have access to create an order for this task.")
                
                amount_to_deduct = price
                if task.job.subnet_id != subnet_id:
                    raise ValueError(f"Task {task_id} does not belong to Subnet {subnet_id}.")
            elif order_type == OrderType.Ask:
                if task_id is not None:
                    raise ValueError(f"Task ID must be None for Ask orders.")
                subnet_stmt = select(Subnet).where(Subnet.id == task.job.subnet_id)
                subnet_result = await session.execute(subnet_stmt)
                subnet = subnet_result.scalar_one_or_none()
                if not subnet:
                    raise ValueError(f"Subnet with ID {task.job.subnet_id} not found.")
                amount_to_deduct = subnet.account_multiplier * price

            await self.handle_account_transaction(session, account.id, amount_to_deduct, AccountTxnType.Withdrawal)

            new_order = Order(
                subnet_id=subnet_id,
                task_id=task_id,
                order_type=order_type,
                account_id=account.id,
                price=price,
                user_id=user_id
            )
            session.add(new_order)
            await session.commit()
            logger.debug(f"Created Order for Task {task_id} with type {order_type} and price {price}.")
            subnet_id = task.job.subnet_id
            await self.match_bid_ask_orders(subnet_id)
            return new_order.id
    
    async def get_num_orders(self, subnet_id: int, order_type: OrderType):
        async with AsyncSessionLocal() as session:
            user_id = await get_user_id()
            stmt = select(Order).filter_by(subnet_id=subnet_id, user_id=user_id, order_type=order_type)
            result = await session.execute(stmt)
            orders = result.scalars().all()
            logger.debug(f"Retrieved {len(orders)} orders for Subnet {subnet_id}.")
            return len(orders)
    
    
    async def create_bids_and_tasks(
        self, 
        job_id: int, 
        num_tasks: int, 
        price: float,
        params: str
    ):
        """
        creates a specified number of tasks and corresponding bid orders for a job.
        increments the job iteration with each task creation.
        
        :param job_id: the id of the job to associate with the tasks.
        :param num_tasks: the number of tasks and bids to create.
        :param price: the price for the bid orders.
        :return: a list of dictionaries containing task ids and corresponding bid ids.
        """
        async with AsyncSessionLocal() as session:
            # retrieve the job
            stmt = select(Job).filter_by(id=job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            account_stmt = select(Account).filter_by(user_id=job.user_id)
            account_result = await session.execute(account_stmt)
            account = account_result.scalar_one_or_none()
            
            if not job:
                raise ValueError(f"job with id {job_id} not found.")
            
            if job.user_id != get_user_id():
                raise PermissionError("you do not have access to modify this job.")
            
            created_items = []
            for i in range(num_tasks):
                # increment job iteration
                job.iteration += 1
                
                # create a new task
                new_task = Task(
                    job_id=job_id,
                    job_iteration=job.iteration,
                    status=TaskStatus.Pending,  # defaulting status to Pending
                    params=params
                )
                session.add(new_task)
                await session.flush()  # flush to get task id
                
                # create a corresponding bid order
                new_bid = Order(
                    task_id=new_task.id,
                    order_type=OrderType.Bid,
                    account_id=account.id,
                    price=price,
                    subnet_id=job.subnet_id,
                    user_id=get_user_id(),
                )
                session.add(new_bid)
                await session.flush()  # flush to get order id
                
                created_items.append({
                    "task_id": new_task.id,
                    "bid_id": new_bid.id
                })
            
            await session.commit()
            logger.debug(f"created {num_tasks} tasks and corresponding bids for job {job_id}.")
            return created_items


    async def delete_order(self, order_id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Order).filter_by(id=order_id)
            result = await session.execute(stmt)
            order = result.scalar_one_or_none()
            if not order or order.user_id != get_user_id():
                raise PermissionError("You do not have access to delete this order.")

            if order.task and order.task.ask is not None:
                raise ValueError(f"Cannot delete order {order_id} because the associated task has an ask.")

            if order.order_type == OrderType.Bid:
                amount_to_reverse = order.price
            elif order.order_type == OrderType.Ask:
                subnet_stmt = select(Subnet).filter_by(id=order.subnet_id)
                subnet_result = await session.execute(subnet_stmt)
                subnet = subnet_result.scalar_one_or_none()
                if not subnet:
                    raise ValueError(f"Subnet with ID {order.subnet_id} not found.")
                amount_to_reverse = subnet.account_multiplier * order.price
            else:
                raise ValueError(f"Invalid order type for order {order_id}.")

            await self.handle_account_transaction(session, order.account_id, amount_to_reverse, AccountTxnType.Deposit)
            await session.delete(order)
            await session.commit()
            logger.debug(f"Deleted Order with ID {order_id} and reversed the account transaction.")

    async def match_bid_ask_orders(self, subnet_id: int):
        """
        finds matching bid and ask orders in the specified subnet and assigns the ask order to the bid's task.
        :param subnet_id: the id of the subnet to search for matching orders.
        :return: list of matched orders or none if no match is found.
        """
        async with AsyncSessionLocal() as session:
            # select bid orders within the given subnet that don't have a corresponding ask
            bid_orders_stmt = (
                select(Order)
                .filter_by(subnet_id=subnet_id, order_type=OrderType.Bid)
                .join(Task, Task.id == Order.task_id)
                .filter(Task.ask == None)  # exclude tasks with existing asks
            )
            ask_orders_stmt = select(Order).filter_by(subnet_id=subnet_id, order_type=OrderType.Ask)

            bid_orders_result = await session.execute(bid_orders_stmt)
            ask_orders_result = await session.execute(ask_orders_stmt)

            bid_orders = bid_orders_result.scalars().all()
            ask_orders = ask_orders_result.scalars().all()

            matches = []

            # attempt to match each bid with an ask
            for bid in bid_orders:
                for ask in ask_orders:
                    if bid.price >= ask.price:  # matching condition: bid price must be at least as high as ask price
                        # link the ask order to the bid's task
                        bid.task.ask = ask
                        bid.task.time_solver_selected = datetime.now(timezone.utc)
                        session.add(bid.task)
                        matches.append((bid, ask))

                        # remove the matched ask order from the list to prevent reuse
                        ask_orders.remove(ask)
                        break

            await session.commit()

            if matches:
                logger.debug(f"matched {len(matches)} bid-ask pairs in subnet {subnet_id}.")
            else:
                logger.info(f"no matching bid-ask orders found in subnet {subnet_id}.")

            return matches


    async def deposit_account(self, amount: float):
        async with AsyncSessionLocal() as session:
            user_id = await get_user_id()
            stmt = select(Account).where(Account.user_id == user_id)
            result = await session.execute(stmt)
            account = result.scalar_one_or_none()
            if not account:
                raise PermissionError("You do not have access to deposit into this account.")

            await self.handle_account_transaction(session, account.id, amount, AccountTxnType.Deposit)
            await session.commit()
            logger.debug(f"Deposited {amount} into Account {account.id}.")


    async def withdraw_account(self, amount: float):
        async with AsyncSessionLocal() as session:
            user_id = await get_user_id()
            stmt = select(Account).where(Account.user_id == user_id)
            result = await session.execute(stmt)
            account = result.scalar_one_or_none()
            if not account:
                raise PermissionError("You do not have access to withdraw from this account.")

            if account.available < amount:
                raise ValueError(f"Insufficient balance in account {account.id} to withdraw {amount}.")

            await self.handle_account_transaction(session, account.id, amount, AccountTxnType.Withdrawal)
            await session.commit()
            logger.debug(f"Withdrew {amount} from Account {account.id}.")


    async def create_job(self, name: str, plugin_id: int, subnet_id: int, sot_url: str, iteration: int):
        async with AsyncSessionLocal() as session:
            new_job = Job(
                name=name,
                plugin_id=plugin_id,
                subnet_id=subnet_id,
                user_id=get_user_id(),
                sot_url=sot_url,
                iteration=iteration,
                done=False,  # assuming default is False
                last_updated=datetime.utcnow(),
                submitted_at=datetime.utcnow()
            )
            session.add(new_job)
            await session.commit()
            logger.debug(f"Created Job {name} with ID {new_job.id}.")
            return new_job.id

    async def create_subnet(
        self,
        dispute_period: int,
        solve_period: int,
        stake_multiplier: float,
    ):
        async with AsyncSessionLocal() as session:
            new_subnet = Subnet(
                dispute_period=dispute_period,
                solve_period=solve_period,
                stake_multiplier=stake_multiplier
            )
            session.add(new_subnet)
            await session.commit()
            logger.debug(f"Created Subnet {new_subnet.id}.")
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
    
    async def update_task_status(
        self,
        task_id: int,
        job_id: int,
        status: TaskStatus,
        result=None,
        solver_address=None
    ):
        async with AsyncSessionLocal() as session:
            stmt = update(Task).where(
                Task.id == task_id
                and Task.job_id == job_id
            ).values(
                status=status, result=result, solver_address=solver_address)
            await session.execute(stmt)
            await session.commit()
            logger.debug(f"Updated Task {task_id} to status {status} with result {result}.")
    
    async def should_dispute(self, task):
        return False
    
    async def resolve_task(
        self,
        task: Task,
        result: str,
        status: TaskStatus
    ):
        async with AsyncSessionLocal() as session:
            # update the task with the result and status
            task.result = result
            task.status = status
            task.time_solved = datetime.now(timezone.utc)

            # reward the solver and re-add the stake
            bid_order = task.bid
            if not bid_order:
                raise ValueError(f"No associated bid order found for task {task.id}.")

            # retrieve solver account
            solver_account_stmt = select(Account).where(Account.id == bid_order.account_id)
            solver_account_result = await session.execute(solver_account_stmt)
            solver_account = solver_account_result.scalar_one_or_none()

            if not solver_account:
                raise ValueError(f"Solver account not found for task {task.id}.")

            # ensure subnet exists
            subnet_stmt = select(Subnet).where(Subnet.id == task.job.subnet_id)
            subnet_result = await session.execute(subnet_stmt)
            subnet = subnet_result.scalar_one_or_none()

            if not subnet:
                raise ValueError(f"Subnet with ID {task.job.subnet_id} not found.")

            # calculate stake amount
            stake_amount = subnet.stake_multiplier * bid_order.price

            # handle reward transaction
            await self.handle_account_transaction(
                session=session,
                account_id=solver_account.id,
                amount=bid_order.price,
                transaction_type=AccountTxnType.Deposit
            )

            # handle stake re-addition transaction
            await self.handle_account_transaction(
                session=session,
                account_id=solver_account.id,
                amount=stake_amount,
                transaction_type=AccountTxnType.Deposit
            )

            await session.commit()
            logger.debug(f"Resolved Task {task.id}, rewarded solver, and re-added stake.")

    async def submit_task_result(
        self,
        task_id: int,
        result: str
    ):
        async with AsyncSessionLocal() as session:
            # retrieve the task
            task_stmt = select(Task).where(Task.id == task_id)
            task_result = await session.execute(task_stmt)
            task = task_result.scalar_one_or_none()

            if not task:
                raise ValueError(f"Task with ID {task_id} not found.")

            # ensure the user has permission to submit the result
            if task.ask.user_id != await get_user_id():
                raise PermissionError("You do not have access to submit a result for this task.")

            # ensure the task is in the correct status
            if task.status != TaskStatus.SolverSelected:
                raise ValueError(f"Task {task_id} is not in SolverSelected status.")

            # check if a dispute should be raised
            if await self.should_dispute(task):
                raise NotImplementedError("Dispute logic not implemented.")

            # resolve the task
            await self.resolve_task(task, result, TaskStatus.ResolvedCorrect)

        

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
    
    async def get_subnet(self, subnet_id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Subnet).filter_by(id=subnet_id)
            result = await session.execute(stmt)
            subnet = result.scalar_one_or_none()
            if subnet:
                logger.debug(f"Retrieved Subnet: {subnet}")
            else:
                logger.error(f"Subnet with ID {subnet_id} not found.")
            return subnet

    async def get_task(self, task_id: int, subnet_id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Task).options(joinedload(Task.job)).join(Task.job).join(Job.subnet).filter(
                Task.id == task_id, 
                Job.subnet_id == subnet_id
            )
            result = await session.execute(stmt)
            task = result.scalar_one_or_none()
            if task:
                logger.debug(f"Retrieved Task: {task}")
            else:
                logger.error(f"Task with ID {task_id} not found or does not match Subnet ID {subnet_id}.")
            return task
    
    async def get_assigned_tasks(self):
        async with AsyncSessionLocal() as session:
            user_id = await get_user_id()
            
            # Query tasks where the ask order's user_id matches the current user_id
            # and the task status is TaskStatus.SolverSelected
            stmt = (
                select(Task)
                .join(Order, Task.id == Order.task_id)
                .filter(
                    Order.order_type == OrderType.Ask,
                    Order.user_id == user_id,
                    Task.status == TaskStatus.SolverSelected
                )
            )

            result = await session.execute(stmt)
            tasks = result.scalars().all()

            logger.debug(f"Retrieved {len(tasks)} tasks assigned to user {user_id}.")
            return {'assigned_tasks': tasks}

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
    
    async def update_sot(self, sot_id: int, url: str):
        async with AsyncSessionLocal() as session:
            stmt = update(Sot).where(Sot.id == sot_id).values(url=url)
            await session.execute(stmt)
            await session.commit()
            logger.debug(f"Updated SOT {sot_id} with URL {url}.")

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
    
    async def create_instance(
        self,
        name: str,
        service_type: ServiceType,
        job_id: Optional[int],
        private_key: str,
        pod_id: str,
        process_id: int
    ):
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
            logger.debug(f"Created Instance {name} of type {service_type} with pod_id {pod_id} for Job {job_id}.")
            return new_instance.id

    async def get_instance_by_service_type(self, service_type: ServiceType, job_id: Optional[int] = None):
        async with AsyncSessionLocal() as session:
            stmt = select(Instance).filter_by(service_type=service_type)
            if job_id is not None:
                stmt = stmt.filter_by(job_id=job_id)
            result = await session.execute(stmt)
            instance = result.scalars().first()
            if instance:
                logger.debug(f"Retrieved Instance: {instance}")
            else:
                logger.error(f"Instance with service_type {service_type} and job_id {job_id} not found.")
            return instance

    async def get_instances_by_job(self, job_id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Instance).filter_by(job_id=job_id)
            result = await session.execute(stmt)
            instances = result.scalars().all()
            logger.debug(f"Retrieved {len(instances)} instances for Job {job_id}.")
            return instances
    
    async def get_all_instances(self):
        async with AsyncSessionLocal() as session:
            stmt = select(Instance)
            result = await session.execute(stmt)
            instances = result.scalars().all()
            logger.debug(f"Retrieved {len(instances)} instances.")
            return instances

    async def update_instance(self, instance_id: int, **kwargs):
        async with AsyncSessionLocal() as session:
            stmt = update(Instance).where(Instance.id == instance_id).values(**kwargs)
            await session.execute(stmt)
            await session.commit()
            logger.debug(f"Updated Instance {instance_id} with {kwargs}.")

    async def get_jobs_without_instances(self):
        async with AsyncSessionLocal() as session:
            stmt = (
                select(Job)
                .outerjoin(Instance, Job.id == Instance.job_id)
                .filter(Instance.id == None)  # filter for jobs with no associated instances
            )
            result = await session.execute(stmt)
            jobs_without_instances = result.scalars().all()
            logger.debug(f"Retrieved {len(jobs_without_instances)} jobs without instances.")
            return jobs_without_instances

    async def get_plugins(self):
        async with AsyncSessionLocal() as session:
            stmt = select(Plugin)
            result = await session.execute(stmt)
            plugins = result.scalars().all()
            logger.debug(f"Retrieved {len(plugins)} plugins.")
            return plugins

# Instantiate the server adapter
db_adapter_server = DBAdapterServer()

