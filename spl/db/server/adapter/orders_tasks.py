# spl/db/server/adapter/orders_tasks.py

import logging
from datetime import datetime, timezone, timedelta
from sqlalchemy import select, update, desc, func, or_
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert
import json

from ....models import (
    AsyncSessionLocal, Job, Task, TaskStatus, Subnet,
    Order, Account, OrderType, HoldType, CreditTxnType, EarningsTxnType, PlatformRevenueTxnType, Hold
)

logger = logging.getLogger(__name__)
PLATFORM_FEE_PERCENTAGE = 0.1
DISPUTE_PAYOUT_DELAY_DAYS = 1

class DBAdapterOrdersTasksMixin:
    async def get_job(self, job_id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Job).filter_by(id=job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            return job

    async def update_job_iteration(self, job_id: int, new_iteration: int):
        async with AsyncSessionLocal() as session:
            stmt = update(Job).where(Job.id == job_id).values(iteration=new_iteration)
            await session.execute(stmt)
            await session.commit()

    async def update_job_sot_url(self, job_id: int, new_sot_url: str):
        async with AsyncSessionLocal() as session:
            stmt = update(Job).where(Job.id == job_id).values(sot_url=new_sot_url)
            await session.execute(stmt)
            await session.commit()

    async def mark_job_as_done(self, job_id: int):
        async with AsyncSessionLocal() as session:
            stmt = update(Job).where(Job.id == job_id).values(done=True)
            await session.execute(stmt)
            await session.commit()

    async def create_task(self, job_id: int, job_iteration: int, status: TaskStatus, params: str):
        async with AsyncSessionLocal() as session:
            stmt = select(Job).filter_by(id=job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if not job or job.user_id != self.get_user_id():
                raise PermissionError("no access to create a task for this job.")

            new_task = Task(
                job_id=job_id,
                job_iteration=job_iteration,
                status=status,
                params=params
            )
            session.add(new_task)
            await session.commit()
            return new_task.id

    async def update_task_status(self, task_id: int, job_id: int, status: TaskStatus, result=None, solver_address=None):
        async with AsyncSessionLocal() as session:
            stmt = update(Task).where(
                Task.id == task_id,
                Task.job_id == job_id
            ).values(
                status=status, result=result
            )
            await session.execute(stmt)
            await session.commit()

    async def create_order(
        self,
        task_id: int | None,
        subnet_id: int,
        order_type: OrderType,
        price: float,
        hold_id: int | None = None
    ):
        """
        Creates either a BID or an ASK order, then calls match_bid_ask_orders().
        Wrapped in one transaction, with a rollback if anything fails.
        """
        async with AsyncSessionLocal() as session:
            try:
                user_id = self.get_user_id()
                account = await self.get_or_create_account(user_id, session)

                stmt_subnet = select(Subnet).where(Subnet.id == subnet_id)
                subnet_result = await session.execute(stmt_subnet)
                subnet = subnet_result.scalar_one_or_none()
                if not subnet:
                    raise ValueError("subnet not found")

                if order_type == OrderType.Bid:
                    if task_id is None:
                        raise ValueError("task_id required for bid")
                    task_stmt = select(Task).where(Task.id == task_id).options(joinedload(Task.job))
                    task_result = await session.execute(task_stmt)
                    task = task_result.scalar_one_or_none()
                    if not task or task.job.user_id != user_id:
                        raise PermissionError("only the job owner can submit bids")

                    hold = await self.select_hold_for_order(session, account, subnet, order_type, price, hold_id)
                    new_order = Order(
                        order_type=order_type,
                        price=price,
                        subnet_id=subnet_id,
                        user_id=user_id,
                        account_id=account.id,
                        bid_task_id=task_id
                    )
                    session.add(new_order)
                    await session.flush()
                    await self.reserve_funds_on_hold(session, hold, price, new_order)
                    new_order.hold_id = hold.id

                elif order_type == OrderType.Ask:
                    if task_id is not None:
                        raise ValueError("task_id must be None for ask")
                    hold = await self.select_hold_for_order(session, account, subnet, order_type, price, hold_id)
                    new_order = Order(
                        order_type=order_type,
                        price=price,
                        subnet_id=subnet_id,
                        user_id=user_id,
                        account_id=account.id
                    )
                    session.add(new_order)
                    await session.flush()
                    stake_amount = subnet.stake_multiplier * price
                    await self.reserve_funds_on_hold(session, hold, stake_amount, new_order)
                    new_order.hold_id = hold.id
                else:
                    raise ValueError("invalid order type")

                # Important: match in the same transaction with row-level locking
                await self.match_bid_ask_orders(session, subnet_id)

                await session.commit()
                return new_order.id

            except Exception:
                await session.rollback()
                raise

    async def get_num_orders(self, subnet_id: int, order_type: str, matched: bool | None):
        async with AsyncSessionLocal() as session:
            query = select(func.count()).select_from(Order).where(
                Order.subnet_id == subnet_id,
                Order.order_type == order_type
            )

            # Apply matched filtering if provided
            if matched is not None:
                if matched:
                    query = query.where(
                        or_(Order.bid_task_id.isnot(None), Order.ask_task_id.isnot(None))
                    )
                else:
                    query = query.where(
                        Order.bid_task_id.is_(None),
                        Order.ask_task_id.is_(None)
                    )

            result = await session.execute(query)
            return {'num_orders': result.scalar_one()}

    async def create_bids_and_tasks(
        self,
        job_id: int,
        num_tasks: int,
        price: float,
        params: str,
        hold_id: int | None
    ):
        """
        Create tasks + BIDs for each, then do a match. All in one transaction.
        """
        async with AsyncSessionLocal() as session:
            try:
                stmt = select(Job).filter_by(id=job_id).options(joinedload(Job.subnet))
                result = await session.execute(stmt)
                job = result.scalar_one_or_none()
                if not job:
                    raise ValueError(f"Job {job_id} not found.")

                user_id = job.user_id
                account = await self.get_or_create_account(user_id, session)

                created_items = []
                for _ in range(num_tasks):
                    job.iteration += 1
                    new_task = Task(
                        job_id=job_id,
                        job_iteration=job.iteration,
                        status=TaskStatus.SelectingSolver,
                        params=params
                    )
                    session.add(new_task)
                    await session.flush()

                    hold = await self.select_hold_for_order(
                        session,
                        account,
                        job.subnet,
                        OrderType.Bid,
                        price,
                        hold_id
                    )
                    new_bid = Order(
                        bid_task=new_task,
                        order_type=OrderType.Bid,
                        account_id=account.id,
                        price=price,
                        subnet_id=job.subnet_id,
                        user_id=user_id,
                    )
                    session.add(new_bid)
                    await session.flush()
                    await self.reserve_funds_on_hold(session, hold, price, new_bid)
                    new_bid.hold_id = hold.id

                    created_items.append({
                        "task_id": new_task.id,
                        "bid_id": new_bid.id
                    })

                # Attempt to match in the same transaction
                await self.match_bid_ask_orders(session, job.subnet_id)
                await session.commit()

                return {'created_items': created_items}
            except Exception:
                await session.rollback()
                raise

    async def delete_order(self, order_id: int):
        """
        Delete an order if not matched, then re-run match to fill others.
        """
        async with AsyncSessionLocal() as session:
            try:
                stmt = select(Order).filter_by(id=order_id)
                result = await session.execute(stmt)
                order = result.scalar_one_or_none()
                if not order or order.user_id != self.get_user_id():
                    raise PermissionError("no access to delete this order.")

                subnet_stmt = select(Subnet).filter_by(id=order.subnet_id)
                subnet_result = await session.execute(subnet_stmt)
                subnet = subnet_result.scalar_one_or_none()
                if not subnet:
                    raise ValueError("Subnet not found")

                if order.order_type == OrderType.Bid and order.bid_task and order.bid_task.ask is not None:
                    raise ValueError("Cannot delete bid order since it's matched.")
                if order.order_type == OrderType.Ask and order.ask_task and order.ask_task.bid is not None:
                    raise ValueError("Cannot delete ask order since it's matched.")

                if order.order_type == OrderType.Bid:
                    amount_to_reverse = order.price
                else:
                    amount_to_reverse = subnet.stake_multiplier * order.price

                if order.hold:
                    await self.free_funds_from_hold(session, order.hold, amount_to_reverse, order)
                else:
                    raise ValueError("no hold found for this order")

                await session.delete(order)

                # Attempt to re-match in the same transaction
                await self.match_bid_ask_orders(session, subnet.id)

                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def match_bid_ask_orders(self, session: AsyncSession, subnet_id: int):
        """
        The core matching function: now uses row-level locks (with_for_update)
        on both Orders and their corresponding Tasks. This prevents concurrency
        issues where a Task might get status=SolverSelected but no 'ask' assigned.
        """

        # 1) Acquire row-level locks on unmatched BIDs
        bid_orders_stmt = (
            select(Order)
            .join(Task, Task.id == Order.bid_task_id)
            .filter(
                Order.subnet_id == subnet_id,
                Order.order_type == OrderType.Bid,
                Task.ask == None
            )
            .with_for_update()
            .options(joinedload(Order.bid_task))
        )
        bid_orders_result = await session.execute(bid_orders_stmt)
        bid_orders = bid_orders_result.scalars().all()

        # Lock the corresponding Tasks for these BIDs
        bid_task_ids = [o.bid_task_id for o in bid_orders if o.bid_task_id]
        if bid_task_ids:
            bid_tasks_stmt = (
                select(Task)
                .where(Task.id.in_(bid_task_ids))
                .with_for_update()
            )
            await session.execute(bid_tasks_stmt)

        # 2) Acquire row-level locks on unmatched ASKs
        ask_orders_stmt = (
            select(Order)
            .filter(
                Order.subnet_id == subnet_id,
                Order.order_type == OrderType.Ask,
                Order.ask_task_id == None
            )
            .with_for_update()
            .options(joinedload(Order.ask_task))
        )
        ask_orders_result = await session.execute(ask_orders_stmt)
        ask_orders = ask_orders_result.scalars().all()

        # Lock the corresponding Tasks for these ASKs
        ask_task_ids = [o.ask_task_id for o in ask_orders if o.ask_task_id]
        if ask_task_ids:
            ask_tasks_stmt = (
                select(Task)
                .where(Task.id.in_(ask_task_ids))
                .with_for_update()
            )
            await session.execute(ask_tasks_stmt)

        # Now we hold row-level locks on relevant Orders and Tasks in one transaction
        # Try matching them
        for bid in bid_orders:
            for ask in ask_orders:
                if bid.price >= ask.price:
                    bid_task = bid.bid_task
                    # set relationships in the same locked transaction
                    bid_task.ask = ask
                    bid_task.status = TaskStatus.SolverSelected
                    bid_task.time_solver_selected = datetime.utcnow()
                    session.add(bid_task)

                    # remove the matched ask so we don't reuse it
                    ask_orders.remove(ask)
                    break

        # flush now; the caller (create_order / etc.) will commit or rollback
        await session.flush()

    async def get_task(self, task_id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Task).options(joinedload(Task.job)).filter(Task.id == task_id)
            result = await session.execute(stmt)
            task = result.scalar_one_or_none()
            return task

    async def get_assigned_tasks(self, subnet_id: int):
        async with AsyncSessionLocal() as session:
            user_id = self.get_user_id()
            stmt = (
                select(Task)
                .join(Order, Task.id == Order.ask_task_id)
                .filter(
                    Order.order_type == OrderType.Ask,
                    Order.user_id == user_id,
                    Task.status == TaskStatus.SolverSelected,
                    Task.job.has(Job.subnet_id == subnet_id)
                )
            )
            result = await session.execute(stmt)
            tasks = result.scalars().all()
            return {'assigned_tasks': tasks}

    async def get_tasks_with_pagination_for_job(self, job_id: int, offset: int = 0, limit: int = 20):
        async with AsyncSessionLocal() as session:
            stmt = (
                select(Task)
                .filter_by(job_id=job_id)
                .order_by(Task.submitted_at.asc())
                .offset(offset)
                .limit(limit)
            )
            result = await session.execute(stmt)
            tasks = result.scalars().all()
            return tasks

    async def get_task_count_for_job(self, job_id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(func.count(Task.id)).filter_by(job_id=job_id)
            result = await session.execute(stmt)
            task_count = result.scalar_one()
            return task_count

    async def get_task_count_by_status_for_job(self, job_id: int, statuses: list[TaskStatus]):
        async with AsyncSessionLocal() as session:
            stmt = select(func.count(Task.id)).filter(
                Task.job_id == job_id,
                Task.status.in_(statuses)
            )
            result = await session.execute(stmt)
            task_status_count = result.scalar_one()
            return task_status_count

    # placeholders for extra checks
    async def should_check(self, task: Task):
        return False

    async def check_invalid(self, task: Task):
        return False

    async def submit_task_result(self, task_id: int, result: str) -> bool:
        async with AsyncSessionLocal() as session:
            task_stmt = select(Task).where(Task.id == task_id).options(
                joinedload(Task.job).joinedload(Job.subnet),
                joinedload(Task.bid).joinedload(Order.hold),
                joinedload(Task.ask).joinedload(Order.hold)
            )
            task_result = await session.execute(task_stmt)
            task = task_result.scalar_one_or_none()

            if not task:
                raise ValueError("Task not found")

            # Key check: must have an ask assigned if status is SolverSelected
            if not task.ask:
                raise PermissionError("No solver assignment")

            if task.ask.user_id != self.get_user_id():
                raise PermissionError("Not the solver")

            if task.status != TaskStatus.SolverSelected:
                raise ValueError("Task not in SolverSelected status")

            try:
                result_data = json.loads(result)
            except json.JSONDecodeError:
                result_data = None

            task.result = result_data
            task.status = TaskStatus.SanityCheckPending
            task.time_solved = datetime.now(timezone.utc)
            session.add(task)
            await session.commit()
            return {'success': True}

    async def finalize_check(self, task_id: int):
        async with AsyncSessionLocal() as session:
            task_stmt = select(Task).where(Task.id == task_id).options(
                joinedload(Task.job).joinedload(Job.subnet),
                joinedload(Task.bid).joinedload(Order.hold),
                joinedload(Task.ask).joinedload(Order.hold)
            )
            task_result = await session.execute(task_stmt)
            task = task_result.scalar_one_or_none()

            if not task:
                raise ValueError("Task not found")

            if task.status != TaskStatus.Checking:
                raise ValueError("Task not in Checking status, cannot finalize check")

            invalid = await self.check_invalid(task)
            if invalid:
                await self.resolve_task(session, task, task.result, correct=False)
            else:
                await self.resolve_task(session, task, task.result, correct=True)
            await session.commit()

    async def resolve_task(self, session: AsyncSession, task: Task, result: str, correct: bool):
        if correct:
            status = TaskStatus.ResolvedCorrect
        else:
            status = TaskStatus.ResolvedIncorrect

        task.result = result
        task.status = status
        task.time_solved = datetime.now(timezone.utc)
        session.add(task)

        if correct:
            await self.handle_correct_resolution_scenario(session, task)
        else:
            await self.handle_incorrect_resolution_scenario(session, task)

    async def handle_incorrect_resolution_scenario(self, session: AsyncSession, task: Task):
        ask_order = task.ask
        if not ask_order:
            raise ValueError("No ask order for incorrect scenario")
        if not ask_order.hold:
            raise ValueError("No hold for ask order in incorrect scenario")

        await self.charge_hold_fully(session, ask_order.hold, add_leftover_to_account=False)
        stake_amount = task.job.subnet.stake_multiplier * ask_order.price
        if stake_amount > ask_order.hold.used_amount:
            raise ValueError("data inconsistency in stake vs hold used")

        await self.add_platform_revenue(session, stake_amount, PlatformRevenueTxnType.Add)
        leftover = ask_order.hold.total_amount - stake_amount
        if leftover > 0:
            stmt = select(Account).where(Account.id == ask_order.hold.account_id)
            result = await session.execute(stmt)
            solver_account = result.scalar_one_or_none()
            if solver_account:
                await self.add_credits_transaction(session, solver_account, leftover, CreditTxnType.Add)

        bid_order = task.bid
        if bid_order and bid_order.hold:
            await self.free_funds_from_hold(session, bid_order.hold, bid_order.price, bid_order)
            if not bid_order.hold.charged and bid_order.hold.used_amount == 0.0:
                stmt = select(Account).where(Account.id == bid_order.hold.account_id)
                result = await session.execute(stmt)
                bidder_account = result.scalar_one_or_none()
                if bidder_account:
                    await self.add_credits_transaction(session, bidder_account, bid_order.hold.total_amount, CreditTxnType.Add)
                bid_order.hold.total_amount = 0.0

    async def handle_correct_resolution_scenario(self, session: AsyncSession, task: Task):
        bid_order = task.bid
        ask_order = task.ask
        if not bid_order or not bid_order.hold:
            raise ValueError("missing bid order/hold")
        if not ask_order or not ask_order.hold:
            raise ValueError("missing ask order/hold")

        subnet = task.job.subnet
        stake_amount = subnet.stake_multiplier * ask_order.price

        if not bid_order.hold.charged:
            await self.charge_hold_fully(session, bid_order.hold)
        if not ask_order.hold.charged:
            await self.charge_hold_fully(session, ask_order.hold)

        await self.free_funds_from_hold(session, ask_order.hold, stake_amount, ask_order)
        solver_user_id = ask_order.user_id
        stmt = select(Account).where(Account.user_id == solver_user_id)
        solver_acc_result = await session.execute(stmt)
        solver_account = solver_acc_result.scalar_one_or_none()
        if not solver_account:
            solver_account = await self.get_or_create_account(solver_user_id, session)

        platform_fee = bid_order.price * PLATFORM_FEE_PERCENTAGE
        solver_earnings = bid_order.price - platform_fee

        await self.add_earnings_transaction(session, solver_account, solver_earnings, EarningsTxnType.Add)
        await self.add_platform_revenue(session, platform_fee, PlatformRevenueTxnType.Add)

    async def get_last_task_with_status(self, job_id: int, statuses: list[TaskStatus]):
        async with AsyncSessionLocal() as session:
            stmt = (
                select(Task)
                .filter(
                    Task.job_id == job_id,
                    Task.status.in_(statuses)
                )
                .order_by(desc(Task.submitted_at))
                .limit(1)
            )
            result = await session.execute(stmt)
            task = result.scalar_one_or_none()
            return task

    async def check_and_cleanup_holds(self):
        async with AsyncSessionLocal() as session:
            logger.debug("Running hold cleanup check...")
            stmt_subnets = select(Subnet).distinct()
            subnets = (await session.execute(stmt_subnets)).scalars().all()
            now = datetime.utcnow()

            for subnet in subnets:
                required_expiry_buffer = timedelta(days=DISPUTE_PAYOUT_DELAY_DAYS)
                min_expiry = (
                    now
                    + timedelta(seconds=subnet.solve_period + subnet.dispute_period)
                    + required_expiry_buffer
                )

                stmt_orders = (
                    select(Order)
                    .join(Task, or_(Order.bid_task_id == Task.id, Order.ask_task_id == Task.id))
                    .where(
                        Order.subnet_id == subnet.id,
                        Task.status.in_([TaskStatus.SelectingSolver, TaskStatus.SolverSelected])
                    )
                    .options(joinedload(Order.hold))
                )
                orders = (await session.execute(stmt_orders)).scalars().all()

                for order in orders:
                    if order.hold and order.hold.expiry < min_expiry:
                        logger.info(f"Canceling order {order.id} due to hold expiry too soon.")
                        if order.order_type == OrderType.Bid:
                            amount_to_reverse = order.price
                        else:
                            amount_to_reverse = subnet.stake_multiplier * order.price

                        await self.free_funds_from_hold(session, order.hold, amount_to_reverse, order)
                        if not order.hold.charged and order.hold.used_amount == 0:
                            stmt_acc = select(Account).where(Account.id == order.hold.account_id)
                            acc_res = await session.execute(stmt_acc)
                            hold_account = acc_res.scalar_one_or_none()
                            if hold_account and order.hold.hold_type == HoldType.Credits:
                                await self.add_credits_transaction(
                                    session,
                                    hold_account,
                                    order.hold.total_amount,
                                    CreditTxnType.Add
                                )
                            order.hold.total_amount = 0.0

                        await session.delete(order)
            await session.commit()

    async def finalize_sanity_check(self, task_id: int, is_valid: bool):
        async with AsyncSessionLocal() as session:
            task_stmt = select(Task).where(Task.id == task_id)
            task_result = await session.execute(task_stmt)
            task = task_result.scalar_one_or_none()

            if not task:
                raise ValueError("Task not found for sanity check finalization")

            if task.status != TaskStatus.SanityCheckPending:
                raise ValueError("Task is not in SanityCheckPending status")

            task.status = TaskStatus.ResolvedCorrect if is_valid else TaskStatus.ResolvedIncorrect
            task.time_solved = datetime.now(timezone.utc)
            session.add(task)
            await session.commit()
