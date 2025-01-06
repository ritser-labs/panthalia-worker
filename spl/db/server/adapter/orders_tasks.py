# file: spl/db/server/adapter/orders_tasks.py

import logging
from datetime import datetime, timezone, timedelta
from sqlalchemy import select, update, desc, func, or_
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Dict
import json

from ....models import (
    Job, Task, TaskStatus, Subnet,
    Order, Account, OrderType, HoldType, CreditTxnType, EarningsTxnType, PlatformRevenueTxnType,
    CreditTransaction, HoldTransaction, Hold, EarningsTransaction
)
from .holds import DBAdapterHoldsMixin

logger = logging.getLogger(__name__)
PLATFORM_FEE_PERCENTAGE = 0.1
DISPUTE_PAYOUT_DELAY_DAYS = 1

class DBAdapterOrdersTasksMixin:
    async def get_job(self, job_id: int):
        async with self.get_async_session() as session:
            stmt = select(Job).filter_by(id=job_id)
            res = await session.execute(stmt)
            return res.scalar_one_or_none()

    async def update_job_iteration(self, job_id: int, new_iteration: int):
        async with self.get_async_session() as session:
            stmt = update(Job).where(Job.id == job_id).values(iteration=new_iteration)
            await session.execute(stmt)
            await session.commit()

    async def update_job_sot_url(self, job_id: int, new_sot_url: str):
        async with self.get_async_session() as session:
            stmt = update(Job).where(Job.id == job_id).values(sot_url=new_sot_url)
            await session.execute(stmt)
            await session.commit()

    async def mark_job_as_done(self, job_id: int):
        async with self.get_async_session() as session:
            stmt = update(Job).where(Job.id == job_id).values(done=True)
            await session.execute(stmt)
            await session.commit()

    async def create_task(self, job_id: int, job_iteration: int, status: TaskStatus, params: str):
        async with self.get_async_session() as session:
            job_stmt = select(Job).where(Job.id == job_id)
            job_res = await session.execute(job_stmt)
            job = job_res.scalar_one_or_none()
            if not job or job.user_id != self.get_user_id():
                raise PermissionError("No access to create a task for this job.")
            if not job.active:
                raise ValueError(f"Cannot create a task for an inactive job (job_id={job_id}).")

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
        async with self.get_async_session() as session:
            stmt = update(Task).where(Task.id == task_id, Task.job_id == job_id)
            stmt = stmt.values(status=status, result=result)
            await session.execute(stmt)
            await session.commit()

    async def create_order(
        self,
        task_id: int | None,
        subnet_id: int,
        order_type: OrderType,
        price: int,
        hold_id: Optional[int] = None
    ):
        async with self.get_async_session() as session:
            try:
                user_id = self.get_user_id()
                account = await self.get_or_create_account(user_id, session)

                subnet_stmt = select(Subnet).where(Subnet.id == subnet_id)
                subnet_res = await session.execute(subnet_stmt)
                subnet = subnet_res.scalar_one_or_none()
                if not subnet:
                    raise ValueError("Subnet not found")

                if order_type == OrderType.Bid:
                    if task_id is None:
                        raise ValueError("task_id required for bid")
                    task_stmt = select(Task).where(Task.id == task_id).options(joinedload(Task.job))
                    task_res = await session.execute(task_stmt)
                    task = task_res.scalar_one_or_none()
                    if not task or task.job.user_id != user_id:
                        raise PermissionError("Only the job owner can submit bids for this task.")

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
                        raise ValueError("task_id must be None for an ask")
                    hold = await self.select_hold_for_order(session, account, subnet, order_type, price, hold_id)
                    new_order = Order(
                        order_type=order_type,
                        price=price,
                        subnet_id=subnet_id,
                        user_id=user_id,
                        account_id=account.id,
                    )
                    session.add(new_order)
                    await session.flush()
                    stake_amount = subnet.stake_multiplier * price
                    await self.reserve_funds_on_hold(session, hold, stake_amount, new_order)
                    new_order.hold_id = hold.id

                else:
                    raise ValueError("Invalid order type")

                await self.match_bid_ask_orders(session, subnet_id)
                await session.commit()
                return new_order.id

            except Exception:
                await session.rollback()
                raise

    async def get_num_orders(self, subnet_id: int, order_type: str, matched: Optional[bool]):
        async with self.get_async_session() as session:
            stmt = select(func.count(Order.id)).where(
                Order.subnet_id == subnet_id,
                Order.order_type == order_type
            )
            if matched is not None:
                if matched:
                    stmt = stmt.where(or_(Order.bid_task_id.isnot(None), Order.ask_task_id.isnot(None)))
                else:
                    stmt = stmt.where(Order.bid_task_id.is_(None), Order.ask_task_id.is_(None))

            res = await session.execute(stmt)
            return {'num_orders': res.scalar_one()}

    async def create_bids_and_tasks(
        self,
        job_id: int,
        num_tasks: int,
        price: int,
        params: str,
        hold_id: Optional[int] = None
    ) -> Optional[Dict[str, List[Dict[str, int]]]]:
        async with self.get_async_session() as session:
            try:
                stmt = select(Job).where(Job.id == job_id).options(joinedload(Job.subnet))
                res = await session.execute(stmt)
                job = res.scalar_one_or_none()
                if not job:
                    raise ValueError(f"Job {job_id} not found.")
                if not job.active:
                    raise ValueError(f"Cannot create tasks for an inactive job (job_id={job_id}).")

                user_id = job.user_id
                account = await self.get_or_create_account(user_id, session)

                created_items = []
                for _ in range(num_tasks):
                    job.iteration += 1
                    new_task = Task(
                        job_id=job.id,
                        job_iteration=job.iteration,
                        status=TaskStatus.SelectingSolver,
                        params=params
                    )
                    session.add(new_task)
                    await session.flush()

                    new_bid = Order(
                        order_type=OrderType.Bid,
                        price=price,
                        subnet_id=job.subnet_id,
                        user_id=user_id,
                        account_id=account.id,
                        bid_task=new_task
                    )
                    session.add(new_bid)
                    await session.flush()

                    deposit_hold = await self.select_hold_for_order(
                        session, account, job.subnet, OrderType.Bid, price, hold_id
                    )
                    await self.reserve_funds_on_hold(session, deposit_hold, price, new_bid)
                    new_bid.hold_id = deposit_hold.id

                    created_items.append({
                        "task_id": new_task.id,
                        "bid_id": new_bid.id
                    })

                await self.match_bid_ask_orders(session, job.subnet_id)
                await session.commit()
                return {"created_items": created_items}

            except Exception:
                await session.rollback()
                raise

    async def delete_order(self, order_id: int):
        async with self.get_async_session() as session:
            try:
                order_stmt = select(Order).where(Order.id == order_id)
                res = await session.execute(order_stmt)
                order = res.scalar_one_or_none()
                if not order or order.user_id != self.get_user_id():
                    raise PermissionError("No access to delete this order.")

                if (order.order_type == OrderType.Bid and order.bid_task and order.bid_task.ask) or \
                   (order.order_type == OrderType.Ask and order.ask_task and order.ask_task.bid):
                    raise ValueError("Cannot delete an order that is already matched.")

                subnet_stmt = select(Subnet).where(Subnet.id == order.subnet_id)
                subnet_res = await session.execute(subnet_stmt)
                subnet = subnet_res.scalar_one_or_none()
                if not subnet:
                    raise ValueError("Subnet not found")

                if order.order_type == OrderType.Bid:
                    freed_amount = order.price
                else:
                    freed_amount = subnet.stake_multiplier * order.price

                if not order.hold:
                    raise ValueError("No hold found for this order.")
                await self.free_funds_from_hold(session, order.hold, freed_amount, order)

                session.delete(order)
                await self.match_bid_ask_orders(session, order.subnet_id)
                await session.commit()

            except Exception:
                await session.rollback()
                raise

    async def match_bid_ask_orders(self, session: AsyncSession, subnet_id: int):
        unmatched_bids_stmt = (
            select(Order)
            .join(Task, Task.id == Order.bid_task_id)
            .where(
                Order.subnet_id == subnet_id,
                Order.order_type == OrderType.Bid,
                Task.ask == None
            )
            .with_for_update()
            .options(joinedload(Order.bid_task))
        )
        unmatched_bids_res = await session.execute(unmatched_bids_stmt)
        unmatched_bids = unmatched_bids_res.scalars().all()

        unmatched_asks_stmt = (
            select(Order)
            .where(
                Order.subnet_id == subnet_id,
                Order.order_type == OrderType.Ask,
                Order.ask_task_id == None
            )
            .with_for_update()
            .options(joinedload(Order.ask_task))
        )
        unmatched_asks_res = await session.execute(unmatched_asks_stmt)
        unmatched_asks = unmatched_asks_res.scalars().all()

        bid_task_ids = [b.bid_task_id for b in unmatched_bids if b.bid_task_id]
        if bid_task_ids:
            tasks_stmt = select(Task).where(Task.id.in_(bid_task_ids)).with_for_update()
            await session.execute(tasks_stmt)
        ask_task_ids = [a.ask_task_id for a in unmatched_asks if a.ask_task_id]
        if ask_task_ids:
            tasks_stmt = select(Task).where(Task.id.in_(ask_task_ids)).with_for_update()
            await session.execute(tasks_stmt)

        for bid in unmatched_bids:
            for ask in unmatched_asks:
                if bid.price >= ask.price:
                    bid_task = bid.bid_task
                    bid_task.ask = ask
                    bid_task.status = TaskStatus.SolverSelected
                    bid_task.time_solver_selected = datetime.utcnow()
                    session.add(bid_task)
                    unmatched_asks.remove(ask)
                    break

        await session.flush()

    async def get_task(self, task_id: int):
        async with self.get_async_session() as session:
            stmt = select(Task).where(Task.id == task_id).options(joinedload(Task.job))
            res = await session.execute(stmt)
            return res.scalar_one_or_none()

    async def get_assigned_tasks(self, subnet_id: int):
        async with self.get_async_session() as session:
            user_id = self.get_user_id()
            stmt = (
                select(Task)
                .join(Order, Order.ask_task_id == Task.id)
                .where(
                    Order.order_type == OrderType.Ask,
                    Order.user_id == user_id,
                    Task.status == TaskStatus.SolverSelected,
                    Task.job.has(Job.subnet_id == subnet_id)
                )
            )
            res = await session.execute(stmt)
            tasks = res.scalars().all()
            return {"assigned_tasks": tasks}

    async def get_tasks_with_pagination_for_job(self, job_id: int, offset=0, limit=20):
        async with self.get_async_session() as session:
            stmt = (
                select(Task)
                .where(Task.job_id == job_id)
                .order_by(Task.submitted_at.asc())
                .offset(offset)
                .limit(limit)
            )
            res = await session.execute(stmt)
            return res.scalars().all()

    async def get_task_count_for_job(self, job_id: int):
        async with self.get_async_session() as session:
            stmt = select(func.count(Task.id)).where(Task.job_id == job_id)
            res = await session.execute(stmt)
            return res.scalar_one()

    async def get_task_count_by_status_for_job(self, job_id: int, statuses: List[str]):
        async with self.get_async_session() as session:
            stmt = select(func.count(Task.id)).where(
                Task.job_id == job_id,
                Task.status.in_(statuses)
            )
            res = await session.execute(stmt)
            return res.scalar_one()

    async def should_check(self, task: Task):
        return False

    async def check_invalid(self, task: Task):
        return False

    async def submit_task_result(self, task_id: int, result: str) -> bool:
        async with self.get_async_session() as session:
            q = (
                select(Task)
                .where(Task.id == task_id)
                .options(
                    joinedload(Task.job).joinedload(Job.subnet),
                    joinedload(Task.bid).joinedload(Order.hold),
                    joinedload(Task.ask).joinedload(Order.hold)
                )
            )
            r = await session.execute(q)
            task = r.scalar_one_or_none()
            if not task:
                raise ValueError(f"Task {task_id} not found")

            if not task.ask:
                raise PermissionError("No solver assignment, cannot submit result.")
            if task.ask.user_id != self.get_user_id():
                raise PermissionError("Not your task to submit a result for.")
            if task.status != TaskStatus.SolverSelected:
                raise ValueError(f"Task {task_id} is not in SolverSelected status")

            try:
                result_data = json.loads(result)
            except json.JSONDecodeError:
                result_data = None

            task.result = result_data
            task.status = TaskStatus.SanityCheckPending
            task.time_solved = datetime.now(timezone.utc)
            session.add(task)
            await session.commit()
            return True

    async def finalize_sanity_check(self, task_id: int, is_valid: bool, force: bool = False):
        async with self.get_async_session() as session:
            stmt = (
                select(Task)
                .where(Task.id == task_id)
                .options(
                    joinedload(Task.bid).joinedload(Order.hold),
                    joinedload(Task.ask).joinedload(Order.hold),
                    joinedload(Task.job).joinedload(Job.subnet)
                )
            )
            r = await session.execute(stmt)
            task = r.scalar_one_or_none()
            if not task:
                raise ValueError("Task not found.")

            if task.status in [TaskStatus.ResolvedCorrect, TaskStatus.ResolvedIncorrect]:
                logger.warning(
                    f"[finalize_sanity_check] Task {task_id} is already in {task.status} => skipping re-finalization."
                )
                return
            if not force and task.status != TaskStatus.SanityCheckPending:
                raise ValueError(f"Task {task_id} must be in SanityCheckPending to finalize sanity check.")

            if is_valid:
                task.status = TaskStatus.ResolvedCorrect
                await self.handle_correct_resolution_scenario(session, task)
            else:
                task.status = TaskStatus.ResolvedIncorrect
                await self.handle_incorrect_resolution_scenario(session, task)

            task.time_solved = datetime.now(timezone.utc)
            session.add(task)
            await session.flush()
            await session.commit()

    async def finalize_check(self, task_id: int):
        pass

    async def resolve_task(self, session: AsyncSession, task: Task, result: str, correct: bool):
        if correct:
            task.status = TaskStatus.ResolvedCorrect
            await self.handle_correct_resolution_scenario(session, task)
        else:
            task.status = TaskStatus.ResolvedIncorrect
            await self.handle_incorrect_resolution_scenario(session, task)

    async def handle_correct_resolution_scenario(self, session: AsyncSession, task: Task):
        bid_order = task.bid
        ask_order = task.ask
        if not bid_order or not bid_order.hold:
            raise ValueError("Missing buyer hold in correct scenario.")
        if not ask_order or not ask_order.hold:
            raise ValueError("Missing solver hold in correct scenario.")

        subnet = task.job.subnet
        buyer_price = bid_order.price
        platform_fee = buyer_price * PLATFORM_FEE_PERCENTAGE
        solver_earnings = buyer_price - platform_fee
        solver_stake = subnet.stake_multiplier * ask_order.price

        buyer_hold_to_charge = bid_order.hold
        await self.charge_hold_for_price(session, buyer_hold_to_charge, buyer_price)
        logger.info(
            f"[handle_correct_resolution_scenario] Buyer hold.id={buyer_hold_to_charge.id} charged for {buyer_price:.2f}"
        )

        await self.free_funds_from_hold(session, ask_order.hold, solver_stake, ask_order)
        logger.info(
            f"[handle_correct_resolution_scenario] Freed solver stake usage {solver_stake:.2f} from solver hold.id={ask_order.hold.id}"
        )

        solver_user_id = ask_order.user_id
        solver_account = await self.get_or_create_account(solver_user_id, session)
        if solver_earnings > 0:
            await self.add_earnings_transaction(session, solver_account, solver_earnings, EarningsTxnType.Add)
            logger.info(
                f"[handle_correct_resolution_scenario] Added {solver_earnings:.2f} to solver's earnings_balance"
            )
            new_earnings_hold = Hold(
                account_id=solver_account.id,
                user_id=solver_user_id,
                hold_type=HoldType.Earnings,
                total_amount=solver_earnings,
                used_amount=0.0,
                expiry=datetime.utcnow() + timedelta(days=365),
                charged=False,
                charged_amount=0.0,
                parent_hold_id=None
            )
            session.add(new_earnings_hold)
            logger.info(
                f"[handle_correct_resolution_scenario] Created Earnings hold=None for solver, {solver_earnings:.2f}, expires ~1 year"
            )

        await self.add_platform_revenue(session, platform_fee, PlatformRevenueTxnType.Add)

    async def handle_incorrect_resolution_scenario(self, session: AsyncSession, task: Task):
        bid_order = task.bid
        ask_order = task.ask
        if not ask_order or not ask_order.hold:
            raise ValueError("Missing solver hold in incorrect scenario.")
        solver_hold = ask_order.hold

        leftover_amount = bid_order.price
        await self.free_funds_from_hold(session, bid_order.hold, leftover_amount, bid_order)
        logger.info(
            f"[handle_incorrect_resolution_scenario] Freed leftover amount={leftover_amount:.2f} from buyer hold.id={bid_order.hold.id}"
        )
        subnet = task.job.subnet
        solver_stake = ask_order.price * subnet.stake_multiplier
        await self.charge_hold_for_price(session, solver_hold, solver_stake)
        await self.add_platform_revenue(session, solver_stake, PlatformRevenueTxnType.Add)

    async def get_last_task_with_status(self, job_id: int, statuses: list[TaskStatus]):
        async with self.get_async_session() as session:
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

    ###########################################################################
    async def check_and_cleanup_holds(self):
        """
        Perform the following cleanup tasks for holds:
        1) For unmatched orders with holds expiring soon (expiry <= now + solve_period + dispute_period),
        cancel (delete) these orders immediately.
        2) For tasks in SolverSelected status where the solver has taken too long (exceeding solve_period),
        finalize the task as ResolvedIncorrect automatically.
        3) For expired holds (where expiry < now):
        - If the hold is a CreditCard hold and `used_amount == 0`, zero out `total_amount` and mark as unusable.
        - Otherwise, charge the hold fully (sets charged=True, zeroes out leftover).
        """
        now = datetime.utcnow()
        async with self.get_async_session() as session:
            async with session.begin():
                logging.debug("[check_and_cleanup_holds] Running hold cleanup check...")

                # STEP 1: Cancel unmatched orders with holds expiring soon
                unmatched_orders_to_delete = []
                unmatched_q = (
                    select(Order, Hold, Subnet)
                    .join(Hold, Order.hold_id == Hold.id)
                    .join(Subnet, Subnet.id == Order.subnet_id)
                    .where(
                        Order.bid_task_id.is_(None),  # unmatched
                        Order.ask_task_id.is_(None)
                    )
                )
                unmatched_res = await session.execute(unmatched_q)
                for order, hold, subnet in unmatched_res:
                    cutoff_time = now + timedelta(seconds=(subnet.solve_period + subnet.dispute_period))
                    if hold.expiry <= cutoff_time:
                        unmatched_orders_to_delete.append(order.id)

                # STEP 2: Resolve overdue solver tasks as incorrect
                tasks_to_fail = []
                tasks_q = (
                    select(Task, Subnet)
                    .join(Job, Task.job_id == Job.id)
                    .join(Subnet, Job.subnet_id == Subnet.id)
                    .where(Task.status == TaskStatus.SolverSelected)
                )
                tasks_res = await session.execute(tasks_q)
                for task, subnet in tasks_res:
                    if not task.time_solver_selected:
                        continue
                    deadline = task.time_solver_selected + timedelta(seconds=subnet.solve_period)
                    if now > deadline:
                        tasks_to_fail.append(task.id)

                # STEP 3: Handle expired holds
                expired_holds_q = (
                    select(Hold)
                    .where(
                        Hold.charged == False,  # still active
                        Hold.expiry < now
                    )
                )
                expired_holds_res = await session.execute(expired_holds_q)
                expired_holds = expired_holds_res.scalars().all()

                for hold in expired_holds:
                    if hold.hold_type == HoldType.CreditCard and hold.used_amount == 0:
                        # Special case: Unused credit card holds are marked as unusable
                        hold.total_amount = 0.0
                        hold.charged = True
                        logger.info(
                            f"[check_and_cleanup_holds] Expired unused CC hold {hold.id} marked unusable."
                        )
                    else:
                        # Charge other expired holds fully
                        leftover = hold.total_amount - hold.used_amount
                        if leftover > 0:
                            logger.warning(
                                f"[check_and_cleanup_holds] Forcing leftover {leftover:.2f} as used for expired hold {hold.id}."
                            )
                            hold.used_amount = hold.total_amount
                        await self.charge_hold_for_price(session, hold, hold.total_amount)
                        logger.info(f"[check_and_cleanup_holds] Expired hold {hold.id} charged fully.")

            # STEP 1 (continued): Delete unmatched orders
            for order_id in unmatched_orders_to_delete:
                try:
                    await self.delete_order(order_id)
                    logger.info(f"[check_and_cleanup_holds] Deleted unmatched order {order_id} due to hold expiry.")
                except Exception as e:
                    logger.error(f"[check_and_cleanup_holds] Failed to delete order {order_id}: {e}")

            # STEP 2 (continued): Finalize overdue tasks as incorrect
            for task_id in tasks_to_fail:
                try:
                    await self.finalize_sanity_check(task_id, is_valid=False, force=True)
                    logger.info(f"[check_and_cleanup_holds] Finalized task {task_id} as incorrect due to solver timeout.")
                except Exception as e:
                    logger.error(f"[check_and_cleanup_holds] Failed to finalize task {task_id} as incorrect: {e}")



    async def get_global_stats(self) -> dict:
        """
        Return a dictionary of various global statistics:
          - total $ in open (unmatched) orders
          - number of open (unmatched) orders
          - number of tasks in completed statuses
          - volume = sum of bid prices for tasks that ended up completed
        """
        from ....models import TaskStatus, OrderType

        completed_statuses = [TaskStatus.ResolvedCorrect, TaskStatus.ResolvedIncorrect]

        async with self.get_async_session() as session:
            # total $ in open orders
            stmt_sum_open = select(func.sum(Order.price)).where(
                Order.bid_task_id.is_(None),
                Order.ask_task_id.is_(None)
            )
            sum_open_result = await session.execute(stmt_sum_open)
            total_open_orders_dollar = sum_open_result.scalar() or 0.0

            # num of open (unmatched) orders
            stmt_count_open = select(func.count(Order.id)).where(
                Order.bid_task_id.is_(None),
                Order.ask_task_id.is_(None)
            )
            count_open_result = await session.execute(stmt_count_open)
            num_open_orders = count_open_result.scalar() or 0

            # num completed tasks
            stmt_count_completed = select(func.count(Task.id)).where(
                Task.status.in_(completed_statuses)
            )
            count_completed_result = await session.execute(stmt_count_completed)
            num_completed_tasks = count_completed_result.scalar() or 0

            # volume (for completed matched bids)
            # We'll interpret "volume" as sum of the *bid* side for tasks that ended up completed.
            stmt_volume = (
                select(func.sum(Order.price))
                .join(Task, Order.bid_task_id == Task.id)
                .where(
                    Order.order_type == OrderType.Bid,
                    Order.ask_task_id.isnot(None),  # matched with an ask
                    Task.status.in_(completed_statuses)
                )
            )
            volume_result = await session.execute(stmt_volume)
            volume = volume_result.scalar() or 0.0

            return {
                "total_open_orders_dollar": int(total_open_orders_dollar),
                "num_open_orders": int(num_open_orders),
                "num_completed_tasks": int(num_completed_tasks),
                "volume": int(volume)
            }

    async def get_orders_for_user(self):
        async with self.get_async_session() as session:
            user_id = self.get_user_id()  # use the session-based user
            stmt = (
                select(Order)
                .where(Order.user_id == user_id)
                .order_by(Order.id.desc())
            )
            result = await session.execute(stmt)
            orders = result.scalars().all()
            # convert each to as_dict if you want to return it directly
            return [o.as_dict() for o in orders]