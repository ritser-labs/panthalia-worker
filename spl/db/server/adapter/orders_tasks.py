# spl/db/server/adapter/orders_tasks.py

import logging
from datetime import datetime, timezone, timedelta
from sqlalchemy import select, update, desc, func, or_
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert
from typing import Optional
import json

from ....models import (
    AsyncSessionLocal, Job, Task, TaskStatus, Subnet,
    Order, Account, OrderType, HoldType, CreditTxnType, EarningsTxnType, PlatformRevenueTxnType,
    CreditTransaction, HoldTransaction, Hold, EarningsTransaction
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
                raise PermissionError("No access to create a task for this job.")
            
            # **NEW**: Forbid if the job is not active
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
        hold_id: Optional[int]
    ):
        async with AsyncSessionLocal() as session:
            try:
                stmt = select(Job).filter_by(id=job_id).options(joinedload(Job.subnet))
                result = await session.execute(stmt)
                job = result.scalar_one_or_none()
                if not job:
                    raise ValueError(f"Job {job_id} not found.")
                user_id = job.user_id
                account = await self.get_or_create_account(user_id, session)

                # **NEW**: Forbid if job is inactive
                if not job.active:
                    raise ValueError(f"Cannot create tasks for an inactive job (job_id={job_id}).")

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
        """
        Patched in tests (with eager-loading) to avoid MissingGreenlet errors
        after the session is out of scope.
        """
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

    async def finalize_sanity_check(self, task_id: int, is_valid: bool):
        """
        FIXED version: do all DB ops in a single session + eager load,
        avoiding lazy-load issues after commit.
        """
        async with AsyncSessionLocal() as session:
            # Eager-load the Task, plus bid->hold, ask->hold, job->subnet
            stmt = (
                select(Task)
                .options(
                    joinedload(Task.bid).joinedload(Order.hold),
                    joinedload(Task.ask).joinedload(Order.hold),
                    joinedload(Task.job).joinedload(Job.subnet)
                )
                .where(Task.id == task_id)
            )
            res = await session.execute(stmt)
            task = res.scalar_one_or_none()

            if not task:
                raise ValueError("Task not found for sanity check finalization")

            if task.status != TaskStatus.SanityCheckPending:
                raise ValueError(f"Task is not in SanityCheckPending status, cannot finalize. Found {task.status}.")

            # Decide final status
            if is_valid:
                task.status = TaskStatus.ResolvedCorrect
            else:
                task.status = TaskStatus.ResolvedIncorrect
            task.time_solved = datetime.now(timezone.utc)

            # Actually do the correct or incorrect resolution
            if is_valid:
                await self.handle_correct_resolution_scenario(session, task)
            else:
                await self.handle_incorrect_resolution_scenario(session, task)

            session.add(task)
            await session.flush()
            await session.commit()

    async def finalize_check(self, task_id: int):
        """
        Unused by the failing tests in question, but left for reference.
        """
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

    async def handle_correct_resolution_scenario(self, session: AsyncSession, task: Task):
        """
        If the solver is correct:
        - Buyer does NOT get leftover refund for the difference between (hold.total_amount) and (bid_price).
        - The buyer's hold is fully charged for 'bid_price'.
        - The solver's stake is freed (i.e., credited back if it was from indefinite credits).
        - The solver's earnings_balance increases by (bid_price - platform_fee).
        - A new earnings hold is created for the solver.
        - The platform revenue increases by the fee.
        """

        bid_order = task.bid
        ask_order = task.ask
        if not bid_order or not bid_order.hold:
            raise ValueError("Missing buyer hold in correct scenario.")
        if not ask_order or not ask_order.hold:
            raise ValueError("Missing solver hold in correct scenario.")

        subnet = task.job.subnet
        buyer_price = bid_order.price
        # Use your defined fee percentage
        PLATFORM_FEE_PERCENTAGE = 0.1
        platform_fee = buyer_price * PLATFORM_FEE_PERCENTAGE
        solver_earnings = buyer_price - platform_fee
        stake_amount = subnet.stake_multiplier * ask_order.price

        # (1) COMMENTED OUT: No leftover refund to buyer on correct resolution
        # leftover = bid_order.hold.total_amount - buyer_price
        # if leftover > 0:
        #     leftover_clamped = min(leftover, bid_order.hold.used_amount)
        #     if leftover_clamped > 0:
        #         await self.free_funds_from_hold(session, bid_order.hold, leftover_clamped, bid_order)
        #         self.logger.info(
        #             f"[handle_correct_resolution_scenario] Freed leftover={leftover_clamped:.2f} from buyer hold.id={bid_order.hold.id}"
        #         )

        # (2) Fully charge buyer's hold => actual cost
        if not bid_order.hold.charged:
            await self.charge_hold_fully(session, bid_order.hold)
            self.logger.info(
                f"[handle_correct_resolution_scenario] Buyer hold charged for {buyer_price:.2f}"
            )

        # (3) Free solver's stake usage (if indefinite credits => solver's credits go back up)
        if not ask_order.hold.charged:
            await self.free_funds_from_hold(session, ask_order.hold, stake_amount, ask_order)
            self.logger.info(
                f"[handle_correct_resolution_scenario] Freed stake usage {stake_amount:.2f} from solver hold.id={ask_order.hold.id}"
            )

        # (4) Add solver's earnings and create an earnings hold for them
        solver_user_id = ask_order.user_id
        solver_account = await self.get_or_create_account(solver_user_id, session=session)

        if solver_earnings > 0:
            # Immediately add to solver.earnings_balance
            await self.add_earnings_transaction(
                session,
                solver_account,
                solver_earnings,
                EarningsTxnType.Add
            )
            self.logger.info(f"Added {solver_earnings:.2f} to solver's earnings_balance")

            # Also create an Earnings hold that expires in 1 year
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
            self.logger.info(
                f"Created Earnings hold for solver, {solver_earnings:.2f}, expires in ~1 year"
            )

        # (5) Finally, record the platform fee
        await self.add_platform_revenue(session, platform_fee, PlatformRevenueTxnType.Add)



    async def handle_incorrect_resolution_scenario(self, session: AsyncSession, task: Task):
        """
        Incorrect solver solution => solver is penalized => solver hold fully charged => leftover => new hold
        Buyer hold is freed => leftover remains in the same hold uncharged.
        Solver's staked portion => platform revenue.
        """
        ask_order = task.ask
        if not ask_order or not ask_order.hold:
            raise ValueError("Missing solver hold in incorrect scenario.")
        solver_hold = ask_order.hold

        # 1) fully charge solver hold => leftover => new hold
        if not solver_hold.charged:
            await self.charge_hold_fully(session, solver_hold)

        # 2) buyer => free usage => if any
        bid_order = task.bid
        if bid_order and bid_order.hold:
            price = bid_order.price
            # free the buyer's usage => leftover remains in that hold
            await self.free_funds_from_hold(session, bid_order.hold, price, bid_order)

        # 3) solver staked portion => platform revenue
        subnet = task.job.subnet
        stake_amount = ask_order.price * subnet.stake_multiplier
        # We effectively consider stake_amount "lost" by the solver. So let's record it:
        await self.add_platform_revenue(session, stake_amount, PlatformRevenueTxnType.Add)

        # If the leftover hold created by charge_hold_fully => that leftover still belongs to solver, 
        # but "used_amount" portion is staked + burned => i.e. platform got it. 
        # No indefinite credits to solver => leftover is in the new leftover hold with same expiry.


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

    ###########################################################################
    async def check_and_cleanup_holds(self):
        """
        Periodically called to:
        1) Cancel orders whose hold expires too soon to finish solve+dispute.
        2) Remove leftover from deposit-based or earnings-based holds after expiry,
           but without adjusting any account.credits_balance or earnings_balance (which no longer exist).
        """
        now = datetime.utcnow()
        async with AsyncSessionLocal() as session:
            async with session.begin():
                logging.debug("[check_and_cleanup_holds] Running hold cleanup check...")

                subnets_stmt = select(Subnet).distinct()
                subnets_result = await session.execute(subnets_stmt)
                subnets = subnets_result.scalars().all()

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
                        .where(Order.subnet_id == subnet.id)
                        .options(joinedload(Order.hold))
                    )
                    orders_result = await session.execute(stmt_orders)
                    orders_result = orders_result.unique()
                    orders = orders_result.scalars().all()

                    for order in orders:
                        hold = order.hold
                        if not hold:
                            continue
                        if hold.expiry < min_expiry:
                            logging.info(f"[check_and_cleanup_holds] Canceling order {order.id} due to hold expiry too soon.")
                            if order.order_type.value == 'bid':
                                amount_to_reverse = order.price
                            else:
                                amount_to_reverse = subnet.stake_multiplier * order.price

                            if hold.used_amount < amount_to_reverse:
                                raise ValueError("Not enough used amount in hold to free for canceled order.")

                            hold.used_amount -= amount_to_reverse
                            hold_txn = HoldTransaction(
                                hold_id=hold.id,
                                order_id=order.id,
                                amount=-amount_to_reverse
                            )
                            session.add(hold_txn)
                            session.delete(order)

                stmt_all_holds = select(Hold).options(joinedload(Hold.hold_transactions))
                all_holds_result = await session.execute(stmt_all_holds)
                all_holds_result = all_holds_result.unique()
                all_holds = all_holds_result.scalars().all()

                for hold in all_holds:
                    if hold.charged:
                        continue

                    if hold.expiry < now:
                        leftover = hold.total_amount - hold.used_amount

                        # No updates to any account.credits_balance or account.earnings_balance.
                        # We simply fully charge this hold.
                        hold.used_amount = hold.total_amount
                        hold.charged = True
                        hold.charged_amount = hold.total_amount

                        logging.info(
                            f"[check_and_cleanup_holds] HOLD {hold.id} {hold.hold_type} expired => leftover {leftover} forcibly removed/burned"
                        )

                logging.debug("[check_and_cleanup_holds] Hold cleanup completed successfully.")



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

        async with AsyncSessionLocal() as session:
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
                "total_open_orders_dollar": float(total_open_orders_dollar),
                "num_open_orders": int(num_open_orders),
                "num_completed_tasks": int(num_completed_tasks),
                "volume": float(volume)
            }

    async def get_orders_for_user(self):
        async with AsyncSessionLocal() as session:
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