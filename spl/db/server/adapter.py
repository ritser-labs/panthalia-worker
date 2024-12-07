import asyncio
import logging
import json
import traceback
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List
import types
from eth_account import Account as EthAccount
from sqlalchemy import select, update, desc, func
from sqlalchemy.orm import joinedload, selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from ...models import (
    AsyncSessionLocal, Job, Task, TaskStatus, Plugin, StateUpdate, Subnet,
    Perm, Sot, PermDescription, PermType, Base, init_db, ServiceType,
    Instance, Order, Account, OrderType, AccountTransaction, AccountTxnType,
    AccountKey, Hold, HoldTransaction, HoldType, CreditTransaction, CreditTxnType,
    EarningsTransaction, EarningsTxnType, PlatformRevenue, PlatformRevenueTxnType
)
from ...auth.view import get_user_id
from ...util.enums import str_to_enum

logger = logging.getLogger(__name__)

# CONFIG CONSTANTS
PLATFORM_FEE_PERCENTAGE = 0.1  # 10% fee
DISPUTE_PAYOUT_DELAY_DAYS = 1
MINIMUM_PAYOUT_AMOUNT = 50.0

class DBAdapterServer:
    def __init__(self):
        asyncio.run(init_db())

    async def get_or_create_account(self, user_id: str, session: Optional[AsyncSession] = None):
        own_session = False
        if session is None:
            session = AsyncSessionLocal()
            own_session = True
        try:
            stmt = select(Account).where(Account.user_id == user_id)
            result = await session.execute(stmt)
            account = result.scalar_one_or_none()

            if account:
                return account

            new_account = Account(
                user_id=user_id,
                credits_balance=0.0,
                earnings_balance=0.0,
                deposited_at=datetime.utcnow()
            )
            session.add(new_account)
            await session.commit()
            await session.refresh(new_account)
            return new_account
        finally:
            if own_session:
                await session.close()

    async def add_credits_transaction(self, session: AsyncSession, account: Account, amount: float, txn_type: CreditTxnType):
        if txn_type == CreditTxnType.Subtract:
            if account.credits_balance < amount:
                raise ValueError("insufficient credits to subtract")
            account.credits_balance -= amount
        elif txn_type == CreditTxnType.Add:
            account.credits_balance += amount

        new_credit_txn = CreditTransaction(
            account_id=account.id,
            user_id=account.user_id,
            amount=amount,
            txn_type=txn_type,
        )
        session.add(new_credit_txn)

    async def add_earnings_transaction(self, session: AsyncSession, account: Account, amount: float, txn_type: EarningsTxnType):
        if txn_type == EarningsTxnType.Subtract:
            if account.earnings_balance < amount:
                raise ValueError("not enough earnings to subtract")
            account.earnings_balance -= amount
        else:
            account.earnings_balance += amount

        new_earnings_txn = EarningsTransaction(
            account_id=account.id,
            user_id=account.user_id,
            amount=amount,
            txn_type=txn_type,
        )
        session.add(new_earnings_txn)

    async def add_platform_revenue(self, session: AsyncSession, amount: float, txn_type: PlatformRevenueTxnType):
        new_rev = PlatformRevenue(
            amount=amount,
            txn_type=txn_type
        )
        session.add(new_rev)

    async def select_hold_for_order(self, session: AsyncSession, account: Account, subnet: Subnet, order_type: OrderType, price: float, specified_hold_id: Optional[int] = None):
        required_expiry_buffer = timedelta(days=DISPUTE_PAYOUT_DELAY_DAYS)
        min_expiry = datetime.utcnow() + timedelta(seconds=subnet.solve_period + subnet.dispute_period) + required_expiry_buffer

        if order_type == OrderType.Bid:
            required_amount = price
        else:
            required_amount = subnet.stake_multiplier * price

        if specified_hold_id is not None:
            stmt = select(Hold).where(Hold.id == specified_hold_id, Hold.account_id == account.id)
            result = await session.execute(stmt)
            hold = result.scalar_one_or_none()
            if not hold:
                raise ValueError("specified hold not found or not yours")
            if hold.charged:
                raise ValueError("this hold is already charged, cannot use")
            if hold.expiry < min_expiry:
                raise ValueError("hold expires too soon")
            if (hold.total_amount - hold.used_amount) < required_amount:
                raise ValueError("not enough hold amount")
            return hold
        else:
            stmt = select(Hold).where(
                Hold.account_id == account.id,
                Hold.charged == False,
                Hold.expiry > min_expiry,
                (Hold.total_amount - Hold.used_amount) >= required_amount
            )
            result = await session.execute(stmt)
            hold = result.scalars().first()
            if hold:
                return hold

            if account.credits_balance >= required_amount:
                new_hold = Hold(
                    account_id=account.id,
                    user_id=account.user_id,
                    hold_type=HoldType.Credits,
                    total_amount=required_amount,
                    used_amount=0.0,
                    expiry=min_expiry,
                    charged=False,
                    charged_amount=0.0
                )
                session.add(new_hold)
                account.credits_balance -= required_amount
                await session.flush()
                return new_hold

            raise ValueError("no suitable hold found, please specify a hold or create one")

    async def reserve_funds_on_hold(self, session: AsyncSession, hold: Hold, amount: float, order: Order):
        if hold.total_amount - hold.used_amount < amount:
            raise ValueError("not enough hold funds")
        hold.used_amount += amount
        hold_txn = HoldTransaction(
            hold_id=hold.id,
            order_id=order.id,
            amount=amount
        )
        session.add(hold_txn)

    async def free_funds_from_hold(self, session: AsyncSession, hold: Hold, amount: float, order: Order):
        if hold.used_amount < amount:
            raise ValueError("not enough used amount in hold to free")
        hold.used_amount -= amount
        hold_txn = HoldTransaction(
            hold_id=hold.id,
            order_id=order.id,
            amount=-amount
        )
        session.add(hold_txn)

    async def charge_hold_fully(self, session: AsyncSession, hold: Hold):
        if hold.charged:
            raise ValueError("hold already charged")
        hold.charged = True
        hold.charged_amount = hold.total_amount

        leftover = hold.total_amount - hold.used_amount
        if leftover > 0:
            stmt = select(Account).where(Account.id == hold.account_id)
            result = await session.execute(stmt)
            account = result.scalar_one_or_none()
            if account:
                await self.add_credits_transaction(session, account, leftover, CreditTxnType.Add)

    async def maybe_payout_earnings(self, session: AsyncSession, account: Account):
        if account.earnings_balance >= MINIMUM_PAYOUT_AMOUNT:
            payout_amount = account.earnings_balance
            account.earnings_balance -= payout_amount
            logger.debug(f"Payout {payout_amount} to {account.user_id}")

    async def handle_dispute_scenario(self, session: AsyncSession, task: Task):
        ask_order = task.ask
        if not ask_order or not ask_order.hold:
            raise ValueError("No ask order or hold for dispute scenario")

        hold = ask_order.hold
        if not hold.charged:
            await self.charge_hold_fully(session, hold)

        stake_amount = task.job.subnet.stake_multiplier * ask_order.price
        if stake_amount > hold.used_amount:
            raise ValueError("data inconsistency in stake vs hold used")

        await self.add_platform_revenue(session, stake_amount, PlatformRevenueTxnType.Add)

        bid_order = task.bid
        if bid_order and bid_order.hold:
            await self.free_funds_from_hold(session, bid_order.hold, bid_order.price, bid_order)

    async def handle_correct_resolution_scenario(self, session: AsyncSession, task: Task):
        bid_order = task.bid
        ask_order = task.ask
        if not bid_order or not bid_order.hold:
            raise ValueError("missing bid order/hold")
        if not ask_order or not ask_order.hold:
            raise ValueError("missing ask order/hold")

        if not bid_order.hold.charged:
            await self.charge_hold_fully(session, bid_order.hold)
        if not ask_order.hold.charged:
            await self.charge_hold_fully(session, ask_order.hold)

        subnet = task.job.subnet
        stake_amount = subnet.stake_multiplier * ask_order.price
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

        await self.maybe_payout_earnings(session, solver_account)

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
            if not job or job.user_id != get_user_id():
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

    async def admin_deposit_account(self, user_id: str, amount: float):
        async with AsyncSessionLocal() as session:
            account = await self.get_or_create_account(user_id, session)
            await self.add_credits_transaction(session, account, amount, CreditTxnType.Add)
            await session.commit()

    async def create_order(self, task_id: int | None, subnet_id: int, order_type: OrderType, price: float, hold_id: Optional[int] = None):
        async with AsyncSessionLocal() as session:
            user_id = get_user_id()
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

            await self.match_bid_ask_orders(session, subnet_id)
            await session.commit()
            return new_order.id

    async def get_num_orders(self, subnet_id: int, order_type: OrderType):
        async with AsyncSessionLocal() as session:
            user_id = get_user_id()
            stmt = select(Order).filter_by(subnet_id=subnet_id, user_id=user_id, order_type=order_type)
            result = await session.execute(stmt)
            orders = result.scalars().all()
            return {'num_orders': len(orders)}

    async def create_bids_and_tasks(self, job_id: int, num_tasks: int, price: float, params: str, hold_id: Optional[int]):
        async with AsyncSessionLocal() as session:
            stmt = select(Job).filter_by(id=job_id).options(joinedload(Job.subnet))
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if not job:
                raise ValueError(f"Job {job_id} not found.")

            user_id = job.user_id
            account = await self.get_or_create_account(user_id, session)

            created_items = []
            for i in range(num_tasks):
                job.iteration += 1
                new_task = Task(
                    job_id=job_id,
                    job_iteration=job.iteration,
                    status=TaskStatus.SelectingSolver,
                    params=params
                )
                session.add(new_task)
                await session.flush()

                hold = await self.select_hold_for_order(session, account, job.subnet, OrderType.Bid, price, hold_id)
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

            await self.match_bid_ask_orders(session, job.subnet_id)
            await session.commit()
            return {'created_items': created_items}

    async def delete_order(self, order_id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Order).filter_by(id=order_id)
            result = await session.execute(stmt)
            order = result.scalar_one_or_none()
            if not order or order.user_id != get_user_id():
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
            await session.commit()

    async def admin_create_account_key(self, user_id: str):
        return await self._create_account_key(user_id)

    async def create_account_key(self):
        return await self._create_account_key(get_user_id())

    async def _create_account_key(self, user_id: str):
        account = EthAccount.create()
        private_key = account.key.hex()
        public_key = account.address.lower()

        async with AsyncSessionLocal() as session:
            new_account_key = AccountKey(
                user_id=user_id,
                public_key=public_key
            )
            session.add(new_account_key)
            await session.commit()
            await session.refresh(new_account_key)

        return {
            "private_key": private_key,
            "public_key": public_key,
            "account_key_id": new_account_key.id
        }

    async def account_key_from_public_key(self, public_key: str):
        async with AsyncSessionLocal() as session:
            stmt = select(AccountKey).filter_by(public_key=public_key.lower())
            result = await session.execute(stmt)
            account_key = result.scalar_one_or_none()
            return account_key

    async def get_account_keys(self):
        async with AsyncSessionLocal() as session:
            user_id = get_user_id()
            stmt = select(AccountKey).filter_by(user_id=user_id)
            result = await session.execute(stmt)
            account_keys = result.scalars().all()
            return account_keys

    async def delete_account_key(self, account_key_id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(AccountKey).filter_by(id=account_key_id)
            result = await session.execute(stmt)
            account_key = result.scalar_one_or_none()
            if not account_key or account_key.user_id != get_user_id():
                raise PermissionError("No access to delete this account key.")
            await session.delete(account_key)
            await session.commit()

    async def match_bid_ask_orders(self, session: AsyncSession, subnet_id: int):
        bid_orders_stmt = (
            select(Order)
            .join(Task, Task.id == Order.bid_task_id)
            .filter(
                Order.subnet_id == subnet_id,
                Order.order_type == OrderType.Bid,
                Task.ask == None
            )
        )

        ask_orders_stmt = (
            select(Order)
            .filter(
                Order.subnet_id == subnet_id,
                Order.order_type == OrderType.Ask,
                Order.ask_task_id == None
            )
        )

        bid_orders_result = await session.execute(bid_orders_stmt)
        ask_orders_result = await session.execute(ask_orders_stmt)

        bid_orders = bid_orders_result.scalars().all()
        ask_orders = ask_orders_result.scalars().all()

        matches = []
        for bid in bid_orders:
            for ask in ask_orders:
                if bid.price >= ask.price:
                    bid_task = bid.bid_task
                    bid_task.ask = ask
                    bid_task.status = TaskStatus.SolverSelected
                    bid_task.time_solver_selected = datetime.utcnow()
                    session.add(bid_task)

                    ask_orders.remove(ask)
                    matches.append((bid, ask))
                    break

        return matches

    async def create_job(self, name: str, plugin_id: int, subnet_id: int, sot_url: str, iteration: int):
        async with AsyncSessionLocal() as session:
            new_job = Job(
                name=name,
                plugin_id=plugin_id,
                subnet_id=subnet_id,
                user_id=get_user_id(),
                sot_url=sot_url,
                iteration=iteration,
                done=False,
                last_updated=datetime.utcnow(),
                submitted_at=datetime.utcnow()
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

    async def should_dispute(self, task):
        return False

    async def resolve_task(self, session: AsyncSession, task: Task, result: str, status: TaskStatus):
        task.result = result
        task.status = status
        task.time_solved = datetime.now(timezone.utc)
        session.add(task)

        should_dispute = await self.should_dispute(task)
        if should_dispute:
            await self.handle_dispute_scenario(session, task)
        else:
            await self.handle_correct_resolution_scenario(session, task)

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

            logging.info(f'task_id: {task_id}')
            if not task.ask:
                logging.info(f"Task ID {task_id} has no associated Ask Order.")
                raise PermissionError("No access to submit result")

            if task.ask.user_id != get_user_id():
                logging.info(f"Task.ask.user_id: {task.ask.user_id}, get_user_id(): {get_user_id()}")
                raise PermissionError("No access to submit result")

            if task.status != TaskStatus.SolverSelected:
                raise ValueError("Task not in SolverSelected status")

            await self.resolve_task(session, task, result, TaskStatus.ResolvedCorrect)
            await session.commit()

    async def create_state_update(self, job_id: int, data: Dict):
        async with AsyncSessionLocal() as session:
            new_state_update = StateUpdate(
                job_id=job_id,
                data=data
            )
            session.add(new_state_update)
            await session.commit()
            await session.refresh(new_state_update)
            return new_state_update.id

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

    async def get_task(self, task_id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Task).options(joinedload(Task.job)).filter(Task.id == task_id)
            result = await session.execute(stmt)
            task = result.scalar_one_or_none()
            return task

    async def get_assigned_tasks(self, subnet_id: int):
        async with AsyncSessionLocal() as session:
            user_id = get_user_id()
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
            stmt = select(Task).filter_by(job_id=job_id).order_by(Task.submitted_at.asc()).offset(offset).limit(limit)
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
            
            # Fetch the updated Perm object
            updated_perm = await session.execute(
                select(Perm).where(Perm.address == address, Perm.perm == perm)
            )
            perm_obj = updated_perm.scalar_one_or_none()
            return perm_obj.id

    async def get_sot(self, id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Sot).filter_by(id=id)
            result = await session.execute(stmt)
            sot = result.scalar_one_or_none()
            return sot

    async def create_perm(self, address: str, perm: int):
        async with AsyncSessionLocal() as session:
            new_perm = Perm(
                address=address,
                perm=perm
            )
            session.add(new_perm)
            await session.commit()
            await session.refresh(new_perm)
            return new_perm.id

    async def create_perm_description(self, perm_type: PermType):
        async with AsyncSessionLocal() as session:
            new_perm_description = PermDescription(
                perm_type=perm_type
            )
            session.add(new_perm_description)
            await session.commit()
            await session.refresh(new_perm_description)
            return new_perm_description.id

    async def create_sot(self, job_id: int, url: str | None):
        async with AsyncSessionLocal() as session:
            perm_id = await self.create_perm_description(perm_type=PermType.ModifySot)
            new_sot = Sot(
                job_id=job_id,
                perm=perm_id,
                url=url
            )
            session.add(new_sot)
            await session.commit()
            await session.refresh(new_sot)
            return new_sot.id

    async def update_sot(self, sot_id: int, url: str | None):
        async with AsyncSessionLocal() as session:
            stmt = update(Sot).where(Sot.id == sot_id).values(url=url)
            await session.execute(stmt)
            await session.commit()
            return True

    async def get_sot_by_job_id(self, job_id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(Sot).filter_by(job_id=job_id)
            result = await session.execute(stmt)
            sot = result.scalar_one_or_none()
            return sot

    async def get_total_state_updates_for_job(self, job_id: int):
        async with AsyncSessionLocal() as session:
            stmt = select(func.count(StateUpdate.id)).filter_by(job_id=job_id)
            result = await session.execute(stmt)
            total_state_updates = result.scalar_one()
            return total_state_updates

    async def get_last_task_with_status(self, job_id: int, statuses: list[TaskStatus]):
        async with AsyncSessionLocal() as session:
            stmt = select(Task).filter(
                Task.job_id == job_id,
                Task.status.in_(statuses)
            ).order_by(desc(Task.submitted_at)).limit(1)
            result = await session.execute(stmt)
            task = result.scalar_one_or_none()
            return task

    async def create_instance(self, name: str, service_type: ServiceType, job_id: Optional[int], private_key: str | None, pod_id: str | None, process_id: int | None):
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

    async def get_instance_by_service_type(self, service_type: ServiceType, job_id: Optional[int] = None):
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

    async def get_jobs_without_instances(self):
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

db_adapter_server = DBAdapterServer()
