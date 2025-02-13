# file: spl/db/server/adapter/global.py

from sqlalchemy import select, func
from ....models import Order, Task, OrderType, TaskStatus

class DBAdapterGlobalMixin:
    async def get_global_stats(self) -> dict:
        """
        Returns a dictionary of various global statistics:
          - total_open_orders_dollar: sum of prices for unmatched orders
          - num_open_orders: count of unmatched orders
          - num_completed_tasks: count of tasks with a completed status
          - volume: total bid price volume for completed matched bids
        """
        completed_statuses = [TaskStatus.ResolvedCorrect, TaskStatus.ResolvedIncorrect]

        async with self.get_async_session() as session:
            # Sum of prices in unmatched orders
            stmt_sum_open = select(func.sum(Order.price)).where(
                Order.bid_task_id.is_(None),
                Order.ask_task_id.is_(None)
            )
            sum_open_result = await session.execute(stmt_sum_open)
            total_open_orders_dollar = sum_open_result.scalar() or 0.0

            # Count of unmatched orders
            stmt_count_open = select(func.count(Order.id)).where(
                Order.bid_task_id.is_(None),
                Order.ask_task_id.is_(None)
            )
            count_open_result = await session.execute(stmt_count_open)
            num_open_orders = count_open_result.scalar() or 0

            # Count of tasks that are completed
            stmt_count_completed = select(func.count(Task.id)).where(
                Task.status.in_(completed_statuses)
            )
            count_completed_result = await session.execute(stmt_count_completed)
            num_completed_tasks = count_completed_result.scalar() or 0

            # Volume for completed matched bids (summing bid prices)
            stmt_volume = (
                select(func.sum(Order.price))
                .join(Task, Order.bid_task_id == Task.id)
                .where(
                    Order.order_type == OrderType.Bid,
                    Order.ask_task_id.isnot(None),
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
