import pytest
import pytest_asyncio
from unittest.mock import patch
from sqlalchemy.orm import joinedload
from sqlalchemy import select

from spl.models import AsyncSessionLocal, Task, Job, Order
from spl.db.server.adapter.orders_tasks import DBAdapterOrdersTasksMixin
from spl.db.server.app import original_app


@pytest_asyncio.fixture(autouse=True)
async def clear_database():
    """
    Clear the database before each test to ensure test isolation.
    """
    async with AsyncSessionLocal() as session:
        from spl.models import Base
        for table in reversed(Base.metadata.sorted_tables):
            await session.execute(table.delete())
        await session.commit()
    yield


@pytest.fixture(autouse=True)
def patch_get_task_eager():
    """
    Patch DBAdapterOrdersTasksMixin.get_task so it uses eager loading.
    This avoids MissingGreenlet issues if the session is closed too soon.
    """
    original_get_task = DBAdapterOrdersTasksMixin.get_task

    async def get_task_eager(self, task_id: int):
        async with AsyncSessionLocal() as session:
            stmt = (
                select(Task)
                .options(
                    joinedload(Task.bid).joinedload(Order.hold),
                    joinedload(Task.ask).joinedload(Order.hold),
                    joinedload(Task.job).joinedload(Job.subnet),
                )
                .where(Task.id == task_id)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    with patch.object(DBAdapterOrdersTasksMixin, 'get_task', get_task_eager):
        yield
