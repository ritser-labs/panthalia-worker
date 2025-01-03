import pytest
import pytest_asyncio
from unittest.mock import patch

from spl.db.init import engine, init_db
from spl.db.server.adapter import DBAdapterServer

import os
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"  # in-memory DB

from spl.models import Base
from sqlalchemy.orm import joinedload
from sqlalchemy import select
from spl.db.server.adapter.orders_tasks import DBAdapterOrdersTasksMixin
from spl.models import Task, Job, Order


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_test_database():
    """
    Creates an in-memory test DB once per session, runs migrations, etc.
    """
    await init_db()
    yield


@pytest_asyncio.fixture(scope="function", autouse=True)
async def fresh_db():
    """
    For every single test function, we:
      1) Drop all tables
      2) Create all tables fresh
    so there's zero leftover from previous tests.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    # (Optional) you could drop again after the test
    # but typically we only need to do it before

@pytest.fixture(scope='session', autouse=True)
def mock_auth_decorators():
    """
    Globally disable or patch out real auth decorators so tests can run without credentials.
    """
    with patch("spl.db.server.routes.requires_user_auth_with_adapter", lambda f: f), \
         patch("spl.db.server.routes.requires_auth", lambda f: f):
        yield


@pytest.fixture(scope='session', autouse=True)
def patch_api_auth():
    """
    Disable actual authentication for all tests so we don't need real JWT/Eth keys.
    """
    with patch("spl.auth.api_auth.requires_authentication", lambda get_db_adapter, get_perm_db: lambda f: f), \
         patch("spl.auth.server_auth.requires_user_auth", lambda get_db_adapter: lambda f: f):
        yield


@pytest.fixture
def db_adapter_server_fixture():
    """Use a DBAdapterServer but override user_id_getter => 'testuser' by default."""
    server = DBAdapterServer(user_id_getter=lambda: "testuser")
    return server


@pytest.fixture(autouse=True)
def patch_get_task_eager():
    """
    Patch DBAdapterOrdersTasksMixin.get_task so it uses eager-loading.
    Avoids MissingGreenlet issues if the session is closed too soon.
    """
    original_get_task = DBAdapterOrdersTasksMixin.get_task

    async def get_task_eager(self, task_id: int):
        async with DBAdapterServer().get_async_session() as session:
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
