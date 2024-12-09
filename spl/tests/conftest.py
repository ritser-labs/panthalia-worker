import pytest
import pytest_asyncio
from unittest.mock import patch

from spl.models import init_db
from spl.db.server.adapter import db_adapter_server

import os
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_test_database():
    await init_db()
    yield

# Removed the app_context_fixture entirely, as we now rely on per-test contexts only.

@pytest.fixture(autouse=True)
def mock_user_id():
    with patch("spl.db.server.adapter.get_user_id", return_value="testuser"):
        yield

@pytest.fixture(scope='session', autouse=True)
def mock_auth_decorators():
    with patch("spl.db.server.routes.requires_user_auth_with_adapter", lambda f: f), \
         patch("spl.db.server.routes.requires_auth", lambda f: f):
        yield

@pytest.fixture(scope='session', autouse=True)
def patch_api_auth():
    with patch("spl.auth.api_auth.requires_authentication", lambda get_db_adapter, get_perm_db: lambda f: f), \
         patch("spl.auth.server_auth.requires_user_auth", lambda get_db_adapter: lambda f: f):
        yield

@pytest_asyncio.fixture
async def db_adapter_server_fixture():
    # Just return the global db_adapter_server
    return db_adapter_server
