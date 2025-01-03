# file: spl/db/init.py

import os
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from spl.models.base import Base  # or wherever Base is defined

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
db_path = os.path.join(parent_dir, 'sqlite.db')
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
engine = create_async_engine(DATABASE_URL, echo=False)

class CheckingInvariantAsyncSession(AsyncSession):
    """
    Custom AsyncSession that can run check_invariant() before flush & after commit.
    If you need `db_adapter_server`, pass it in at session-creation-time 
    so there's no import from spl.db.server.adapter here.
    """
    async def flush(self, *args, **kwargs):
        # Possibly call self._db_adapter_server.check_invariant() if you have it
        return await super().flush(*args, **kwargs)

    async def commit(self):
        result = await super().commit()
        # Possibly call invariant check here again
        return result

async def init_db():
    """
    Creates tables, etc. Called once at startup or in tests.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

def create_invariant_async_session(db_adapter_server=None):
    """
    Returns a sessionmaker that yields CheckingInvariantAsyncSession objects,
    optionally embedding the db_adapter_server reference if needed.
    """
    # We define a subclass to embed db_adapter_server if needed:
    class CustomInvariantSession(CheckingInvariantAsyncSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._db_adapter_server = db_adapter_server

        async def flush(self, *args, **kwargs):
            # if needed, call self._db_adapter_server.check_invariant() here
            return await super().flush(*args, **kwargs)

        async def commit(self):
            result = await super().commit()
            # if needed, call self._db_adapter_server.check_invariant() again
            return result

    return sessionmaker(
        bind=engine,
        class_=CustomInvariantSession,
        expire_on_commit=False,
        autoflush=True,
        autocommit=False,
    )

AsyncSessionLocal = create_invariant_async_session()