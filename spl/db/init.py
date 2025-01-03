# file: spl/db/init.py

import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import logging
from spl.models.base import Base

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
db_path = os.path.join(parent_dir, 'sqlite.db')

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
engine = create_async_engine(DATABASE_URL, echo=False)

class CheckingInvariantAsyncSession(AsyncSession):
    """
    Base AsyncSession that we can override for invariant checks.
    """
    async def flush(self, *args, **kwargs):
        return await super().flush(*args, **kwargs)

    async def commit(self):
        return await super().commit()

async def init_db():
    """
    Creates tables, etc. Called once at startup or in tests.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

def create_invariant_async_session(db_adapter_server=None):
    """
    Returns a sessionmaker for a custom session that calls
    db_adapter_server.check_invariant() before committing.
    """
    class CustomInvariantSession(CheckingInvariantAsyncSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._db_adapter_server = db_adapter_server

        async def commit(self):
            # Ensure flush is done first:
            await self.flush()
            if self._db_adapter_server:
                logging.debug("CustomInvariantSession.commit => checking invariant")
                invariant_result = await self._db_adapter_server.check_invariant()
                if not invariant_result.get("invariant_holds", True):
                    await self.rollback()
                    raise SQLAlchemyError(
                        f"Invariant check failed before commit: {invariant_result}"
                    )
            return await super().commit()

    return sessionmaker(
        bind=engine,
        class_=CustomInvariantSession,
        expire_on_commit=False,
        autoflush=True,
        autocommit=False,
    )
