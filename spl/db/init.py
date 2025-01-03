# file: spl/db/init.py

import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import logging
from spl.models.base import Base
import traceback

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

class InvariantError(SQLAlchemyError):
    """Raised when the ledger invariant fails for debugging stack traces."""
    pass

def create_invariant_async_session(db_adapter_server=None):
    class CustomInvariantSession(CheckingInvariantAsyncSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._db_adapter_server = db_adapter_server

            # Save the call stack right now, so we know which caller created the session
            self._session_creation_stack = traceback.format_stack()

        async def commit(self):
            await self.flush()

            if self._db_adapter_server:
                logging.debug("CustomInvariantSession.commit => checking invariant")
                try:
                    invariant_result = await self._db_adapter_server.check_invariant(self)
                    if not invariant_result.get("invariant_holds", True):
                        logging.error("Invariant check failed with result: %s", invariant_result)
                        # Here we embed the session creation stack directly
                        raise InvariantError(
                            "Invariant check failed before commit:\n"
                            f"{invariant_result}\n\n"
                            f"--- Session was originally created at:\n"
                            + "".join(self._session_creation_stack)
                        )
                except InvariantError:
                    stack_trace = traceback.format_exc()
                    logging.error("Traceback for invariant failure:\n%s", stack_trace)
                    # Also log how the session was created
                    logging.error("Session creation stack:\n%s", "".join(self._session_creation_stack))
                    await self.rollback()
                    raise
                except Exception:
                    logging.exception("Unexpected error while checking invariants!")
                    raise

            return await super().commit()

    return sessionmaker(
        bind=engine,
        class_=CustomInvariantSession,
        expire_on_commit=False,
        autoflush=True,
        autocommit=False,
    )
