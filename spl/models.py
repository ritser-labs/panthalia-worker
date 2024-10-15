from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import Column, Integer, String, ForeignKey, Text, UniqueConstraint, DateTime, Enum, Float, Boolean, JSON
from datetime import datetime
from .common import TaskStatus
import os
import asyncio
import enum

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Compute the parent directory
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

# Construct the database path relative to the parent directory
db_path = os.path.join(parent_dir, "sqlite.db")

# Modify DATABASE_URL to use the computed db_path
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

class Serializable(Base):
    __abstract__ = True

    def as_dict(self):
        result = {}
        for c in self.__table__.columns:
            value = getattr(self, c.name)
            # If the value is an Enum, serialize it as its name
            if isinstance(value, enum.Enum):
                result[c.name] = value.name
            else:
                result[c.name] = value
        return result


class PermType(enum.Enum):
    ModifyDb = 0
    ModifySot = 1

class TimestampMixin:
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    submitted_at = Column(DateTime, nullable=False, default=datetime.utcnow)

class Plugin(TimestampMixin, Serializable):
    __tablename__ = 'plugins'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    code = Column(Text, nullable=False)

class Subnet(Serializable):
    __tablename__ = 'subnets'
    id = Column(Integer, primary_key=True, index=True)
    address = Column(String, nullable=False)
    rpc_url = Column(String, nullable=False)
class Job(TimestampMixin, Serializable):
    __tablename__ = 'jobs'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    plugin_id = Column(Integer, ForeignKey('plugins.id'), nullable=False)
    subnet_id = Column(Integer, ForeignKey('subnets.id'), nullable=False)
    sot_url = Column(String, nullable=False)
    done = Column(Boolean, nullable=False, default=False)
    iteration = Column(Integer, nullable=False)

    plugin = relationship("Plugin", backref="jobs")
    subnet = relationship("Subnet", backref="jobs")

class Task(TimestampMixin, Serializable):
    __tablename__ = 'tasks'
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey('jobs.id'), nullable=False)
    subnet_task_id = Column(Integer, nullable=False)
    job_iteration = Column(Integer, nullable=False)
    status = Column(Enum(TaskStatus), nullable=False)
    result = Column(JSON, nullable=True)
    
    __table_args__ = (UniqueConstraint('job_id', 'subnet_task_id', name='_job_task_uc'),)

    job = relationship("Job", backref="tasks")

class StateUpdate(Serializable):
    __tablename__ = 'state_updates'
    id = Column(Integer, primary_key=True, index=True)
    state_iteration = Column(Integer, nullable=False)
    job_id = Column(Integer, ForeignKey('jobs.id'), nullable=False)
    data = Column(JSON, nullable=False)

    job = relationship("Job", backref="state_updates")

class Perm(Serializable):
    __tablename__ = 'perms'
    id = Column(Integer, primary_key=True, index=True)
    address = Column(String, nullable=False, index=True)
    perm = Column(Integer, ForeignKey('perm_descriptions.id'), nullable=False)
    last_nonce = Column(String, nullable=False, default=0)

class Sot(Serializable):
    __tablename__ = 'sots'
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey('jobs.id'), nullable=False)
    perm = Column(Integer, ForeignKey('perm_descriptions.id'), nullable=False)
    url = Column(String, nullable=False)

    job = relationship("Job", backref="sots")

class PermDescription(Serializable):
    __tablename__ = 'perm_descriptions'
    id = Column(Integer, primary_key=True, index=True)
    perm_type = Column(Enum(PermType), nullable=False)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Run the initialization and an example operation
async def main():
    await init_db()  # Create tables

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())