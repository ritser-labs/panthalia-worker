# models.py

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import Column, Integer, String, ForeignKey, Text, UniqueConstraint, DateTime, Enum, Float, Boolean, JSON
from datetime import datetime
from .common import TaskStatus
import os
import asyncio
import enum

# Database setup
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
db_path = os.path.join(parent_dir, "sqlite.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# Enum definitions
class ServiceType(enum.Enum):
    Anvil = "anvil"
    Db = "db"
    Master = "master"
    Sot = "sot"
    Worker = "worker"

class PermType(enum.Enum):
    ModifyDb = 0
    ModifySot = 1

class OrderType(enum.Enum):
    Bid = "bid"
    Ask = "ask"

class AccountTxnType(enum.Enum):
    Deposit = "deposit"
    Withdrawal = "withdrawal"

# Mixins and base classes
class Serializable(Base):
    __abstract__ = True
    def as_dict(self):
        result = {}
        for c in self.__table__.columns:
            value = getattr(self, c.name)
            result[c.name] = value.name if isinstance(value, enum.Enum) else value
        return result

class TimestampMixin:
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    submitted_at = Column(DateTime, nullable=False, default=datetime.utcnow)

# Models
class Plugin(TimestampMixin, Serializable):
    __tablename__ = 'plugins'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    code = Column(Text, nullable=False)
    jobs = relationship("Job", back_populates="plugin")

class Subnet(Serializable):
    __tablename__ = 'subnets'
    id = Column(Integer, primary_key=True, index=True)
    dispute_period = Column(Integer, nullable=False)
    solve_period = Column(Integer, nullable=False)
    stake_multiplier = Column(Float, nullable=False)
    jobs = relationship("Job", back_populates="subnet")
    orders = relationship("Order", back_populates="subnet")

class Job(TimestampMixin, Serializable):
    __tablename__ = 'jobs'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    plugin_id = Column(Integer, ForeignKey('plugins.id'), nullable=False)
    subnet_id = Column(Integer, ForeignKey('subnets.id'), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    sot_url = Column(String, nullable=False)
    done = Column(Boolean, nullable=False, default=False)
    iteration = Column(Integer, nullable=False)

    plugin = relationship("Plugin", back_populates="jobs")
    subnet = relationship("Subnet", back_populates="jobs")
    instances = relationship("Instance", back_populates="job")
    tasks = relationship("Task", back_populates="job")
    state_updates = relationship("StateUpdate", back_populates="job")
    sots = relationship("Sot", back_populates="job")

class Task(TimestampMixin, Serializable):
    __tablename__ = 'tasks'
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey('jobs.id'), nullable=False)
    job_iteration = Column(Integer, nullable=False)
    status = Column(Enum(TaskStatus), nullable=False)
    params = Column(String, nullable=False)
    result = Column(JSON, nullable=True)
    time_solved = Column(DateTime, nullable=True)
    time_solver_selected = Column(DateTime, nullable=True)

    job = relationship("Job", back_populates="tasks")
    bid = relationship(
        "Order",
        primaryjoin="and_(Task.id == Order.bid_task_id, Order.order_type == 'Bid')",
        uselist=False,
        back_populates="bid_task"
    )
    ask = relationship(
        "Order",
        primaryjoin="and_(Task.id == Order.ask_task_id, Order.order_type == 'Ask')",
        uselist=False,
        back_populates="ask_task"
    )
    account = relationship("Account", back_populates="task")


class StateUpdate(Serializable):
    __tablename__ = 'state_updates'
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey('jobs.id'), nullable=False)
    data = Column(JSON, nullable=False)
    submitted_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    job = relationship("Job", back_populates="state_updates")

class Perm(Serializable):
    __tablename__ = 'perms'
    id = Column(Integer, primary_key=True, index=True)
    address = Column(String, nullable=False, index=True)
    perm = Column(Integer, ForeignKey('perm_descriptions.id'), nullable=False)
    last_nonce = Column(String, nullable=False, default='0')

class Sot(Serializable):
    __tablename__ = 'sots'
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey('jobs.id'), nullable=False)
    perm = Column(Integer, ForeignKey('perm_descriptions.id'), nullable=False)
    url = Column(String)
    job = relationship("Job", back_populates="sots")

class PermDescription(Serializable):
    __tablename__ = 'perm_descriptions'
    id = Column(Integer, primary_key=True, index=True)
    perm_type = Column(Enum(PermType), nullable=False)

class Instance(Serializable):
    __tablename__ = 'instances'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    service_type = Column(Enum(ServiceType), nullable=False)
    job_id = Column(Integer, ForeignKey('jobs.id'), nullable=True)
    private_key = Column(String)
    pod_id = Column(String)
    process_id = Column(Integer)
    job = relationship("Job", back_populates="instances")

class Order(Serializable):
    __tablename__ = 'orders'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    order_type = Column(Enum(OrderType), nullable=False)  # Distinguishes between Bid and Ask
    price = Column(Float, nullable=False, index=True)
    subnet_id = Column(Integer, ForeignKey('subnets.id'), nullable=False)
    bid_task_id = Column(Integer, ForeignKey('tasks.id'), nullable=True)
    ask_task_id = Column(Integer, ForeignKey('tasks.id'), nullable=True)
    submitted_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=False)

    subnet = relationship("Subnet", back_populates="orders")
    bid_task = relationship("Task", back_populates="bid", foreign_keys=[bid_task_id])
    ask_task = relationship("Task", back_populates="ask", foreign_keys=[ask_task_id])
    account = relationship("Account", back_populates="orders", uselist=False)



class Account(Serializable):
    __tablename__ = 'accounts'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    amount = Column(Float, nullable=False)
    available = Column(Float, nullable=False)
    current_task_id = Column(Integer, ForeignKey('tasks.id'), nullable=True)
    deposited_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    task = relationship("Task", back_populates="account")
    orders = relationship("Order", back_populates="account", uselist=True)
    transactions = relationship("AccountTransaction", back_populates="account")

class AccountTransaction(Serializable):
    __tablename__ = 'account_transactions'
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    amount = Column(Float, nullable=False)
    transaction_type = Column(Enum(AccountTxnType), nullable=False)  # "deposit" or "withdrawal"
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    account = relationship("Account", back_populates="transactions")

class AccountKey(Serializable):
    __tablename__ = 'account_keys'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    public_key = Column(String, nullable=False, index=True, unique=True)

# Database initialization
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Main function to run the initialization
async def main():
    await init_db()

if __name__ == "__main__":
    asyncio.run(main())
