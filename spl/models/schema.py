# spl/models/schema.py

from datetime import datetime
from pytz import UTC
from sqlalchemy import (
    Column, Integer, String, ForeignKey, Text, DateTime, Enum, Float, Boolean, JSON
)
from sqlalchemy.orm import relationship
from .enums import (
    ServiceType, PermType, OrderType, AccountTxnType, HoldType,
    CreditTxnType, EarningsTxnType, PlatformRevenueTxnType, WithdrawalStatus,
    TaskStatus
)
from .base import Base
import enum
from ..common import device

class Serializable(Base):
    __abstract__ = True
    def as_dict(self):
        result = {}
        for c in self.__table__.columns:
            value = getattr(self, c.name)
            if isinstance(value, enum.Enum):
                value = value.name
            result[c.name] = value
        return result

class TimestampMixin:
    last_updated = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))
    submitted_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

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
    master_state_json = Column(JSON, nullable=True, default={})
    sot_state_json = Column(JSON, nullable=True, default={})

    # Add an 'active' column
    active = Column(Boolean, nullable=False, default=True)

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
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=True)
    job = relationship("Job", back_populates="tasks")
    bid = relationship(
        "Order",
        primaryjoin="and_(Task.id == Order.bid_task_id)",
        uselist=False,
        back_populates="bid_task"
    )
    ask = relationship(
        "Order",
        primaryjoin="and_(Task.id == Order.ask_task_id)",
        uselist=False,
        back_populates="ask_task"
    )
    account = relationship("Account", back_populates="tasks")

class StateUpdate(Serializable):
    __tablename__ = 'state_updates'
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey('jobs.id'), nullable=False)
    data = Column(JSON, nullable=False)
    submitted_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
    job = relationship("Job", back_populates="state_updates")

class PermDescription(Serializable):
    __tablename__ = 'perm_descriptions'
    id = Column(Integer, primary_key=True, index=True)
    perm_type = Column(Enum(PermType), nullable=False)
    perms = relationship("Perm", back_populates="perm_description")

class Perm(Serializable):
    __tablename__ = 'perms'
    id = Column(Integer, primary_key=True, index=True)
    address = Column(String, nullable=False, index=True)
    perm = Column(Integer, ForeignKey('perm_descriptions.id'), nullable=False)
    last_nonce = Column(String, nullable=False, default='0')
    perm_description = relationship("PermDescription", back_populates="perms")

class Sot(Serializable):
    __tablename__ = 'sots'
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey('jobs.id'), nullable=False)
    perm = Column(Integer, ForeignKey('perm_descriptions.id'), nullable=False)
    url = Column(String)
    job = relationship("Job", back_populates="sots")

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

class Account(Serializable):
    __tablename__ = 'accounts'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    deposited_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    tasks = relationship("Task", back_populates="account")
    orders = relationship("Order", back_populates="account")
    transactions = relationship("AccountTransaction", back_populates="account")
    holds = relationship("Hold", back_populates="account")
    credit_transactions = relationship("CreditTransaction", back_populates="account")
    earnings_transactions = relationship("EarningsTransaction", back_populates="account")
    withdrawals = relationship("WithdrawalRequest", back_populates="account")

class AccountKey(Serializable):
    __tablename__ = 'account_keys'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    public_key = Column(String, nullable=False, index=True, unique=True)

class Order(Serializable):
    __tablename__ = 'orders'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    order_type = Column(Enum(OrderType), nullable=False)
    price = Column(Integer, nullable=False, index=True)
    subnet_id = Column(Integer, ForeignKey('subnets.id'), nullable=False)
    bid_task_id = Column(Integer, ForeignKey('tasks.id'), nullable=True)
    ask_task_id = Column(Integer, ForeignKey('tasks.id'), nullable=True)
    submitted_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC), index=True)
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=False)
    hold_id = Column(Integer, ForeignKey('holds.id'), nullable=True)

    subnet = relationship("Subnet", back_populates="orders")
    bid_task = relationship("Task", back_populates="bid", foreign_keys=[bid_task_id])
    ask_task = relationship("Task", back_populates="ask", foreign_keys=[ask_task_id])
    account = relationship("Account", back_populates="orders")
    hold = relationship("Hold", back_populates="orders")

class AccountTransaction(Serializable):
    __tablename__ = 'account_transactions'
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    amount = Column(Integer, nullable=False)
    transaction_type = Column(Enum(AccountTxnType), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    account = relationship("Account", back_populates="transactions")

class Hold(Serializable):
    __tablename__ = 'holds'
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    hold_type = Column(Enum(HoldType), nullable=False)
    total_amount = Column(Integer, nullable=False)
    used_amount = Column(Integer, nullable=False, default=0.0)
    expiry = Column(DateTime, nullable=False)
    charged = Column(Boolean, nullable=False, default=False)
    charged_amount = Column(Integer, nullable=False, default=0.0)

    account = relationship("Account", back_populates="holds")
    hold_transactions = relationship("HoldTransaction", back_populates="hold")
    orders = relationship("Order", back_populates="hold")

    parent_hold_id = Column(Integer, ForeignKey('holds.id'), nullable=True)
    parent_hold = relationship("Hold", remote_side=[id], backref="child_holds")

class HoldTransaction(Serializable):
    __tablename__ = 'hold_transactions'
    id = Column(Integer, primary_key=True, index=True)
    hold_id = Column(Integer, ForeignKey('holds.id'), nullable=False)
    order_id = Column(Integer, ForeignKey('orders.id'), nullable=True)
    amount = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
    hold = relationship("Hold", back_populates="hold_transactions")

class CreditTransaction(Serializable):
    __tablename__ = 'credit_transactions'
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    amount = Column(Integer, nullable=False)
    txn_type = Column(Enum(CreditTxnType), nullable=False)
    reason = Column(String, nullable=True)
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    account = relationship("Account", back_populates="credit_transactions")

class EarningsTransaction(Serializable):
    __tablename__ = 'earnings_transactions'
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    amount = Column(Integer, nullable=False)
    txn_type = Column(Enum(EarningsTxnType), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    account = relationship("Account", back_populates="earnings_transactions")

class PlatformRevenue(Serializable):
    __tablename__ = 'platform_revenue'
    id = Column(Integer, primary_key=True, index=True)
    amount = Column(Integer, nullable=False)
    txn_type = Column(Enum(PlatformRevenueTxnType), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

class WithdrawalRequest(Serializable):
    __tablename__ = 'pending_withdrawals'
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    amount = Column(Integer, nullable=False)
    status = Column(Enum(WithdrawalStatus), nullable=False, default=WithdrawalStatus.PENDING)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))
    payment_instructions = Column(Text, nullable=True)
    payment_record = Column(Text, nullable=True)
    rejection_reason = Column(Text, nullable=True)

    account = relationship("Account", back_populates="withdrawals")

class StripeDeposit(Base):
    __tablename__ = "stripe_deposits"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    deposit_amount = Column(Integer, nullable=False)
    stripe_session_id = Column(String, nullable=False, unique=True, index=True)
    status = Column(String, nullable=False, default="pending")
    is_authorization = Column(Boolean, nullable=False, default=False)

    credit_transaction_id = Column(Integer, ForeignKey("credit_transactions.id"), nullable=True)
    credit_transaction = relationship("CreditTransaction", backref="stripe_deposit")

    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))
