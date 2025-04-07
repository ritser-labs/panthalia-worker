# spl/models/enums.py

import enum

from ..common import TaskStatus

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

class HoldType(enum.Enum):
    CreditCard = "cc"
    Credits = "credits"
    Earnings = "earnings"

class CreditTxnType(enum.Enum):
    Add = "add"
    Subtract = "subtract"

class EarningsTxnType(enum.Enum):
    Add = "add"
    Subtract = "subtract"

class PlatformRevenueTxnType(enum.Enum):
    Add = "add"
    Subtract = "subtract"

##
# NEW WITHDRAWAL STATUS ENUM
##
class WithdrawalStatus(enum.Enum):
    PENDING = "pending"
    FINALIZED = "finalized"
    REJECTED = "rejected"

class SlotType(enum.Enum):
    SOT = "sot"
    WORKER = "worker"
    MASTER = "master"

class PluginReviewStatus(enum.Enum):
    Unreviewed = "unreviewed"
    Approved = "approved"
    Rejected = "rejected"
