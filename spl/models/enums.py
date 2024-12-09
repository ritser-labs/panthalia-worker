import enum

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

class CreditTxnType(enum.Enum):
    Add = "add"
    Subtract = "subtract"

class EarningsTxnType(enum.Enum):
    Add = "add"
    Subtract = "subtract"

class PlatformRevenueTxnType(enum.Enum):
    Add = "add"
    Subtract = "subtract"
