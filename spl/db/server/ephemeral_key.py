import logging
from eth_account import Account

logger = logging.getLogger(__name__)

_db_sot_keypair = None  # or store a small class with private_key, address

def generate_ephemeral_db_sot_key():
    global _db_sot_keypair
    acct = Account.create()
    _db_sot_keypair = {
        "private_key": acct.key.hex(),
        "address": acct.address.lower()
    }
    logger.info(f"[ephemeral_key] ephemeral DB-SOT address = {_db_sot_keypair['address']}")

def get_db_sot_private_key():
    return _db_sot_keypair["private_key"] if _db_sot_keypair else None

def get_db_sot_address():
    return _db_sot_keypair["address"] if _db_sot_keypair else None
