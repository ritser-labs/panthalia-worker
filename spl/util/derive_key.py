# file: spl/util/derive_key.py

from eth_account import Account
from eth_keys.constants import SECPK1_N
import hashlib

def derive_sot_key(master_private_key: str, job_id: int) -> dict:
    """
    Deterministically derives a new wallet from the master private key and a job id.
    """
    # Convert master private key from hex to an integer.
    master_int = int(master_private_key, 16)
    # Derive a new key by adding the job id and taking modulo SECPK1_N.
    new_key_int = (master_int + job_id) % SECPK1_N
    # Convert the integer back to a 64-character hexadecimal string.
    new_key_hex = format(new_key_int, 'x').zfill(64)
    new_account = Account.from_key(new_key_hex)
    return {"private_key": new_key_hex, "address": new_account.address}
