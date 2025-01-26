# file: spl/auth/nonce_cache.py

from cachetools import TTLCache

cache = TTLCache(maxsize=100_000, ttl=300)  # e.g. store up to 100k items, each for 300s

def check_nonce(address: str, nonce: str) -> bool:
    key = f"{address}:{nonce}"
    if key in cache:
        return False  # replay
    cache[key] = True
    return True