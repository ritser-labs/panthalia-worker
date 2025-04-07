# spl/worker/shutdown_flag.py
import asyncio

_shutdown_requested = False
_lock = asyncio.Lock()

async def is_shutdown_requested() -> bool:
    async with _lock:
        return _shutdown_requested

def set_shutdown_requested(value: bool) -> None:
    global _shutdown_requested
    _shutdown_requested = value
