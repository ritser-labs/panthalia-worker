# spl/worker/order_state.py
import asyncio

_order_state = {}
_lock = asyncio.Lock()

async def mark_order_pending(order_id: int):
    async with _lock:
        _order_state[order_id] = "pending"

async def mark_order_processing(order_id: int):
    async with _lock:
        _order_state[order_id] = "processing"

async def mark_order_completed(order_id: int):
    async with _lock:
        _order_state[order_id] = "completed"

async def mark_order_cancelled(order_id: int):
    async with _lock:
        _order_state[order_id] = "cancelled"

async def get_pending_orders():
    async with _lock:
        # Return only orders that remain pending (i.e. unmatched)
        return {oid: state for oid, state in _order_state.items() if state == "pending"}

async def get_processing_orders():
    async with _lock:
        # Return orders that are either pending or processing.
        return {oid: state for oid, state in _order_state.items() if state in ("pending", "processing")}
