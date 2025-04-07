# spl/worker/queued_lock.py

import itertools
import logging
import asyncio
import heapq

class AsyncQueuedLock:
    def __init__(self):
        self._queue = []             # Heap queue to order waiting tasks by priority.
        self._lock = asyncio.Lock()  # Protects access to the queue and internal flag.
        self._is_locked = False      # Indicates whether the lock is held.
        self._counter = itertools.count()  # A counter to break ties in priority.

    async def acquire(self, priority):
        """
        Acquire the lock for a task with the given priority.
        Lower values mean higher priority.
        When a task is granted the lock, it is immediately removed from the waiting queue.
        """
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        count = next(self._counter)
        async with self._lock:
            heapq.heappush(self._queue, (priority, count, fut))
            logging.debug(f"Task with priority {priority} added to the queue.")
            # If the lock is free and this task is at the head of the queue,
            # immediately grant the lock and remove its future from the queue.
            if not self._is_locked and self._queue[0][2] is fut:
                heapq.heappop(self._queue)
                self._is_locked = True
                fut.set_result(True)
                logging.debug(f"Lock acquired immediately for priority {priority}.")
        await fut  # Wait until the future is set.
        logging.debug(f"Lock acquired for priority {priority}.")

    async def release(self):
        """
        Release the lock.
        If there are waiting tasks, grant the lock to the one with the highest priority.
        """
        async with self._lock:
            if not self._is_locked:
                raise RuntimeError("Cannot release an unlocked AsyncQueuedLock")
            if self._queue:
                # Pop the next waiting task (with the lowest priority value)
                next_priority, _, next_fut = heapq.heappop(self._queue)
                self._is_locked = True  # Remains locked, now held by the next task.
                next_fut.set_result(True)
                logging.debug(f"Lock passed to next task with priority {next_priority}.")
            else:
                self._is_locked = False
                logging.debug("Lock is now free.")

    def locked(self):
        """
        Check if the lock is currently held.
        Returns True if locked, False otherwise.
        """
        return self._is_locked
