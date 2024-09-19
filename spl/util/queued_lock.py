import threading
import queue
import itertools
import logging
import asyncio
import heapq


# AsyncQueuedLock implementation
class AsyncQueuedLock:
    def __init__(self):
        self._queue = []
        self._lock = asyncio.Lock()
        self._is_locked = False
        self._counter = itertools.count()
    
    async def acquire(self, priority):
        """
        Acquire the lock for a task with a given priority.
        :param priority: The priority (lower values get higher priority)
        """
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        count = next(self._counter)
        async with self._lock:
            heapq.heappush(self._queue, (priority, count, fut))
            logging.debug(f"Task with priority {priority} added to the queue.")
            if not self._is_locked and self._queue[0][2] == fut:
                self._is_locked = True
                fut.set_result(True)
                logging.debug(f"Lock acquired immediately for priority {priority}.")
        
        await fut  # Wait until the future is set
        logging.debug(f"Lock acquired for priority {priority}.")

    async def release(self):
        """
        Release the lock, allowing the next task in the queue (based on priority) to proceed.
        """
        async with self._lock:
            if not self._is_locked:
                raise RuntimeError("Cannot release an unlocked AsyncQueuedLock")
            
            # Remove the current holder
            if self._queue:
                released_task = heapq.heappop(self._queue)
                logging.debug(f"Lock released for priority {released_task[0]}.")
            
            # Assign the lock to the next task in the queue
            if self._queue:
                next_priority, next_count, next_fut = self._queue[0]
                self._is_locked = True
                if not next_fut.done():
                    next_fut.set_result(True)
                logging.debug(f"Lock passed to next task with priority {next_priority}.")
            else:
                self._is_locked = False
                logging.debug("Lock is now free.")
    
    def locked(self):
        """
        Check if the lock is currently held.
        :return: True if the lock is held, False otherwise.
        """
        return self._is_locked