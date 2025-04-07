# spl/worker/queue.py
import time
import logging
import asyncio

logger = logging.getLogger(__name__)

class TaskQueue:
    def __init__(self):
        self.queue = []
        self.current_version = None
        self.lock = asyncio.Lock()
        logger.debug("Initialized TaskQueue")

    async def add_task(self, task):
        async with self.lock:
            self.queue.append(task)
            self.queue.sort(key=lambda t: t['time_solver_selected'])
            how_old = int(time.time()) - task['time_solver_selected']
            logger.debug(f"Added task: {task} that is {how_old} seconds old. Queue size is now {len(self.queue)}")

    async def get_next_task(self):
        async with self.lock:
            if self.queue:
                task = self.queue.pop(0)
                logger.debug(f"Retrieved task: {task}. Queue size is now {len(self.queue)}")
                return task
            logger.debug("No tasks in the queue.")
            return None

    async def queue_length(self):
        async with self.lock:
            return len(self.queue)
