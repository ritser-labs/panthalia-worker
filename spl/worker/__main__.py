# spl/worker/main.py
import asyncio
import time
import logging

from .config import args
from .db_client import db_adapter
from .tasks import deposit_stake, handle_task, report_sync_status
from .logging_config import logger

async def main():
    logger.info("Starting main process")

    subnet_in_db = await db_adapter.get_subnet(args.subnet_id)
    logger.info("Starting tensor synchronization...")
    reported = False

    processed_tasks = set()
    last_loop_time = time.time()

    while True:
        logger.debug(f'Loop time: {time.time() - last_loop_time:.2f} seconds')
        last_loop_time = time.time()

        await deposit_stake()

        if not reported:
            await report_sync_status()
            reported = True

        assigned_tasks = await db_adapter.get_assigned_tasks(subnet_in_db.id)
        for task in assigned_tasks:
            if task.id not in processed_tasks:
                asyncio.create_task(handle_task(task, time.time()))
                processed_tasks.add(task.id)

        await asyncio.sleep(args.poll_interval)

if __name__ == "__main__":
    asyncio.run(main())
