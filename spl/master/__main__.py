# spl/master/__main__.py
import asyncio
from .config import args
from .jobs import check_for_new_jobs
from .logging_config import logger

if __name__ == "__main__":
    logger.info("Starting master process")
    asyncio.run(check_for_new_jobs(
        args.private_key,
        args.db_url,
        args.detailed_logs,
        args.num_workers,
        args.deploy_type,
        args.num_master_wallets
    ))
