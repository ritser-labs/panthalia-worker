import argparse
import asyncio
from hypercorn.asyncio import serve
from hypercorn.config import Config
import logging

from .app import app, logger, set_perm_modify_db, get_perm_modify_db
from .adapter import db_adapter_server, init_db  # Import init_db

async def background_tasks():
    # Run a periodic background loop to ensure open orders still meet hold expiry criteria
    # For demonstration, run it every 3600 seconds (1 hour)
    # In production, choose an appropriate interval.
    while True:
        try:
            await db_adapter_server.check_and_cleanup_holds()
        except Exception as e:
            logger.error(f"Error in check_and_cleanup_holds: {e}")
        await asyncio.sleep(3600)  # Wait 1 hour before next check

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Database Server")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind')
    parser.add_argument('--port', type=int, default=8000, help='Port')
    parser.add_argument('--perm', type=int, default=0, help='Permission ID')
    parser.add_argument('--root_wallet', type=str, default='0x0', help='Root wallet address')
    args = parser.parse_args()

    set_perm_modify_db(args.perm)

    async def main():
        await init_db()
        await db_adapter_server.create_perm(args.root_wallet, get_perm_modify_db())

        config = Config()
        config.bind = [f'{args.host}:{args.port}']

        # Run background tasks concurrently with the main server
        server_task = asyncio.create_task(serve(app, config))
        bg_task = asyncio.create_task(background_tasks())
        logger.info(f"Starting DB Server on {args.host}:{args.port}...")
        logger.info(f"Permission ID: {args.perm}")
        logger.info(f"Root wallet address: {args.root_wallet}")

        await asyncio.gather(server_task, bg_task)

    asyncio.run(main())
