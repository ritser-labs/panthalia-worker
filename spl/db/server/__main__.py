# file: spl/db/server/__main__.py

import argparse
import asyncio
from hypercorn.asyncio import serve
from hypercorn.config import Config
import logging

from .app import (
    app, 
    logger, 
    set_perm_modify_db, 
    get_perm_modify_db,
    generate_ephemeral_db_sot_key,
    get_db_sot_address
)
from .adapter import init_db
from .db_server_instance import db_adapter_server

async def background_tasks():
    while True:
        try:
            # Cleanup old holds
            await db_adapter_server.check_and_cleanup_holds()

            # Also expire old Stripe sessions
            # e.g. older_than_minutes=120 => 2 hours
            await db_adapter_server.expire_old_stripe_deposits(older_than_minutes=120)

        except Exception as e:
            logger.error(f"Error in background_tasks: {e}")
        await asyncio.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Database Server")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind')
    parser.add_argument('--port', type=int, default=8000, help='Port')
    parser.add_argument('--perm', type=int, default=0, help='Permission ID')
    parser.add_argument('--root_wallet', type=str, default='0x0', help='Root wallet address')
    args = parser.parse_args()

    set_perm_modify_db(args.perm)

    async def main():
        # 1) create ephemeral DB key
        generate_ephemeral_db_sot_key()
        ephemeral_addr = get_db_sot_address()

        await init_db()
        # Possibly create a perm for the root wallet
        await db_adapter_server.create_perm(args.root_wallet, get_perm_modify_db())

        config = Config()
        config.bind = [f'{args.host}:{args.port}']

        server_task = asyncio.create_task(serve(app, config))
        bg_task = asyncio.create_task(background_tasks())

        logger.info(f"Starting DB Server on {args.host}:{args.port}")
        logger.info(f"DB ephemeral address for SOT: {ephemeral_addr}")

        await asyncio.gather(server_task, bg_task)

    asyncio.run(main())
