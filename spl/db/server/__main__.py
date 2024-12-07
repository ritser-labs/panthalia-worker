import argparse
import asyncio
from hypercorn.asyncio import serve
from hypercorn.config import Config
import logging

from .app import app, logger, set_perm_modify_db, get_perm_modify_db
from .adapter import db_adapter_server

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Database Server")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind')
    parser.add_argument('--port', type=int, default=8000, help='Port')
    parser.add_argument('--perm', type=int, default=0, help='Permission ID')
    parser.add_argument('--root_wallet', type=str, default='0x0', help='Root wallet address')
    args = parser.parse_args()

    set_perm_modify_db(args.perm)

    asyncio.run(db_adapter_server.create_perm(args.root_wallet, get_perm_modify_db()))

    config = Config()
    config.bind = [f'{args.host}:{args.port}']

    logger.info(f"Starting DB Server on {args.host}:{args.port}...")
    logger.info(f"Permission ID: {args.perm}")
    logger.info(f"Root wallet address: {args.root_wallet}")
    asyncio.run(serve(app, config))
