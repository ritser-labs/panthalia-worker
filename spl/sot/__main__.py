# spl/sot/main.py
import argparse
import asyncio
import logging
from hypercorn.asyncio import serve
from hypercorn.config import Config
from .app import create_app
from .routes import register_routes

def main():
    parser = argparse.ArgumentParser(description="Source of Truth (SOT) Service")
    parser.add_argument('--enable_memory_logging', action='store_true', help="Enable memory logging")
    parser.add_argument('--sot_id', type=int, required=True, help="ID for the SOT service")
    parser.add_argument('--db_url', type=str, required=True, help="URL for the database")
    parser.add_argument('--private_key', type=str, required=True, help="Private key for the database")

    args = parser.parse_args()

    app = create_app(args.sot_id, args.db_url, args.private_key, enable_memory_logging=args.enable_memory_logging)
    register_routes(app)

    logging.info("Starting SOT service...")

    config = Config()
    from ..common import SOT_PRIVATE_PORT
    config.bind = [f'0.0.0.0:{SOT_PRIVATE_PORT}']

    asyncio.run(serve(app, config))

if __name__ == "__main__":
    main()
