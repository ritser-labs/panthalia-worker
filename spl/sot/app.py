# spl/sot/app.py
import os
import logging
import asyncio
import tracemalloc
from quart import Quart
from ..db.db_adapter_client import DBAdapterClient
from ..common import SOT_PRIVATE_PORT
from ..plugins.manager import get_plugin
from .utils import (
    log_memory_usage, initialize_all_tensors
)

MEMORY_LOGGING_ENABLED = False

def create_app(sot_id, db_url, private_key, enable_memory_logging=False):
    global MEMORY_LOGGING_ENABLED
    MEMORY_LOGGING_ENABLED = enable_memory_logging

    if MEMORY_LOGGING_ENABLED:
        tracemalloc.start()

    app = Quart(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024 * 1024  # 1 TB
    # END FIX

    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, 'data')
    state_dir = os.path.join(data_dir, 'state')
    temp_dir = os.path.join(state_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    app.config['SOT_ID'] = sot_id
    app.config['DB_URL'] = db_url
    app.config['PRIVATE_KEY'] = private_key
    app.config['DATA_DIR'] = data_dir
    app.config['STATE_DIR'] = state_dir
    app.config['TEMP_DIR'] = temp_dir
    app.config['MEMORY_LOGGING_ENABLED'] = enable_memory_logging

    app.config['db_adapter'] = None
    app.config['plugin'] = None
    app.config['perm_db'] = None
    app.config['job_id'] = None
    app.config['synced_workers'] = 0
    app.config['latest_loss'] = None

    # Initialize locks for all JSON files or resources accessed concurrently
    app.config['file_locks'] = {
        'block_timestamps': asyncio.Lock(),
        'num_updates': asyncio.Lock(),
        'iteration_number': asyncio.Lock(),
        'last_future_version_number': asyncio.Lock(),
        'latest_loss': asyncio.Lock()
    }

    # Lock used during timestamp updates
    app.config['update_timestamp_lock'] = asyncio.Lock()

    log_memory_usage('Before initializing or loading initial state', enabled=enable_memory_logging)

    logging.info("Initializing or loading initial state...")

    @app.before_serving
    async def before_serving():
        logging.info("App is starting to serve.")
        db_adapter = DBAdapterClient(
            app.config['DB_URL'],
            app.config['PRIVATE_KEY']
        )
        app.config['db_adapter'] = db_adapter

        # Initialize service (async)
        await initialize_service(app)

    @app.after_serving
    async def after_serving():
        logging.info("App is shutting down.")

    return app

async def initialize_service(app):
    sot_id = app.config['SOT_ID']
    db_adapter = app.config['db_adapter']

    sot_db_obj = await db_adapter.get_sot(sot_id)
    job_id = sot_db_obj.job_id
    perm_db = sot_db_obj.perm
    app.config['job_id'] = job_id
    app.config['perm_db'] = perm_db

    plugin_id = (await db_adapter.get_job(job_id)).plugin_id
    plugin = await get_plugin(plugin_id, db_adapter)
    app.config['plugin'] = plugin

    logging.info(
        f"Initializing service for SOT {sot_id}, job {job_id}, plugin {plugin_id}, perm {perm_db}"
    )

    # Pass file_locks explicitly to initialize_all_tensors
    await initialize_all_tensors(
        app.config['STATE_DIR'],
        plugin,
        memory_logging=app.config['MEMORY_LOGGING_ENABLED'],
        file_locks=app.config['file_locks']
    )
    await plugin.call_submodule('dataset', 'initialize_dataset')
    logging.info("Service initialized")
