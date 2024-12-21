# spl/sot/__main__.py

import argparse
import asyncio
import logging
from spl.db.db_adapter_client import DBAdapterClient
from spl.plugins.manager import get_plugin
from spl.plugins.serialize import serialize_data, deserialize_data

logging.basicConfig(level=logging.INFO)

plugin = None  # a global to store the plugin we fetch

async def main(sot_id, db_url, private_key, job_id, plugin_id, perm, port):
    global plugin
    # get the plugin
    plugin = await get_plugin(plugin_id, DBAdapterClient(db_url, private_key), port)
    # initialize the adapter; now it will start serving
    await plugin.call_submodule(
        'sot_adapter',
        'initialize',
        sot_id,
        db_url,
        private_key,
        job_id,
        perm,
        port
    )

def run():
    parser = argparse.ArgumentParser(description="SOT HTTP Proxy (direct serving)")
    parser.add_argument('--sot_id', type=int, required=True)
    parser.add_argument('--db_url', type=str, required=True)
    parser.add_argument('--private_key', type=str, required=True)
    parser.add_argument('--port', type=int, default=5001)
    args = parser.parse_args()

    loop = asyncio.get_event_loop()

    # Get SOT and job
    db_adapter = DBAdapterClient(args.db_url, args.private_key)
    sot_db_obj = loop.run_until_complete(db_adapter.get_sot(args.sot_id))
    job_id = sot_db_obj.job_id
    job_obj = loop.run_until_complete(db_adapter.get_job(job_id))
    plugin_id = job_obj.plugin_id
    perm = sot_db_obj.perm

    # run main coroutine
    loop.run_until_complete(main(
        args.sot_id,
        args.db_url,
        args.private_key,
        job_id,
        plugin_id,
        perm,
        args.port
    ))

if __name__ == "__main__":
    run()
