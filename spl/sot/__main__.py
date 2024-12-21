# spl/sot/__main__.py

import argparse
import asyncio
import logging
from aiohttp import web

from spl.db.db_adapter_client import DBAdapterClient
from spl.plugins.manager import get_plugin

logging.basicConfig(level=logging.INFO)

plugin = None  # a global to store the plugin we fetch

async def handle_request(request):
    """
    For every HTTP request that hits the SOT (on /whatever),
    we forward it to 'sot_adapter.handle_request(...)' in the plugin.
    """
    global plugin

    # Gather request info
    method = request.method
    path = request.rel_url.path.lstrip('/')   # e.g. "health" or "get_batch"
    query = dict(request.query)
    headers = dict(request.headers)
    body = await request.read()

    # Now forward to plugin's sot_adapter submodule
    result = await plugin.call_submodule(
        'sot_adapter',           # we named it 'sot_adapter' in plugin
        'handle_request',
        method, path, query, headers, body
    )
    # 'result' should be a dict with 'status', 'headers', 'body'

    if (not isinstance(result, dict)
            or 'status' not in result
            or 'body' not in result):
        return web.Response(
            status=500,
            text="Invalid response from plugin sot_adapter"
        )

    status_code = result['status']
    r_headers = result.get('headers', {})
    r_body = result['body']

    # Build the aiohttp response
    resp = web.StreamResponse(status=status_code)
    for k, v in r_headers.items():
        resp.headers[k] = v
    await resp.prepare(request)

    # Write the body
    if hasattr(r_body, '__aiter__'):
        # If it's an async generator, we can stream it
        async for chunk in r_body:
            await resp.write(chunk)
    else:
        # Otherwise, just write once
        await resp.write(r_body)
    return resp

async def init_app(sot_id, db_url, private_key, port):
    """
    Initialize the SOT app:
     - Connect to DB
     - Get plugin
     - Initialize plugin's sot_adapter
     - Return an aiohttp web.Application
    """
    db_adapter = DBAdapterClient(db_url, private_key)
    sot_db_obj = await db_adapter.get_sot(sot_id)
    job_id = sot_db_obj.job_id
    job_obj = await db_adapter.get_job(job_id)
    plugin_id = job_obj.plugin_id

    # 1) get the plugin from manager
    global plugin
    plugin = await get_plugin(plugin_id, db_adapter)

    # 3) Initialize the adapter
    await plugin.call_submodule(
        'sot_adapter',
        'initialize',
        sot_id,
        db_url,
        private_key,
        job_id,
        sot_db_obj.perm
    )

    # 4) Create the web.Application
    app = web.Application()
    app.router.add_route('*', '/{tail:.*}', handle_request)
    logging.info(f"Starting SOT proxy on port {port}")
    return app

def main():
    parser = argparse.ArgumentParser(description="SOT HTTP Proxy")
    parser.add_argument('--sot_id', type=int, required=True)
    parser.add_argument('--db_url', type=str, required=True)
    parser.add_argument('--private_key', type=str, required=True)
    parser.add_argument('--port', type=int, default=5001)
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    app = loop.run_until_complete(init_app(
        args.sot_id,
        args.db_url,
        args.private_key,
        args.port
    ))

    web.run_app(app, port=args.port)

if __name__ == "__main__":
    main()
