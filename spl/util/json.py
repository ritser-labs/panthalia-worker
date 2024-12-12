# spl/util/json.py
import os
import json
import aiofiles

async def load_json(file_path, default, file_lock):
    """
    Loads a JSON file asynchronously using aiofiles, protected by an asyncio.Lock.
    The file_lock must be an asyncio.Lock() to ensure exclusive access.
    """
    async with file_lock:
        if not os.path.exists(file_path):
            return default
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
            return json.loads(content)

async def save_json(file_path, data, file_lock):
    """
    Saves data to a JSON file asynchronously using aiofiles, protected by an asyncio.Lock.
    The file_lock must be an asyncio.Lock() to ensure exclusive access.
    """
    async with file_lock:
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(json.dumps(data))
