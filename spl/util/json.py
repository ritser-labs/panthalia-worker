import json
import os
import logging
import aiofiles

async def save_json(file_path, data, file_lock):
    with file_lock:
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(json.dumps(data))

async def load_json(file_path, default, file_lock):
    with file_lock:
        if os.path.exists(file_path):
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                if not content.strip():  # Check if the file is empty
                    logging.error(f"The file {file_path} is empty. Returning default value.")
                    return default
                try:
                    return json.loads(content)  # Try loading the JSON content
                except json.JSONDecodeError as e:
                    logging.error(f"JSONDecodeError in file {file_path}: {e}. Returning default value.")
                    async with aiofiles.open(file_path, 'w') as f:
                        await f.write(json.dumps(default))
                    return default
        else:
            logging.info(f"The file {file_path} does not exist. Saving default value.")
            return default
