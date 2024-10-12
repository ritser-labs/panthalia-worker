from .db_adapter import DBAdapter
import aiofiles
import os
import sys
import shutil  # For copying directories and files
import logging

logger = logging.getLogger(__name__)

USE_SECIMPORT = False


# Define a global directory for plugins
global_plugin_dir = '/tmp/my_plugins'  # Or any directory of your choice

def setup_dir():
    if not os.path.exists(global_plugin_dir):
        os.makedirs(global_plugin_dir)
    if global_plugin_dir not in sys.path:
        sys.path.append(global_plugin_dir)

last_plugin_id = None
last_plugin = None
db_adapter = DBAdapter()

async def get_plugin(plugin_id):
    global last_plugin_id, last_plugin
    setup_dir()
    logger.info(f'Fetching plugin {plugin_id}')
    if plugin_id != last_plugin_id:
        # Create a subdirectory for the plugin (e.g., /tmp/my_plugins/plugin_5)
        plugin_package_dir = os.path.join(global_plugin_dir, f'plugin_{plugin_id}')
        if not os.path.exists(plugin_package_dir):
            os.makedirs(plugin_package_dir)

        # Write the plugin code to its own subdirectory
        plugin_file_name = f'plugin_{plugin_id}.py'
        plugin_path = os.path.join(plugin_package_dir, plugin_file_name)

        # Fetch the plugin code and write it to the plugin's subdirectory
        async with aiofiles.open(plugin_path, mode='w') as f:
            await f.write(await db_adapter.get_plugin_code(plugin_id))

        # Copy the adapters directory to the plugin's subdirectory
        local_adapters_dir = os.path.join(os.path.dirname(__file__), 'adapters')
        global_adapters_dir = os.path.join(plugin_package_dir, 'adapters')

        # Copy the entire adapters directory if it doesn't already exist
        if os.path.exists(local_adapters_dir) and not os.path.exists(global_adapters_dir):
            shutil.copytree(local_adapters_dir, global_adapters_dir)
        
        local_datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets')
        global_datasets_dir = os.path.join(plugin_package_dir, 'datasets')
        
        # Copy the entire datasets directory if it doesn't already exist
        if os.path.exists(local_datasets_dir) and not os.path.exists(global_datasets_dir):
            shutil.copytree(local_datasets_dir, global_datasets_dir)

        # Copy the tokenizer.py file to the plugin's subdirectory
        local_tokenizer_path = os.path.join(os.path.dirname(__file__), 'tokenizer.py')
        global_tokenizer_path = os.path.join(plugin_package_dir, 'tokenizer.py')

        if os.path.exists(local_tokenizer_path) and not os.path.exists(global_tokenizer_path):
            shutil.copy(local_tokenizer_path, global_tokenizer_path)

        # Copy the device.py file to the plugin's subdirectory
        local_device_path = os.path.join(os.path.dirname(__file__), 'device.py')
        global_device_path = os.path.join(plugin_package_dir, 'device.py')

        if os.path.exists(local_device_path) and not os.path.exists(global_device_path):
            shutil.copy(local_device_path, global_device_path)

        # Create an __init__.py file to mark the plugin directory as a package
        init_file_path = os.path.join(plugin_package_dir, '__init__.py')
        if not os.path.exists(init_file_path):
            with open(init_file_path, 'w') as f:
                f.write('# This is the init file for plugin package\n')

        logger.info(f'Plugin {plugin_id} code fetched and written to {plugin_path}')
        if USE_SECIMPORT:
            import secimport
            last_plugin = secimport.secure_import(f'plugin_{plugin_id}.plugin_{plugin_id}')
            last_plugin = getattr(last_plugin, f'plugin_{plugin_id}')
        else:
            import importlib
            last_plugin = importlib.import_module(f'plugin_{plugin_id}.plugin_{plugin_id}')
        last_plugin = getattr(last_plugin, 'exported_plugin')
        logger.info(f'Imported plugin {plugin_id} as package: {last_plugin} with dir {dir(last_plugin)}')
        last_plugin_id = plugin_id
        last_plugin.model_adapter.initialize_environment()

    return last_plugin
