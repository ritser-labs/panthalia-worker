import aiofiles
import os
import sys
import shutil  # for copying directories and files
import logging
import importlib

logger = logging.getLogger(__name__)

USE_SECIMPORT = False

global_plugin_dir = '/tmp/my_plugins'  # or any directory of your choice

def setup_dir():
    if not os.path.exists(global_plugin_dir):
        os.makedirs(global_plugin_dir)
    if global_plugin_dir not in sys.path:
        sys.path.append(global_plugin_dir)

def create_subdirectory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def copy_if_missing(src, dst):
    if os.path.exists(src) and not os.path.exists(dst):
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy(src, dst)

async def fetch_and_write_plugin_code(plugin_id, db_adapter, plugin_path):
    async with aiofiles.open(plugin_path, mode='w') as f:
        await f.write((await db_adapter.get_plugin(plugin_id)).code)

def setup_plugin_files(plugin_package_dir):
    # define the file and dir pairs to be copied
    resources = {
        'adapters': 'adapters',
        'datasets': 'datasets',
        'tokenizer.py': 'tokenizer.py',
        'device.py': 'device.py',
        'common.py': 'common.py'
    }
    
    # copy resources only if missing in target location
    for local, global_target in resources.items():
        src = os.path.join(os.path.dirname(__file__), local)
        dst = os.path.join(plugin_package_dir, global_target)
        copy_if_missing(src, dst)

def import_plugin(plugin_id):
    if USE_SECIMPORT:
        import secimport
        plugin_module = secimport.secure_import(f'plugin_{plugin_id}.plugin_{plugin_id}')
    else:
        plugin_module = importlib.import_module(f'plugin_{plugin_id}.plugin_{plugin_id}')
    return getattr(plugin_module, 'exported_plugin')

last_plugin_id = None
last_plugin = None

async def get_plugin(plugin_id, db_adapter):
    global last_plugin_id, last_plugin
    setup_dir()
    logger.info(f'fetching plugin {plugin_id}')

    if plugin_id != last_plugin_id:
        plugin_package_dir = os.path.join(global_plugin_dir, f'plugin_{plugin_id}')
        create_subdirectory(plugin_package_dir)

        # fetch and write plugin code
        plugin_file_name = f'plugin_{plugin_id}.py'
        plugin_path = os.path.join(plugin_package_dir, plugin_file_name)
        await fetch_and_write_plugin_code(plugin_id, db_adapter, plugin_path)

        # copy necessary plugin resources
        setup_plugin_files(plugin_package_dir)

        # create __init__.py to make it a package
        init_file_path = os.path.join(plugin_package_dir, '__init__.py')
        if not os.path.exists(init_file_path):
            with open(init_file_path, 'w') as f:
                f.write('# this is the init file for plugin package\n')

        # import the plugin
        last_plugin = import_plugin(plugin_id)
        logger.info(f'imported plugin {plugin_id} as package: {last_plugin} with dir {dir(last_plugin)}')
        last_plugin_id = plugin_id
        last_plugin.model_adapter.initialize_environment()

    return last_plugin
