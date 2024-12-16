import subprocess
import json
import os
import time
import argparse
import requests
import threading
import curses
from flask import Flask, jsonify, send_from_directory
from ..common import wait_for_health, DB_PORT
from ..db.db_adapter_client import DBAdapterClient
from ..models import init_db, db_path, PermType, ServiceType
from ..plugins.manager import get_plugin, global_plugin_dir
from eth_account import Account
import glob
import shutil
import asyncio
import logging
import aiofiles
import docker

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
package_root_dir = os.path.dirname(parent_dir)

DATA_DIR = os.path.join(parent_dir, 'data')
STATE_DIR = os.path.join(DATA_DIR, 'state')
TEMP_DIR = os.path.join(STATE_DIR, 'temp')
MASTER_WALLETS_FILE = os.path.join(DATA_DIR, 'master_wallets.json')
DEPLOY_SCRIPT = os.path.join(parent_dir, 'script', 'Deploy.s.sol')
LOG_DIR = os.path.join(parent_dir, 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'test_run.log')
SOT_LOG_FILE = os.path.join(LOG_DIR, 'sot.log')
DB_LOG_FILE = os.path.join(LOG_DIR, 'db.log')
BLOCK_TIMESTAMPS_FILE = os.path.join(STATE_DIR, 'block_timestamps.json')
LAST_FUTURE_VERSION_FILE = os.path.join(STATE_DIR, 'last_future_version_number.json')
plugin_file = os.path.join(parent_dir, 'plugins', 'plugin.py')
DOCKER_IMAGE = 'zerogoliath/magnum:latest'
DB_HOST = '127.0.0.1'
db_url = f"http://{DB_HOST}:{DB_PORT}"

GUESS_DB_PERM_ID = 1

# Configure logging
os.makedirs(LOG_DIR, exist_ok=True)

# Setup the logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create file handler and add it to logger
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(formatter)

# Avoid adding duplicate FileHandlers
if not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
    logger.addHandler(file_handler)

# Check if StreamHandler is already present and avoid duplicates
if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def parse_args():
    parser = argparse.ArgumentParser(description="Test run script for starting workers and master")
    parser.add_argument('--private_key', type=str, required=True, help="Private key of the deployer's Ethereum account")
    parser.add_argument('--detailed_logs', action='store_true', help="Enable detailed logs for all processes")
    parser.add_argument('--num_master_wallets', type=int, default=70, help="Number of wallets to generate for the master process")
    parser.add_argument('--worker_count', type=int, default=1, help="Number of workers to start")
    parser.add_argument('--torch_compile', action='store_true', help="Enable torch.compile and model warmup")
    return parser.parse_args()

sync_status = {}

latest_loss_cache = {
    'value': None,
    'last_fetched': 0
}

LOSS_REFRESH_INTERVAL = 60
app = Flask(__name__)

def tail_file(file_path, n=1000):
    """
    Efficiently fetch the last n lines from a file without reading the entire file.
    """
    lines = []
    try:
        with open(file_path, 'rb') as f:
            f.seek(0, 2)
            file_size = f.tell()
            block_size = 1024
            data = []
            lines_found = 0
            while lines_found < n and file_size > 0:
                read_size = min(block_size, file_size)
                f.seek(file_size - read_size)
                block = f.read(read_size)
                data.insert(0, block)
                lines_in_block = block.count(b'\n')
                lines_found += lines_in_block
                file_size -= read_size
            all_data = b''.join(data).splitlines()
            lines = all_data[-n:]
    except Exception as e:
        logging.debug(f"Error reading tail of file {file_path}: {e}")
    return [line.decode('utf-8', errors='replace') for line in lines]

def generate_wallets(num_wallets):
    wallets = []
    for _ in range(num_wallets):
        account = Account.create()
        wallets.append({'private_key': account._private_key.hex(), 'address': account.address})
    return wallets

def delete_old_tensor_files(directory, timestamps_file, last_future_version_file):
    if not os.path.exists(directory):
        return
    if not os.path.exists(timestamps_file):
        return

    with open(timestamps_file, 'r') as f:
        block_timestamps = json.load(f)
    
    with open(last_future_version_file, 'r') as f:
        last_future_version = json.load(f)

    tensor_files = glob.glob(os.path.join(directory, '*.pt'))
    latest_files = {f"{name}_{version}.pt" for name, version in block_timestamps.items()}
    last_future_version_files = {f"{name}_{version}.pt" for name, version in last_future_version.items()}

    for tensor_file in tensor_files:
        if os.path.basename(tensor_file) not in latest_files and os.path.basename(tensor_file) not in last_future_version_files:
            try:
                os.remove(tensor_file)
                logging.debug(f"Deleted old tensor file: {tensor_file}")
            except Exception as e:
                logging.debug(f"Error deleting file {tensor_file}: {e}")

def delete_directory_contents(directory):
    if os.path.exists(directory):
        try:
            shutil.rmtree(directory)
            logging.debug(f"Deleted directory: {directory}")
        except Exception as e:
            logging.debug(f"Error deleting directory {directory}: {e}")

async def terminate_processes(db_adapter):
    # Initialize the Docker client
    docker_client = docker.from_env()

    # Fetch all running Docker containers
    containers = docker_client.containers.list()

    # Filter and stop containers whose image name is "panthalia_plugin"
    for container in containers:
        if "panthalia_plugin" in container.image.tags:
            try:
                container.stop()
                logging.info(f"Stopped Docker container with image: {container.image.tags}")
            except Exception as e:
                logging.error(f"Error stopping Docker container with image {container.image.tags}: {e}")

    # Terminate the processes as before
    instances = await db_adapter.get_all_instances()
    for instance in instances:
        try:
            pid = int(instance.process_id)
            if pid > 0:
                process = subprocess.Popen(["kill", "-TERM", str(pid)])
                process.wait(timeout=5)
                logging.info(f"Process {instance.name} (PID {pid}) terminated successfully.")
        except subprocess.TimeoutExpired:
            logging.warning(f"Process {instance.name} (PID {pid}) did not terminate in time, forcefully killing it.")
            process = subprocess.Popen(["kill", "-KILL", str(pid)])
            process.wait()
            logging.info(f"Process {instance.name} (PID {pid}) was killed forcefully.")
        except Exception as e:
            logging.error(f"Error terminating process {instance.name}: {e}")


def reset_logs(log_dir):
    """Delete all log files in the log directory except for the LOG_FILE, which is truncated (reset)."""
    if os.path.exists(log_dir):
        for file_name in os.listdir(log_dir):
            file_path = os.path.join(log_dir, file_name)
            try:
                if file_path == LOG_FILE:
                    # Truncate LOG_FILE by opening it in write mode
                    open(file_path, 'w').close()
                    logging.debug(f"Reset log file: {file_path}")
                else:
                    # Remove other log files
                    os.remove(file_path)
                    logging.debug(f"Deleted log file: {file_path}")
            except Exception as e:
                logging.debug(f"Error processing log file {file_path}: {e}")
    else:
        logging.debug(f"Log directory {log_dir} does not exist.")

def fetch_latest_loss(sot_url):
    global latest_loss_cache
    current_time = time.time()

    # Check if the cache is expired
    if current_time - latest_loss_cache['last_fetched'] > LOSS_REFRESH_INTERVAL:
        try:
            response = requests.get(f"{sot_url}/get_loss", timeout=1)
            if response.status_code == 200:
                data = response.json()
                latest_loss_cache['value'] = data.get('loss', None)
                latest_loss_cache['last_fetched'] = current_time
            else:
                logging.error(f"Error fetching latest loss: {response.status_code} - {response.text}")
        except requests.RequestException as e:
            logging.error(f"Error fetching latest loss: {e}")

    return latest_loss_cache['value']

async def monitor_processes(stdscr, db_adapter, task_counts):
    global args
    logger = logging.getLogger()
    for handler in logger.handlers:
        if not isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)
    curses.noecho()
    selected_process = 0
    last_resize = None

    # Initialize colors
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
    
    instances_last = await db_adapter.get_all_instances()
    instances_last_check = time.time()
    
    async def get_instances():
        nonlocal instances_last, instances_last_check
        POLL_INTERVAL = 1
        if time.time() - instances_last_check > POLL_INTERVAL:
            instances_last = await db_adapter.get_all_instances()
            instances_last_check = time.time()
        return instances_last

    max_name_length = max(len(instance.name) for instance in await get_instances()) + 14
    v_string = "PANTHALIA SIMULATOR V0"
    max_name_length = max(max_name_length, len(v_string) + 2)
    right_col_width = max_name_length + 2
    

    async def draw_screen():
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        split_point = width - right_col_width

        instances = await get_instances()
        ordered_process_names = sorted([instance.name for instance in instances], key=lambda name: (
            name.startswith('worker_final_logits'),
            name.startswith('worker_forward'),
            name.startswith('worker_embed'),
            not name.startswith('worker'),
            name
        ))

        # Display logs on the left side
        process_name = ordered_process_names[selected_process]
        log_file = os.path.join(LOG_DIR, f"{process_name}.log")

        # Instead of reading the entire file, we only read the last (height-1) lines:
        if os.path.exists(log_file):
            log_lines = tail_file(log_file, height - 1)
        else:
            log_lines = []

        for i, line in enumerate(log_lines):
            stdscr.addstr(i, 0, line[:split_point - 2])

        # Draw the separator line
        for y in range(height):
            stdscr.addch(y, split_point - 2, curses.ACS_VLINE)

        for i, name in enumerate(ordered_process_names):
            found_instance = None
            for instance in instances:
                if instance.name == name:
                    found_instance = instance
                    break
            if found_instance is None:
                continue
            instance = found_instance
            is_selected = (i == selected_process)
            pid = int(instance.process_id)
            status = (pid > 0) and (os.path.exists(f"/proc/{pid}"))
            color = curses.color_pair(1) if status else curses.color_pair(2)
            indicator = '*' if is_selected else ' '

            stdscr.addstr(i, split_point, f"{indicator} {name}", color)

        # Draw task counts
        task_start = height - 3 - len(task_counts)
        for i, (task_type, (solver_selected, active)) in enumerate(task_counts.items()):
            stdscr.addstr(task_start + i, split_point, f"{task_type}: {solver_selected}/{active}", curses.color_pair(3))
        logging.debug(f'H: {height}, W: {width}, SP: {split_point}, RP: {right_col_width}')
        stdscr.addstr(height - 1, split_point, v_string, curses.color_pair(3))
        stdscr.refresh()

    await draw_screen()  # Initial draw

    while True:
        key = stdscr.getch()
        if key == curses.KEY_UP:
            selected_process = (selected_process - 1) % len(await get_instances())
            await draw_screen()
        elif key == curses.KEY_DOWN:
            selected_process = (selected_process + 1) % len(await get_instances())
            await draw_screen()
        elif key == curses.KEY_RESIZE:
            last_resize = time.time()
        elif key == ord('q'):
            await terminate_processes(db_adapter)
            break

        if last_resize and time.time() - last_resize > 0.1:
            await draw_screen()
            last_resize = None

        await draw_screen()
        time.sleep(0.05)

    stdscr.keypad(False)
    curses.endwin()
    os._exit(0)  # Force exit the program


def run_monitor_processes(stdscr, db_adapter, task_counts):
    asyncio.run(monitor_processes(stdscr, db_adapter, task_counts))

async def main():
    global base_url
    task_counts = {}

    base_url = f"http://localhost:5002"

    db_adapter = DBAdapterClient(db_url, args.private_key)
    
    reset_logs(LOG_DIR)
    if os.path.exists(db_path):
        os.remove(db_path)
    
    if os.path.exists(global_plugin_dir):
        shutil.rmtree(global_plugin_dir, ignore_errors=True)

    logging.info("Starting DB instance...")
    db_log = open(DB_LOG_FILE, 'w')
    public_key = Account.from_key(args.private_key).address
    db_process = subprocess.Popen(
        [
            'python', '-m', 'spl.db.server',
            '--host', DB_HOST,
            '--port', DB_PORT,
            '--perm', str(GUESS_DB_PERM_ID),
            '--root_wallet', public_key
        ],
        stdout=db_log, stderr=db_log, cwd=package_root_dir
    )
    await wait_for_health(db_url)
    await db_adapter.create_instance("db", ServiceType.Db.name, None, args.private_key, '', db_process.pid)
    logging.info(f"DB instance started with PID {db_process.pid}")

    try:
        delete_old_tensor_files(STATE_DIR, BLOCK_TIMESTAMPS_FILE, LAST_FUTURE_VERSION_FILE)
        delete_directory_contents(TEMP_DIR)

        # Start Flask server in a separate thread
        flask_thread = threading.Thread(target=lambda: app.run(port=5002))
        flask_thread.start()

        logging.info("Starting deployment...")

        logging.info(f'Time in string: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
        
        subnet_id = await db_adapter.create_subnet(
            5 * 60,
            2 * 60,
            10
        )

        db_perm_id = await db_adapter.create_perm_description(PermType.ModifyDb.name)
        
        assert db_perm_id == GUESS_DB_PERM_ID, f"Expected db_perm_id {GUESS_DB_PERM_ID}, got {db_perm_id}"


        await db_adapter.create_perm(args.private_key, db_perm_id)

        try:
            logging.info("Starting master process...")

            master_log = open(os.path.join(LOG_DIR, 'master.log'), 'w')
            master_command = [
                'python', '-m', 'spl.master',
                '--private_key', args.private_key,
                '--db_url', db_url,
                '--num_workers', str(args.worker_count),
                '--deploy_type', 'local'
            ]
            if args.detailed_logs:
                master_command.append('--detailed_logs')
            if args.torch_compile:
                master_command.append('--torch_compile')
            master_process = subprocess.Popen(master_command, stdout=master_log, stderr=master_log, cwd=package_root_dir)
            await db_adapter.create_instance("master", ServiceType.Master.name, None, args.private_key, '', master_process.pid)
            logging.info(f"Started master process with command: {' '.join(master_command)}")

            logging.info("Master process started.")

            curses_thread = threading.Thread(target=curses.wrapper, args=(run_monitor_processes, db_adapter, task_counts))
            curses_thread.start()
        except Exception as e:
            logging.error(f"Error: {e}", exc_info=True)
            await terminate_processes(db_adapter)
            exit(1)
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        await terminate_processes(db_adapter)
        exit(1)
    finally:
        for instance in await db_adapter.get_all_instances():
            pid = int(instance.process_id)
            if (pid > 0) and (os.path.exists(f"/proc/{pid}")):
                logging.info(f"Process {instance.name} terminated with exit code {pid}")
            else:
                logging.warning(f"Process {instance.name} was killed before completion.")

        logging.info("All processes terminated.")

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main())
