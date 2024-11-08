import subprocess
import json
import os
import time
import argparse
import requests
import threading
import curses
from flask import Flask, jsonify, send_from_directory
from ..common import load_abi, wait_for_health, wait_for_rpc_available, fund_wallets, DB_PORT, generate_wallets
from ..db.db_adapter_client import DBAdapterClient
from ..models import init_db, db_path, PermType, ServiceType
from ..plugin_manager import get_plugin, global_plugin_dir
from web3 import AsyncWeb3, Web3
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
ANVIL_LOG_FILE = os.path.join(LOG_DIR, 'anvil.log')
SOT_LOG_FILE = os.path.join(LOG_DIR, 'sot.log')
DB_LOG_FILE = os.path.join(LOG_DIR, 'db.log')
BLOCK_TIMESTAMPS_FILE = os.path.join(STATE_DIR, 'block_timestamps.json')
LAST_FUTURE_VERSION_FILE = os.path.join(STATE_DIR, 'last_future_version_number.json')
plugin_file = os.path.join(parent_dir, 'plugins', 'plugin.py')
DOCKER_IMAGE = 'zerogoliath/magnum:latest'
DB_HOST = '127.0.0.1'
db_url = f"http://{DB_HOST}:{DB_PORT}"
RPC_URL = 'http://localhost:8545'

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
    parser.add_argument('--subnet_addresses', type=str, required=True, help="Path to the subnet addresses JSON file")
    parser.add_argument('--deployment_config', type=str, required=True, help="Path to the deployment configuration JSON file")
    parser.add_argument('--sot_url', type=str, required=True, help="Source of Truth URL for streaming gradient updates")
    parser.add_argument('--private_key', type=str, required=True, help="Private key of the deployer's Ethereum account")
    parser.add_argument('--group', type=int, required=True, help="Group for depositing stake")
    parser.add_argument('--local_storage_dir', type=str, default='data', help="Directory for local storage of files")
    parser.add_argument('--forge_script', type=str, default=DEPLOY_SCRIPT, help="Path to the Forge deploy script")
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

async def terminate_processes(db_adapter, job_id):
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

async def monitor_processes(stdscr, db_adapter, job_id, task_counts):
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
        log_lines = []

        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_lines.extend(f.readlines())

        for i, line in enumerate(log_lines[-(height - 1):]):
            stdscr.addstr(i, 0, line[:split_point - 2])

        # Draw the separator line
        for y in range(height):
            stdscr.addch(y, split_point - 2, curses.ACS_VLINE)

        for i, name in enumerate(ordered_process_names):
            # inefficient but works for now
            found_instance = None
            for instance in instances:
                if instance.name == name:
                    found_instance = instance
            if found_instance is None:
                continue
            instance = found_instance
            is_selected = (i == selected_process)
            pid = int(instance.process_id)
            status = (pid > 0) and (os.path.exists(f"/proc/{pid}"))
            color = curses.color_pair(1) if status else curses.color_pair(2)
            indicator = '*' if is_selected else ' '

            stdscr.addstr(i, split_point, f"{indicator} {name}", color)

        # Fetch and display the latest loss
        latest_loss = fetch_latest_loss(args.sot_url)
        loss_display = f"Latest Loss: {latest_loss:.3f}" if latest_loss is not None else "Latest Loss: N/A"
        loss_y = height - len(task_counts) - 5
        stdscr.addstr(loss_y, split_point, loss_display, curses.color_pair(4))

        # Draw task counts below the latest loss
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
            await terminate_processes(db_adapter, job_id)
            break

        if last_resize and time.time() - last_resize > 0.1:
            await draw_screen()
            last_resize = None

        await draw_screen()
        time.sleep(0.05)

    stdscr.keypad(False)
    curses.endwin()
    os._exit(0)  # Force exit the program


async def track_tasks(web3, subnet_addresses, pool_contract, task_counts):
    contracts = {}
    filters = {}
    tasks = {}

    # Load the contracts and set up filters for events
    for task_type, address in subnet_addresses.items():
        abi = load_abi('SubnetManager')
        contracts[task_type] = web3.eth.contract(address=address, abi=abi)

        # Create filters for task-related events
        filters[task_type] = {
            'TaskRequestSubmitted': await contracts[task_type].events.TaskRequestSubmitted.create_filter(fromBlock='latest'),
            'SolutionSubmitted': await contracts[task_type].events.SolutionSubmitted.create_filter(fromBlock='latest'),
            'SolverSelected': await contracts[task_type].events.SolverSelected.create_filter(fromBlock='latest'),
            'TaskResolved': await contracts[task_type].events.TaskResolved.create_filter(fromBlock='latest')
        }

    # Main tracking loop
    while True:
        for task_type, contract_filters in filters.items():
            for event_name, event_filter in contract_filters.items():
                new_entries = await event_filter.get_new_entries()
                for event in new_entries:
                    task_id = event['args']['taskId']
                    if task_type not in tasks:
                        tasks[task_type] = {}
                    if event_name == 'TaskRequestSubmitted':
                        tasks[task_type][task_id] = {'active': True, 'solver_selected': False}
                    elif event_name == 'SolverSelected':
                        if task_id in tasks[task_type]:
                            tasks[task_type][task_id]['solver_selected'] = True
                    elif event_name == 'SolutionSubmitted':
                        if task_id in tasks[task_type]:
                            tasks[task_type][task_id]['active'] = False
                    elif event_name == 'TaskResolved':
                        if task_id in tasks[task_type]:
                            tasks[task_type][task_id]['active'] = False

        # Update the task counts
        for task_type in subnet_addresses.keys():
            active_tasks = sum(1 for task in tasks.get(task_type, {}).values() if task['active'])
            solver_selected_tasks = sum(1 for task in tasks.get(task_type, {}).values() if task['solver_selected'] and task['active'])
            task_counts[task_type] = (solver_selected_tasks, active_tasks)

        await asyncio.sleep(0.5)  # Polling interval

async def set_interval_mining(web3, interval):
    """Set the mining interval on the Ethereum node."""
    await web3.provider.make_request('evm_setIntervalMining', [interval])

def run_monitor_processes(stdscr, db_adapter, job_id, task_counts):
    asyncio.run(monitor_processes(stdscr, db_adapter, job_id, task_counts))

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
            'python', '-m', 'spl.db.db_server',
            '--host', DB_HOST,
            '--port', DB_PORT,
            '--perm', str(GUESS_DB_PERM_ID),
            '--root_wallet', public_key
        ],
        stdout=db_log, stderr=db_log, cwd=package_root_dir
    )
    await wait_for_health(db_url)
    await db_adapter.create_instance("db", ServiceType.Db.name, None, args.private_key, '', str(db_process.pid))
    logging.info(f"DB instance started with PID {db_process.pid}")

    try:
        delete_old_tensor_files(STATE_DIR, BLOCK_TIMESTAMPS_FILE, LAST_FUTURE_VERSION_FILE)
        delete_directory_contents(TEMP_DIR)

        # Start anvil process
        logging.info("Starting anvil...")
        anvil_log = open(ANVIL_LOG_FILE, 'w')
        anvil_process = subprocess.Popen(['anvil'], stdout=anvil_log, stderr=anvil_log, cwd=package_root_dir)
        instance_id = await db_adapter.create_instance("anvil", ServiceType.Anvil.name, None, args.private_key, '', str(anvil_process.pid))
        logging.info(f"Anvil started with PID {anvil_process.pid}")

        # Start Flask server in a separate thread
        flask_thread = threading.Thread(target=lambda: app.run(port=5002))
        flask_thread.start()

        logging.info("Starting deployment...")

        os.environ['SUBNET_ADDRESSES_JSON'] = args.subnet_addresses
        os.environ['PANTHALIA_DEPLOYMENT'] = args.deployment_config
        os.environ['SOT_URL'] = args.sot_url

        logging.info(f'Time in string: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
        
        web3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(RPC_URL))
        if not await wait_for_rpc_available(web3):
            exit(1)

        await set_interval_mining(web3, 1)

        # Run the deployment command
        deploy_command = [
            'forge', 'script', os.path.basename(args.forge_script),
            '--broadcast', '--rpc-url', RPC_URL,
            '--private-key', args.private_key, '-vv'
        ]
        subprocess.run(deploy_command, cwd=os.path.dirname(args.forge_script), check=True)

        logging.info("Deployment completed successfully.")

        with open(args.subnet_addresses, 'r') as file:
            subnet_addresses = json.load(file)

        with open(args.deployment_config, 'r') as file:
            deployment_config = json.load(file)

        pool_address = deployment_config['pool']
        distributor_contract_address = deployment_config['distributor']

        deployer_account = web3.eth.account.from_key(args.private_key)
        deployer_address = deployer_account.address
        pool_contract = web3.eth.contract(address=pool_address, abi=load_abi('Pool'))
        token_address = await pool_contract.functions.token().call()
        token_contract = web3.eth.contract(address=token_address, abi=load_abi('ERC20'))

        async with aiofiles.open(plugin_file, mode='r') as f:
            code = await f.read()
        plugin_id = await db_adapter.create_plugin('plugin', code)
        
        subnet_id = await db_adapter.create_subnet(
            list(subnet_addresses.values())[0],
            RPC_URL,
            distributor_contract_address,
            pool_address,
            token_address,
            args.group
        )
        
        job_id = await db_adapter.create_job('test_job', plugin_id, subnet_id, args.sot_url, 0)

        db_perm_id = await db_adapter.create_perm_description(PermType.ModifyDb.name)
        
        assert db_perm_id == GUESS_DB_PERM_ID, f"Expected db_perm_id {GUESS_DB_PERM_ID}, got {db_perm_id}"

        plugin = await get_plugin(plugin_id, db_adapter)


        await db_adapter.create_perm(args.private_key, db_perm_id)

        try:
            logging.info("Starting master process...")

            master_log = open(os.path.join(LOG_DIR, 'master.log'), 'w')
            master_command = [
                'python', '-m', 'spl.master',
                '--private_key', args.private_key,
                '--max_concurrent_iterations', str(plugin.max_concurrent_iterations),
                '--db_url', db_url,
                '--num_workers', str(args.worker_count),
                '--deploy_type', 'local',
            ]
            if args.detailed_logs:
                master_command.append('--detailed_logs')
            if args.torch_compile:
                master_command.append('--torch_compile')
            master_process = subprocess.Popen(master_command, stdout=master_log, stderr=master_log, cwd=package_root_dir)
            await db_adapter.create_instance("master", ServiceType.Master.name, None, args.private_key, '', str(master_process.pid))
            logging.info(f"Started master process with command: {' '.join(master_command)}")

            logging.info("Master process started.")

            curses_thread = threading.Thread(target=curses.wrapper, args=(run_monitor_processes, db_adapter, job_id, task_counts))
            curses_thread.start()
        except Exception as e:
            logging.error(f"Error: {e}", exc_info=True)
            await terminate_processes(db_adapter, job_id)
            exit(1)
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        await terminate_processes(db_adapter, job_id)
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
