import subprocess
import json
import os
import time
import argparse
import requests
import threading
import curses
from ..common import load_abi, wait_for_rpc_available, wait_for_sot, SOT_PRIVATE_PORT, fund_wallets, MAX_CONCURRENT_ITERATIONS, DB_PORT
from ..db.db_adapter_client import DBAdapterClient
from ..models import init_db, db_path, PermType
from ..plugin_manager import get_plugin
from web3 import AsyncWeb3, Web3
from eth_account import Account
from .cloud_adapters.runpod import launch_instance_and_record_logs, terminate_all_pods, get_public_ip_and_port, INPUT_JSON_PATH
import glob
from .cloud_adapters.runpod_config import BASE_TEMPLATE_ID
import shutil
import asyncio
import logging
import traceback
import socket
import shlex
import signal
from .util import is_port_open
import paramiko
import runpod
import aiofiles
from .cloud_adapters.runpod import get_pod_ssh_ip_port, reconnect_and_initialize_existing_pod

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Go one level above the parent directory to get the package's root directory
package_root_dir = os.path.dirname(parent_dir)

DATA_DIR = os.path.join(parent_dir, 'data')
STATE_DIR = os.path.join(DATA_DIR, 'state')
TEMP_DIR = os.path.join(STATE_DIR, 'temp')
DEPLOY_SCRIPT = os.path.join(parent_dir, 'script', 'Deploy.s.sol')
LOG_DIR = os.path.join(parent_dir, 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'test_run.log')
BLOCK_TIMESTAMPS_FILE = os.path.join(STATE_DIR, 'block_timestamps.json')
STATE_FILE = os.path.join(STATE_DIR, 'state.json')  # State file to save/load state
plugin_file = os.path.join(parent_dir, 'plugins', 'plugin.py')
REMOTE_MODEL_FILE = '/app/spl/data/state/model.pt'
LOCAL_MODEL_FILE = os.path.join(parent_dir, 'data', 'state', 'model.pt')

GUESSED_SUBNET_ID = 1
GUESSED_PLUGIN_ID = 1
GUESS_DB_PERM_ID = 1

DOCKER_IMAGE = 'runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04'
GPU_TYPE = 'NVIDIA GeForce RTX 4090'
with open(os.path.join(script_dir, 'env_setup.sh'), 'r') as f:
    DOCKER_CMD = f.read()

with open(os.path.join(script_dir, 'anvil_setup.sh'), 'r') as f:
    ANVIL_CMD = f.read()

# Configure logging to file and stdout
os.makedirs(LOG_DIR, exist_ok=True)
file_handler = logging.FileHandler(LOG_FILE)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger().addHandler(file_handler)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
for handler in logging.getLogger().handlers:
    handler.setFormatter(formatter)

def get_log_file(instance_name):
    """
    Helper function to get the log file path based on the instance name.
    """
    return os.path.join(LOG_DIR, f"{instance_name}.log")

def parse_args():
    parser = argparse.ArgumentParser(description="Test run script for starting workers and master")
    parser.add_argument('--subnet_addresses', type=str, required=True, help="Path to the subnet addresses JSON file")
    parser.add_argument('--deployment_config', type=str, required=True, help="Path to the deployment configuration JSON file")
    parser.add_argument('--private_key', type=str, required=True, help="Private key of the deployer's Ethereum account")
    parser.add_argument('--group', type=int, required=True, help="Group for depositing stake")
    parser.add_argument('--local_storage_dir', type=str, default='data', help="Directory for local storage of files")
    parser.add_argument('--forge_script', type=str, default=DEPLOY_SCRIPT, help="Path to the Forge deploy script")
    parser.add_argument('--backend', type=str, default='nccl', help="Distributed backend to use (default: nccl, use 'gloo' for macOS)")
    parser.add_argument('--detailed_logs', action='store_true', help="Enable detailed logs for all processes")
    parser.add_argument('--num_master_wallets', type=int, default=70, help="Number of wallets to generate for the master process")
    parser.add_argument('--worker_count', type=int, default=1, help="Number of workers to start")
    parser.add_argument('--torch_compile', action='store_true', help="Enable torch.compile and model warmup")
    parser.add_argument('--terminate', action='store_true', help="Allow termination of all running pods")
    return parser.parse_args()

args = parse_args()

latest_loss_cache = {
    'value': None,
    'last_fetched': 0
}

sot_url = None
rpc_url = None

LOSS_REFRESH_INTERVAL = 60  # For example, update every 60 seconds

def get_public_ip():
    # Function to retrieve the public IP address of the machine
    try:
        public_ip = requests.get('https://api.ipify.org').text
        return public_ip
    except requests.RequestException as e:
        logging.error(f"Unable to retrieve public IP: {e}")
        return None

async def wait_for_workers_to_sync(worker_count, timeout=600):
    global sot_url
    start_time = time.time()
    get_num_workers_url = os.path.join(sot_url, 'get_num_synced')
    while time.time() - start_time < timeout:
        try:
            response = requests.get(get_num_workers_url)
            synced_workers = response.json()
            logging.debug(f"Synced {synced_workers}/{worker_count} workers.")
            if synced_workers >= worker_count:
                logging.debug("All workers have synced.")
                return True
        except requests.RequestException as e:
            logging.error(f"Error checking worker sync status: {e}")
        await asyncio.sleep(2)
    logging.debug("Timeout waiting for workers to sync.")
    return False

def generate_wallets(num_wallets):
    wallets = []
    for _ in range(num_wallets):
        account = Account.create()
        wallets.append({'private_key': account._private_key.hex(), 'address': account.address})
    return wallets

def delete_old_tensor_files(directory, timestamps_file):
    if not os.path.exists(directory):
        return

    if not os.path.exists(timestamps_file):
        return

    with open(timestamps_file, 'r') as f:
        block_timestamps = json.load(f)

    tensor_files = glob.glob(os.path.join(directory, '*.pt'))
    latest_files = {f"{name}_{version}.pt" for name, version in block_timestamps.items()}

    for tensor_file in tensor_files:
        if os.path.basename(tensor_file) not in latest_files:
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

def terminate_processes():
    """Terminate all running pods and remove the state file."""
    if not args.terminate:
        logging.info("Pod termination disabled. Exiting without terminating pods.")
        return
    terminate_all_pods()
    # Remove the state file upon termination
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        logging.info(f"State file {STATE_FILE} deleted successfully.")

def load_or_prompt_state():
    """Check if a state file exists and prompt whether to load or delete it."""
    if os.path.exists(STATE_FILE):
        while True:
            choice = input(f"State file {STATE_FILE} exists. Do you want to load it (L) or delete it (D)? [L/D]: ").strip().lower()
            if choice == 'l':
                return load_state()
            elif choice == 'd':
                os.remove(STATE_FILE)
                logging.info(f"State file {STATE_FILE} deleted successfully.")
                return {'pods': {}, 'deployscript_run': False}
            else:
                print("Invalid choice. Please enter 'L' to load or 'D' to delete.")
    else:
        return {'pods': {}, 'deployscript_run': False}

def signal_handler(signal_received, frame):
    logging.info("SIGINT received, shutting down...")
    terminate_processes()
    os._exit(0)  # Force exit the program

# Register the signal handler for SIGINT
signal.signal(signal.SIGINT, signal_handler)

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

def fetch_latest_loss():
    global latest_loss_cache, sot_url
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
            # In case of an error, do not update the last fetched time to retry on next fetch

    return latest_loss_cache['value']

def write_private_key_to_temp(private_key, temp_dir):
    import tempfile
    import os
    fd, path = tempfile.mkstemp(dir=temp_dir)
    with os.fdopen(fd, 'w') as tmp:
        tmp.write(private_key)
    return path

def monitor_processes(stdscr, processes, pod_helpers, task_counts):
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

    max_name_length = max(len(name) for name in processes.keys()) + 14
    right_col_width = max_name_length + 2

    def draw_screen():
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        split_point = width - right_col_width

        # Order processes by name
        ordered_process_names = sorted(processes.keys(), key=lambda name: (
            name.startswith('worker_final_logits'),
            name.startswith('worker_forward'),
            name.startswith('worker_embed'),
            not name.startswith('worker'),
            name
        ))

        # Display logs on the left side
        process_name = ordered_process_names[selected_process]
        log_file = get_log_file(process_name)  # Inferred log file path
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
            is_selected = (i == selected_process)
            status = pod_helpers.get(name, {}).get('is_ssh_session_alive', lambda: False)()
            color = curses.color_pair(1) if status else curses.color_pair(2)
            indicator = '*' if is_selected else ' '

            stdscr.addstr(i, split_point, f"{indicator} {name}", color)

        # Fetch and display the latest loss
        latest_loss = fetch_latest_loss()
        loss_display = f"Latest Loss: {latest_loss:.3f}" if latest_loss is not None else "Latest Loss: N/A"
        loss_y = height - len(task_counts) - 5
        stdscr.addstr(loss_y, split_point, loss_display, curses.color_pair(4))

        # Draw task counts below the latest loss
        task_start = height - 3 - len(task_counts)
        for i, (task_type, (solver_selected, active)) in enumerate(task_counts.items()):
            stdscr.addstr(task_start + i, split_point, f"{task_type}: {solver_selected}/{active}", curses.color_pair(3))

        # Instructions
        #stdscr.addstr(height - 1, 0, "Use arrow keys to navigate. Press 'q' to quit and terminate pods, 'e' to exit without terminating.", curses.A_BOLD)
        stdscr.addstr(height - 1, split_point, "PANTHALIA SIMULATOR V0", curses.color_pair(3))
        stdscr.refresh()

    draw_screen()  # Initial draw

    while True:
        key = stdscr.getch()
        if key == curses.KEY_UP:
            selected_process = (selected_process - 1) % len(processes)
            draw_screen()
        elif key == curses.KEY_DOWN:
            selected_process = (selected_process + 1) % len(processes)
            draw_screen()
        elif key == curses.KEY_RESIZE:
            last_resize = time.time()
        elif key == ord('s'):
            # Handle the "s" key press - SSH download model.pt from the SOT instance
            try:
                logging.info(f"Attempting to download {REMOTE_MODEL_FILE} to {LOCAL_MODEL_FILE}...")

                # Get the SSH client for the SOT instance from pod_helpers
                sftp = pod_helpers['sot']['sftp']
                sftp.get(REMOTE_MODEL_FILE, LOCAL_MODEL_FILE)
                logging.info(f"Downloaded model: {LOCAL_MODEL_FILE}")
            except Exception as e:
                logging.error(f"Failed to download model: {e}")
            draw_screen()
        elif key == ord('q'):
            terminate_processes()
            # Remove the state file upon termination
            if os.path.exists(STATE_FILE):
                os.remove(STATE_FILE)
            break
        elif key == ord('e'):
            # Exit without terminating pods
            logging.info("Exiting monitoring interface without terminating pods.")
            break

        if last_resize and time.time() - last_resize > 0.1:
            draw_screen()
            last_resize = None

        draw_screen()
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
                try:
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
                except Exception as e:
                    logging.error(f"Error processing events for {task_type}: {e}")

        # Update the task counts
        for task_type in subnet_addresses.keys():
            active_tasks = sum(1 for task in tasks.get(task_type, {}).values() if task['active'])
            solver_selected_tasks = sum(1 for task in tasks.get(task_type, {}).values() if task['solver_selected'] and task['active'])
            task_counts[task_type] = (solver_selected_tasks, active_tasks)

        await asyncio.sleep(0.5)  # Polling interval

async def set_interval_mining(web3, interval):
    """Set the mining interval on the Ethereum node."""
    await web3.provider.make_request('evm_setIntervalMining', [interval])

async def launch_worker(worker_idx, subnet_addresses, worker_wallets, token_contract, pool_address, state, db_url):
    global sot_url, rpc_url
    # Define environment variables for the worker
    this_worker_wallets = worker_wallets[worker_idx * len(subnet_addresses):(worker_idx + 1) * len(subnet_addresses)]

    env = {
        'GITHUB_TOKEN': os.environ.get('GITHUB_TOKEN', ''),
        'SERVICE_TYPE': 'worker',
        'RANK': '0',
        'WORLD_SIZE': '1',
        'TASK_TYPES': '+'.join(list(subnet_addresses.keys())),
        'SUBNET_ADDRESSES': '+'.join(list(subnet_addresses.values())),
        'PRIVATE_KEYS': '+'.join([x['private_key'] for x in this_worker_wallets]),
        'RPC_URL': rpc_url,
        'SOT_URL': sot_url,
        'POOL_ADDRESS': pool_address,
        'GROUP': str(args.group),
        'LOCAL_STORAGE_DIR': args.local_storage_dir,
        'BACKEND': args.backend,
        'DB_URL': db_url,
    }

    if args.torch_compile:
        env['TORCH_COMPILE'] = 'true'

    worker_name = f'worker_{worker_idx}'

    # Launch the worker instance and save state
    worker_instance, worker_helpers = await launch_instance_and_record_logs(
        name=worker_name,
        gpu_type=GPU_TYPE,
        image=DOCKER_IMAGE,
        gpu_count=1,
        ports='',
        log_file=get_log_file(worker_name),  # Inferred log file path
        env=env,
        template_id=BASE_TEMPLATE_ID,
        cmd=DOCKER_CMD
    )

    # Save the private key value instead of the path to the state
    state['pods'][worker_name] = {
        'pod_id': worker_instance['id'],
        'private_key': env['PRIVATE_KEYS'],
        # 'log_file': get_log_file(worker_name)  # Removed as per requirement
    }
    save_state(state)

    logging.info(f"Started worker process {worker_idx} for tasks on instance {worker_instance['id']} with env {env}")

    return worker_name, worker_instance, worker_helpers

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    else:
        logging.error(f"State file {STATE_FILE} does not exist.")
        return {'pods': {}, 'deployscript_run': False}

def save_state(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

def check_guessed_ids(guessed_id, actual_id, name):
    if guessed_id != actual_id:
        logging.error(f"Expected {name} ID {actual_id}, got {guessed_id}")
        exit(1)

async def launch_db_instance(state, db_adapter, db_perm_id):
    global sot_url, rpc_url
    # Check if the DB server is already running
    if 'db' not in state['pods']:
        logging.info("Starting DB instance...")
        public_key = Account.from_key(args.private_key).address
        env = {
            'GITHUB_TOKEN': os.environ.get('GITHUB_TOKEN', ''),
            'SERVICE_TYPE': 'db',
            'DB_HOST': '0.0.0.0',
            'DB_PORT': DB_PORT,
            'DB_PERM': str(db_perm_id),
            'ROOT_WALLET': public_key
        }

        # Start the DB service on a remote instance
        db_instance, db_helpers = await launch_instance_and_record_logs(
            name="db",
            gpu_type=GPU_TYPE,
            container_disk_in_gb=20,  # Set disk size appropriately for DB instance
            image=DOCKER_IMAGE,
            gpu_count=0,  # No GPU required for the DB instance
            ports=f'{DB_PORT}/tcp',  # Expose the DB port
            log_file=get_log_file("db"),  # Inferred log file path
            template_id=BASE_TEMPLATE_ID,
            cmd=DOCKER_CMD,
            env=env
        )

        # Save the DB instance to state with the private key value
        state['pods']['db'] = {
            'pod_id': db_instance['id'],
            'private_key': db_helpers['private_key'],
            # 'log_file': get_log_file("db")  # Removed as per requirement
        }
        db_ip, db_port = await get_public_ip_and_port(db_instance['id'], private_port=int(DB_PORT))
        state['db_url'] = f"http://{db_ip}:{db_port}"
        save_state(state)
        logging.info(f"DB service started on {state['db_url']}")

        return db_instance, db_helpers
    else:
        logging.info("DB instance already running. Reconnecting...")
        pod_id = state['pods']['db']['pod_id']
        private_key = state['pods']['db']['private_key']
        temp_path = write_private_key_to_temp(private_key, TEMP_DIR)
        db_helpers = await reconnect_and_initialize_existing_pod(
            pod_id, 'db', temp_path, log_file=get_log_file("db")  # Inferred log file path
        )
        processes['db'] = runpod.get_pod(pod_id)
        logging.info(f"Reconnected to DB at {state['db_url']}")

        return runpod.get_pod(pod_id), db_helpers

async def main():
    global sot_url, rpc_url
    processes = {}
    task_counts = {}  # Dictionary to store task counts
    pod_helpers = {}

    # Use the new function to load state or prompt to delete
    state = load_or_prompt_state()

    # Retrieve the public IP address of the machine
    public_ip = get_public_ip()
    if not public_ip:
        logging.error("Could not retrieve public IP address.")
        exit(1)

    # Reset logs
    reset_logs(LOG_DIR)
    master_wallets = generate_wallets(args.num_master_wallets)

    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    
    db_instance, db_helpers = await launch_db_instance(state, db_adapter, GUESSED_DB_PERM_ID)
    db_url = state['db_url']
    db_adapter = DBAdapterClient(
        db_url,
        args.private_key,
    )

    # First, launch or reconnect the DB instance
    db_perm_id = await db_adapter.create_perm_description(PermType.ModifyDb.name)
    check_guessed_ids(db_perm_id, GUESS_DB_PERM_ID, 'DB Perm')

    # Now launch the SOT service, which will use the `db_url`
    sot_promise = None
    if 'sot' not in state['pods']:
        logging.info("Starting SOT instance...")

        # Assuming `create_job` and `create_sot` are async functions
        job_id = await db_adapter.create_job('test_job', GUESSED_PLUGIN_ID, GUESSED_SUBNET_ID, sot_url, 0)
        sot_id = await db_adapter.create_sot(job_id, sot_url)
        sot_wallet = generate_wallets(1)[0]

        await db_adapter.create_perm(sot_wallet['address'], db_perm_id)

        env = {
            'GITHUB_TOKEN': os.environ.get('GITHUB_TOKEN', ''),
            'SERVICE_TYPE': 'sot',
            'RANK': '0',
            'WORLD_SIZE': '1',
            'SOT_PRIVATE_PORT': str(SOT_PRIVATE_PORT),
            'SOT_ID': str(sot_id),
            'DB_URL': db_url,
            'PRIVATE_KEY': sot_wallet['private_key'],
        }

        logging.info(f'Environment variables for SOT: {env}')

        # Start the SOT service on a remote instance
        sot_promise = launch_instance_and_record_logs(
            name="sot",
            gpu_type=GPU_TYPE,
            container_disk_in_gb=40,
            image=DOCKER_IMAGE,
            gpu_count=1,
            ports=f'{SOT_PRIVATE_PORT}/tcp',
            log_file=get_log_file("sot"),  # Inferred log file path
            template_id=BASE_TEMPLATE_ID,
            cmd=DOCKER_CMD,
            env=env
        )

    if 'anvil' not in state['pods']:
        logging.info("Starting Anvil instance...")

        anvil_instance, anvil_helpers = await launch_instance_and_record_logs(
            name="anvil",
            gpu_count=0,
            ports='8545/tcp',
            log_file=get_log_file("anvil"),  # Inferred log file path
            template_id=BASE_TEMPLATE_ID,
            cmd=ANVIL_CMD,
            env={},
            timeout=300
        )
        pod_helpers['anvil'] = anvil_helpers
        anvil_ip, anvil_port = await get_public_ip_and_port(anvil_instance['id'], private_port=8545)
        rpc_url = f"http://{anvil_ip}:{anvil_port}"
        processes['anvil'] = anvil_instance
        logging.info(f"Anvil started on {rpc_url}")

        # Save Anvil instance to state with the private key value
        state['pods']['anvil'] = {
            'pod_id': anvil_instance['id'],
            'private_key': anvil_helpers['private_key'],
            # 'log_file': get_log_file("anvil")  # Removed as per requirement
        }
        state['rpc_url'] = rpc_url
        save_state(state)

        # Wait until the Anvil port is open
        logging.info(f"Waiting for Anvil to open port {anvil_port}...")
        while not is_port_open(anvil_ip, anvil_port):
            logging.info("Anvil port not open yet. Retrying in 1 second...")
            await asyncio.sleep(1)
        logging.info("Anvil port is now open.")

        # Delete all .pt files in the state directory except for the latest version for each tensor
        delete_old_tensor_files(STATE_DIR, BLOCK_TIMESTAMPS_FILE)

        # Delete the temp directory
        delete_directory_contents(TEMP_DIR)

        # Print initial stage
        logging.info("Starting deployment...")

        # Set environment variables for deployment
        os.environ['SUBNET_ADDRESSES_JSON'] = args.subnet_addresses
        os.environ['PANTHALIA_DEPLOYMENT'] = args.deployment_config

        web3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))

        # Wait for the RPC to be available before proceeding
        if not await wait_for_rpc_available(web3):
            exit(1)
        await set_interval_mining(web3, 1)

        # Run Deploy.s.sol script from the correct path
        deploy_command = [
            'forge', 'script', os.path.basename(args.forge_script),
            '--broadcast', '--rpc-url', rpc_url,
            '--private-key', args.private_key, '-vv'
        ]
        subprocess.run(deploy_command, cwd=os.path.dirname(args.forge_script), check=True)

        # Print deployment stage completion
        logging.info("Deployment completed successfully, loading JSON files...")

        logging.info('JSONs loaded, parsing deployment config...')

        # Load subnet_addresses and deployment_config
        with open(args.subnet_addresses, 'r') as file:
            subnet_addresses = json.load(file)

        with open(args.deployment_config, 'r') as file:
            deployment_config = json.load(file)
        
        state['subnet_addresses'] = subnet_addresses
        state['deployment_config'] = deployment_config

        distributor_contract_address = deployment_config['distributor']
        pool_address = deployment_config['pool']

        deployer_account = web3.eth.account.from_key(args.private_key)
        deployer_address = deployer_account.address
        pool_contract = web3.eth.contract(address=pool_address, abi=load_abi('Pool'))
        token_address = await pool_contract.functions.token().call()
        token_contract = web3.eth.contract(address=token_address, abi=load_abi('ERC20'))

        await init_db()
        
        async with aiofiles.open(plugin_file, mode='r') as f:
            code = await f.read()
        plugin_id = await db_adapter.create_plugin('plugin', code)
        
        subnet_id = await db_adapter.create_subnet(
            list(subnet_addresses.values())[0],
            rpc_url,
            distributor_contract_address,
            pool_address,
        )
    
        plugin = await get_plugin(plugin_id, db_adapter)

        check_guessed_ids(subnet_id, GUESSED_SUBNET_ID, 'Subnet')
        check_guessed_ids(plugin_id, GUESSED_PLUGIN_ID, 'Plugin')

        sot_perm_id = (await db_adapter.get_sot(sot_id)).perm
        authorized_master_wallet = master_wallets[0]
        await db_adapter.create_perm(authorized_master_wallet['address'], sot_perm_id)
        await db_adapter.create_perm(authorized_master_wallet['address'], db_perm_id)

        logging.info('Generating wallets')

        await fund_wallets(web3, args.private_key, master_wallets, deployer_address, token_contract, 1, 10000 * 10**18, distributor_contract_address)
        
        # Generate wallets for workers and fund them
        worker_wallets = generate_wallets(args.worker_count * len(subnet_addresses))
        await fund_wallets(web3, args.private_key, worker_wallets, deployer_address, token_contract, 1, 10000 * 10**18, distributor_contract_address)

        state['deployscript_run'] = True
        save_state(state)
    else:
        logging.info("Anvil instance already running. Reconnecting...")
        pod_id = state['pods']['anvil']['pod_id']
        private_key = state['pods']['anvil']['private_key']
        temp_path = write_private_key_to_temp(private_key, TEMP_DIR)
        pod_helpers['anvil'] = await reconnect_and_initialize_existing_pod(
            pod_id, 'anvil', temp_path, log_file=get_log_file("anvil")  # Inferred log file path
        )
        processes['anvil'] = runpod.get_pod(pod_id)
        rpc_url = state['rpc_url']
        logging.info(f"Reconnected to Anvil at {rpc_url}")
        logging.info("Deployment script has already been run. Skipping deployment.")
        subnet_addresses = state['subnet_addresses']

        deployment_config = state['deployment_config']
        web3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
        pool_address = deployment_config['pool']
        pool_contract = web3.eth.contract(address=pool_address, abi=load_abi('Pool'))
        token_address = await pool_contract.functions.token().call()
        token_contract = web3.eth.contract(address=token_address, abi=load_abi('ERC20'))

        # Load worker wallets
        worker_wallets = generate_wallets(args.worker_count * len(subnet_addresses))

    # Reconnect or start SOT after deployment
    if 'sot' not in state['pods']:
        sot_instance, sot_helpers = await sot_promise
        pod_helpers['sot'] = sot_helpers
        sot_ip, sot_port = await get_public_ip_and_port(sot_instance['id'], private_port=SOT_PRIVATE_PORT)
        sot_url = f"http://{sot_ip}:{sot_port}"
        processes['sot'] = sot_instance
        logging.info(f"SOT service started on {sot_url}")

        # Save SOT instance to state with the private key value
        state['pods']['sot'] = {
            'pod_id': sot_instance['id'],
            'private_key': sot_helpers['private_key'],
            # 'log_file': get_log_file("sot")  # Removed as per requirement
        }
        state['sot_url'] = sot_url
        save_state(state)

        # Wait for the SOT service to be available
        if not await wait_for_sot(sot_url):
            logging.error("Error: SOT service did not become available within the timeout period.")
            exit(1)
    else:
        logging.info("SOT instance already running. Reconnecting...")
        pod_id = state['pods']['sot']['pod_id']
        private_key = state['pods']['sot']['private_key']
        temp_path = write_private_key_to_temp(private_key, TEMP_DIR)
        pod_helpers['sot'] = await reconnect_and_initialize_existing_pod(
            pod_id, 'sot', temp_path, log_file=get_log_file("sot")  # Inferred log file path
        )
        processes['sot'] = runpod.get_pod(pod_id)
        sot_url = state['sot_url']

    # Print worker initialization stage
    logging.info("Starting worker processes...")

    worker_tasks = []
    for worker_idx in range(args.worker_count):
        worker_name = f'worker_{worker_idx}'
        if worker_name not in state['pods']:
            worker_tasks.append(
                launch_worker(
                    worker_idx,
                    subnet_addresses,
                    worker_wallets,
                    token_contract,
                    pool_address,
                    state,
                    db_url
                )
            )
        else:
            logging.info(f"Worker {worker_name} already running. Reconnecting...")
            pod_id = state['pods'][worker_name]['pod_id']
            private_key = state['pods'][worker_name]['private_key']
            temp_path = write_private_key_to_temp(private_key, TEMP_DIR)
            pod_helpers[worker_name] = await reconnect_and_initialize_existing_pod(
                pod_id, worker_name, temp_path, log_file=get_log_file(worker_name)  # Inferred log file path
            )
            processes[worker_name] = runpod.get_pod(pod_id)

    try:
        # Print master initialization stage
        if 'master' not in state['pods']:
            logging.info("Starting master process...")

            env = {
                'GITHUB_TOKEN': os.environ.get('GITHUB_TOKEN', ''),
                'SERVICE_TYPE': 'master',
                'RANK': '0',
                'WORLD_SIZE': '1',
                'RPC_URL': rpc_url,
                'WALLETS': INPUT_JSON_PATH + '_0',
                'SOT_URL': sot_url,
                'SUBNET_ADDRESSES': INPUT_JSON_PATH + '_1',
                'MAX_CONCURRENT_ITERATIONS': MAX_CONCURRENT_ITERATIONS,
                'JOB_ID': str(job_id),
                'DB_URL': db_url,
            }

            if args.torch_compile:
                env['TORCH_COMPILE'] = 'true'

            # Start master.py on a remote instance
            master_instance, master_helpers = await launch_instance_and_record_logs(
                name="master",
                gpu_count=0,
                ports='',
                log_file=get_log_file("master"),  # Inferred log file path
                env=env,
                template_id=BASE_TEMPLATE_ID,
                cmd=DOCKER_CMD,
                input_jsons=[master_wallets, subnet_addresses]
            )
            pod_helpers['master'] = master_helpers
            processes['master'] = master_instance
            logging.info(f"Master process started on instance {master_instance['id']}")

            # Save master instance to state with the private key value
            state['pods']['master'] = {
                'pod_id': master_instance['id'],
                'private_key': master_helpers['private_key'],
                # 'log_file': get_log_file("master")  # Removed as per requirement
            }
            save_state(state)
        else:
            logging.info("Master instance already running. Reconnecting...")
            pod_id = state['pods']['master']['pod_id']
            private_key = state['pods']['master']['private_key']
            temp_path = write_private_key_to_temp(private_key, TEMP_DIR)
            pod_helpers['master'] = await reconnect_and_initialize_existing_pod(
                pod_id, 'master', temp_path, log_file=get_log_file("master")  # Inferred log file path
            )
            processes['master'] = runpod.get_pod(pod_id)

        if worker_tasks:
            worker_results = await asyncio.gather(*worker_tasks)

            # Unpack worker results
            for worker_name, worker_instance, worker_helpers in worker_results:
                pod_helpers[worker_name] = worker_helpers
                processes[worker_name] = worker_instance

        # Start the curses interface in a new thread
        curses_thread = threading.Thread(target=curses.wrapper, args=(monitor_processes, processes, pod_helpers, task_counts))
        curses_thread.start()

        # Run the task tracking in an asyncio loop
        await track_tasks(web3, subnet_addresses, pool_contract, task_counts)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        traceback.print_exc()
        terminate_processes()
        raise e

if __name__ == "__main__":
    asyncio.run(main())
