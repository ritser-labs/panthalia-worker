import subprocess
import json
import os
import time
import argparse
import requests
import threading
import curses
from flask import Flask, jsonify
from ..common import (
    load_abi,
    wait_for_health,
    SOT_PRIVATE_PORT,
    DB_PORT,
    generate_wallets
)
from ..db.db_adapter_client import DBAdapterClient
from ..models import (
    init_db,
    db_path,
    PermType,
    ServiceType
)
from ..plugin_manager import get_plugin
from web3 import AsyncWeb3, Web3
from eth_account import Account
from .cloud_adapters.runpod import (
    launch_instance_and_record_logs,
    terminate_all_pods,
    get_public_ip_and_port,
    INPUT_JSON_PATH,
    BASE_TEMPLATE_ID,
)
import glob
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

# Define directories and paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
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

# Read Docker and Anvil setup commands
with open(os.path.join(script_dir, 'env_setup.sh'), 'r') as f:
    DOCKER_CMD = f.read()

with open(os.path.join(script_dir, 'anvil_setup.sh'), 'r') as f:
    ANVIL_CMD = f.read()

# Configure logging to file and stdout
os.makedirs(LOG_DIR, exist_ok=True)
file_handler = logging.FileHandler(LOG_FILE)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
if not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
    logger.addHandler(file_handler)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
for handler in logger.handlers:
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
    parser.add_argument('--detailed_logs', action='store_true', help="Enable detailed logs for all processes")
    parser.add_argument('--num_master_wallets', type=int, default=70, help="Number of wallets to generate for the master process")
    parser.add_argument('--worker_count', type=int, default=1, help="Number of workers to start")
    parser.add_argument('--torch_compile', action='store_true', help="Enable torch.compile and model warmup")
    parser.add_argument('--terminate', action='store_true', help="Allow termination of all running pods")
    return parser.parse_args()

args = None

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

def terminate_processes(db_adapter, job_id):
    """Terminate all processes associated with the given job_id via the DB."""
    asyncio.run(async_terminate_processes(db_adapter, job_id))

async def async_terminate_processes(db_adapter, job_id):
    instances = await db_adapter.get_instances_by_job(job_id)
    if not instances:
        logging.info("No instances to terminate.")
        return
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

def write_private_key_to_temp(private_key, temp_dir):
    import tempfile
    fd, path = tempfile.mkstemp(dir=temp_dir)
    with os.fdopen(fd, 'w') as tmp:
        tmp.write(private_key)
    return path

async def monitor_processes(stdscr, db_adapter, job_id, task_counts):
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

    while True:
        try:
            height, width = stdscr.getmaxyx()
            split_point = width - 50  # Adjusted for right column width

            instances = await db_adapter.get_all_instances()
            if not instances:
                ordered_process_names = []
            else:
                ordered_process_names = sorted([instance.name for instance in instances], key=lambda name: (
                    name.startswith('worker_final_logits'),
                    name.startswith('worker_forward'),
                    name.startswith('worker_embed'),
                    not name.startswith('worker'),
                    name
                ))

            if selected_process >= len(ordered_process_names):
                selected_process = max(0, len(ordered_process_names) - 1)

            # Display logs on the left side
            if ordered_process_names:
                process_name = ordered_process_names[selected_process]
                log_file = get_log_file(process_name)
                log_lines = []

                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        log_lines.extend(f.readlines())

                for i, line in enumerate(log_lines[-(height - 2):]):
                    try:
                        stdscr.addstr(i, 0, line[:split_point - 2])
                    except curses.error:
                        pass  # Ignore if the line doesn't fit

            # Draw the separator line
            for y in range(height):
                try:
                    stdscr.addch(y, split_point - 2, curses.ACS_VLINE)
                except curses.error:
                    pass  # Ignore if the position is out of bounds

            # Display process list on the right side
            for i, name in enumerate(ordered_process_names):
                is_selected = (i == selected_process)
                instance = next((inst for inst in instances if inst.name == name), None)
                if instance:
                    pid = int(instance.process_id) if instance.process_id.isdigit() else -1
                    status = (pid > 0) and (os.path.exists(f"/proc/{pid}"))
                    color = curses.color_pair(1) if status else curses.color_pair(2)
                    indicator = '*' if is_selected else ' '
                    try:
                        stdscr.addstr(i, split_point, f"{indicator} {name}", color)
                    except curses.error:
                        pass  # Ignore if the position is out of bounds

            # Fetch and display the latest loss
            if sot_url:
                latest_loss = fetch_latest_loss(sot_url)
                loss_display = f"Latest Loss: {latest_loss:.3f}" if latest_loss is not None else "Latest Loss: N/A"
                loss_y = height - len(task_counts) - 5
                try:
                    stdscr.addstr(loss_y, split_point, loss_display, curses.color_pair(4))
                except curses.error:
                    pass  # Ignore if the position is out of bounds

            # Draw task counts below the latest loss
            task_start = height - 3 - len(task_counts)
            for i, (task_type, (solver_selected, active)) in enumerate(task_counts.items()):
                try:
                    stdscr.addstr(task_start + i, split_point, f"{task_type}: {solver_selected}/{active}", curses.color_pair(3))
                except curses.error:
                    pass  # Ignore if the position is out of bounds

            # Footer
            try:
                stdscr.addstr(height - 1, split_point, "PANTHALIA SIMULATOR V0", curses.color_pair(3))
            except curses.error:
                pass  # Ignore if the position is out of bounds

            stdscr.refresh()

            # Handle key presses
            key = stdscr.getch()
            if key == curses.KEY_UP and ordered_process_names:
                selected_process = (selected_process - 1) % len(ordered_process_names)
            elif key == curses.KEY_DOWN and ordered_process_names:
                selected_process = (selected_process + 1) % len(ordered_process_names)
            elif key == curses.KEY_RESIZE:
                last_resize = time.time()
            elif key == ord('q'):
                await terminate_processes(db_adapter, job_id)
                break

            if last_resize and time.time() - last_resize > 0.1:
                last_resize = None

            await asyncio.sleep(0.05)
        except Exception as e:
            logging.error(f"Error in monitor_processes: {e}", exc_info=True)
            break

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


def load_state():
    """Load the state from the state file."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    else:
        logging.info("State file does not exist. Initializing new state.")
        return {}

def save_state(state):
    """Save the state to the state file."""
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)

def check_guessed_ids(guessed_id, actual_id, name):
    if guessed_id != actual_id:
        logging.error(f"Expected {name} ID {actual_id}, got {guessed_id}")
        exit(1)

async def launch_db_instance(state, db_adapter, is_reconnecting):
    """Launch the DB instance if not already running."""
    if not is_reconnecting:
        logging.info("Starting DB instance...")
        public_key = Account.from_key(args.private_key).address
        env = {
            'GITHUB_TOKEN': os.environ.get('GITHUB_TOKEN', ''),
            'SERVICE_TYPE': 'db',
            'DB_HOST': '0.0.0.0',
            'DB_PORT': DB_PORT,
            'DB_PERM': str(GUESS_DB_PERM_ID),
            'ROOT_WALLET': public_key
        }

        db_instance, db_helpers = await launch_instance_and_record_logs(
            name="db",
            gpu_type=GPU_TYPE,
            container_disk_in_gb=20,  # Set disk size appropriately for DB instance
            image=DOCKER_IMAGE,
            gpu_count=0,  # No GPU required for the DB instance
            ports=f'{DB_PORT}/tcp',  # Expose the DB port
            log_file=get_log_file("db"),
            template_id=BASE_TEMPLATE_ID,
            cmd=DOCKER_CMD,
            env=env
        )

        # Get public IP and port for DB
        db_ip, db_port = await get_public_ip_and_port(db_instance['id'], private_port=int(DB_PORT))
        db_url = f"http://{db_ip}:{db_port}"

        # Save DB info to state
        state['db'] = {
            'private_key': db_helpers['private_key'],
            'address': db_url
        }
        db_perm_id = await db_adapter.create_perm_description(PermType.ModifyDb.name)
        
        assert db_perm_id == GUESS_DB_PERM_ID, f"Expected db_perm_id {GUESS_DB_PERM_ID}, got {db_perm_id}"

        save_state(state)
        logging.info(f"DB service started on {db_url}")

        # Wait for DB to become healthy
        if not await wait_for_health(db_url):
            logging.error("Error: DB service did not become available within the timeout period.")
            exit(1)

        # Initialize DB schema
        await init_db()
    else:
        logging.info("DB instance already running. Reconnecting...")
        db_url = state['db']['address']
        db_adapter.base_url = db_url
        db_adapter.private_key = state['db']['private_key']
        logging.info(f"Reconnected to DB at {db_url}")

async def main():
    global sot_url, rpc_url
    task_counts = {}

    # Load or initialize state
    state = load_state()

    # Determine if we are reconnecting based on existing DB info in state
    is_reconnecting = 'db' in state and 'address' in state['db'] and 'private_key' in state['db']

    # Initialize DB adapter
    db_adapter = DBAdapterClient(base_url="", private_key=args.private_key)

    # Launch or reconnect DB instance
    await launch_db_instance(state, db_adapter, is_reconnecting)

    # Update db_adapter with actual DB URL
    db_url = state['db']['address']
    db_adapter.base_url = db_url
    db_adapter.private_key = state['db']['private_key']

    # Reset logs
    reset_logs(LOG_DIR)

    # If not reconnecting, perform initial setup
    if not is_reconnecting:
        # Remove existing DB file if exists
        if os.path.exists(db_path):
            os.remove(db_path)

        # Remove global plugin directory if exists
        global_plugin_dir = os.path.join(parent_dir, 'plugins', 'global_plugins')  # Adjust as per your directory structure
        if os.path.exists(global_plugin_dir):
            shutil.rmtree(global_plugin_dir, ignore_errors=True)

        # Start Flask server in a separate thread
        app = Flask(__name__)

        @app.route('/')
        def index():
            return jsonify({"status": "DB is running"})

        flask_thread = threading.Thread(target=lambda: app.run(port=5002))
        flask_thread.start()

        logging.info("Starting deployment...")

        # Set environment variables for deployment
        os.environ['SUBNET_ADDRESSES_JSON'] = args.subnet_addresses
        os.environ['PANTHALIA_DEPLOYMENT'] = args.deployment_config
        os.environ['SOT_URL'] = args.sot_url

        logging.info(f'Time in string: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')

        web3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
        if not await wait_for_rpc_available(web3):
            exit(1)

        await set_interval_mining(web3, 1)

        # Run the deployment command
        deploy_command = [
            'forge', 'script', os.path.basename(args.forge_script),
            '--broadcast', '--rpc-url', rpc_url,
            '--private-key', args.private_key, '-vv'
        ]
        subprocess.run(deploy_command, cwd=os.path.dirname(args.forge_script), check=True)

        logging.info("Deployment completed successfully, loading JSON files...")

        # Load subnet_addresses and deployment_config
        with open(args.subnet_addresses, 'r') as file:
            subnet_addresses = json.load(file)

        with open(args.deployment_config, 'r') as file:
            deployment_config = json.load(file)

        state['subnet_addresses'] = subnet_addresses
        state['deployment_config'] = deployment_config
        save_state(state)

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
            token_address,
            args.group
        )

        plugin = await get_plugin(plugin_id, db_adapter)

        check_guessed_ids(subnet_id, GUESSED_SUBNET_ID, 'Subnet')
        check_guessed_ids(plugin_id, GUESSED_PLUGIN_ID, 'Plugin')

        await db_adapter.create_perm(deployer_address, GUESS_DB_PERM_ID)

        logging.info('Generating wallets')


        # Start master process
        logging.info("Starting master process...")

        master_env = {
            'GITHUB_TOKEN': os.environ.get('GITHUB_TOKEN', ''),
            'SERVICE_TYPE': 'master',
            'RANK': '0',
            'WORLD_SIZE': '1',
            'PRIVATE_KEY': args.private_key,
            'MAX_CONCURRENT_ITERATIONS': str(await plugin.get('max_concurrent_iterations')),
            'DB_URL': db_url,
            'NUM_WORKERS': str(args.worker_count),
            'DEPLOY_TYPE': 'cloud',
            'CLOUD_KEY': os.environ.get('CLOUD_KEY', ''),
        }

        if args.detailed_logs:
            master_env['DETAILED_LOGS'] = 'true'
        if args.torch_compile:
            master_env['TORCH_COMPILE'] = 'true'

        master_instance, master_helpers = await launch_instance_and_record_logs(
            name="master",
            gpu_type=GPU_TYPE,
            container_disk_in_gb=50,
            image=DOCKER_IMAGE,
            gpu_count=1,
            ports='',
            log_file=get_log_file("master"),
            template_id=BASE_TEMPLATE_ID,
            cmd=DOCKER_CMD,
            env=master_env
        )

        # Create instance entry in the DB
        await db_adapter.create_instance(
            name="master",
            service_type=ServiceType.Master.name,
            job_id=None,
            private_key=master_helpers['private_key'],
            pod_id=master_instance['id'],
            process_id=str(master_instance['pid']) if 'pid' in master_instance else '0'
        )

        logging.info(f"Master process started on instance {master_instance['id']}")

    else:
        # If reconnecting, skip launching new instances
        logging.info("Skipping launching new instances as we are reconnecting to an existing DB.")
        # Here, you might want to fetch existing instances from the DB and ensure they are running
        # Depending on your DB and infrastructure setup, additional reconnection logic might be needed

    # Start curses interface in a separate thread
    curses_thread = threading.Thread(
        target=curses.wrapper,
        args=(monitor_processes, db_adapter, 1, task_counts)  # Assuming job_id is 1
    )
    curses_thread.start()

    # If not reconnecting, proceed with launching services
    if not is_reconnecting:
        # Run the task tracking in an asyncio loop
        pool_contract = web3.eth.contract(address=pool_address, abi=load_abi('Pool'))
        await track_tasks(web3, subnet_addresses, pool_contract, task_counts)

    # Keep the main thread alive to allow curses and tracking to run
    while True:
        await asyncio.sleep(1)

def signal_handler(signal_received, frame):
    logging.info("SIGINT received, shutting down...")
    terminate_processes(db_adapter=None, job_id=1)  # Adjust as needed
    os._exit(0)  # Force exit the program

# Register the signal handler for SIGINT
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main())
