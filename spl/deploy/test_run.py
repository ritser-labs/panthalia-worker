import subprocess
import json
import os
import time
import argparse
import requests
import threading
import curses
from ..common import load_abi, wait_for_rpc_available, wait_for_sot, SOT_PRIVATE_PORT, fund_wallets, MAX_CONCURRENT_ITERATIONS
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
from .util import is_port_open  # Importing the is_port_open function

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
ANVIL_LOG_FILE = os.path.join(LOG_DIR, 'anvil.log')
SOT_LOG_FILE = os.path.join(LOG_DIR, 'sot.log')
BLOCK_TIMESTAMPS_FILE = os.path.join(STATE_DIR, 'block_timestamps.json')

REMOTE_MODEL_FILE = '/app/spl/data/state/model.pt'
LOCAL_MODEL_FILE = os.path.join(parent_dir, 'data', 'state', 'model.pt')

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
    return parser.parse_args()

args = parse_args()

sync_status = {}
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
        response = requests.get(get_num_workers_url)
        synced_workers = response.json()
        logging.debug(f"Synced {synced_workers}/{worker_count} workers.")
        if synced_workers >= worker_count:
            logging.debug("All workers have synced.")
            return True
        time.sleep(2)
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
    terminate_all_pods()

def signal_handler(signal, frame):
    logging.info("SIGINT received, shutting down...")
    terminate_processes()
    os._exit(0)  # Force exit the program

# Register the signal handler for SIGINT
signal.signal(signal.SIGINT, signal_handler)

def reset_logs(log_dir):
    if os.path.exists(log_dir):
        for file_name in os.listdir(log_dir):
            file_path = os.path.join(log_dir, file_name)
            try:
                with open(file_path, 'w') as file:
                    pass  # This will truncate the file to zero length
                logging.debug(f"Erased log file: {file_path}")
            except Exception as e:
                logging.debug(f"Error erasing log file {file_path}: {e}")

def fetch_latest_loss():
    global latest_loss_cache, sot_url
    current_time = time.time()
    
    logging.debug(10)

    # Check if the cache is expired
    if current_time - latest_loss_cache['last_fetched'] > LOSS_REFRESH_INTERVAL:
        logging.debug(20)
        try:
            response = requests.get(f"{sot_url}/get_loss", timeout=1)
            logging.debug(30)
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
        logging.debug(1)
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
        logging.debug(2)

        # Display logs on the left side
        process_name = ordered_process_names[selected_process]
        log_file = os.path.join(LOG_DIR, f"{process_name}.log")
        log_lines = []

        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_lines.extend(f.readlines())

        for i, line in enumerate(log_lines[-(height - 2):]):
            stdscr.addstr(i, 0, line[:split_point - 2])

        # Draw the separator line
        for y in range(height):
            stdscr.addch(y, split_point - 2, curses.ACS_VLINE)
        
        logging.debug(3)

        for i, name in enumerate(ordered_process_names):
            is_selected = (i == selected_process)
            status = pod_helpers[name]['is_ssh_session_alive']()
            color = curses.color_pair(1) if status else curses.color_pair(2)
            indicator = '*' if is_selected else ' '

            stdscr.addstr(i, split_point, f"{indicator} {name}", color)
        logging.debug(4)

        # Fetch and display the latest loss
        latest_loss = fetch_latest_loss()
        loss_display = f"Latest Loss: {latest_loss:.3f}" if latest_loss is not None else "Latest Loss: N/A"
        loss_y = height - len(task_counts) - 5
        stdscr.addstr(loss_y, split_point, loss_display, curses.color_pair(4))
        
        logging.debug(5)

        # Draw task counts below the latest loss
        task_start = height - 3 - len(task_counts)
        for i, (task_type, (solver_selected, active)) in enumerate(task_counts.items()):
            stdscr.addstr(task_start + i, split_point, f"{task_type}: {solver_selected}/{active}", curses.color_pair(3))

        stdscr.addstr(height - 1, 0, "Use arrow keys to navigate. Press 'q' to quit, 's' to download model.", curses.A_BOLD)
        stdscr.addstr(height - 1, split_point, "PANTHALIA SIMULATOR V0", curses.color_pair(3))
        stdscr.refresh()
        logging.debug(6)

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
            break

        if last_resize and time.time() - last_resize > 0.1:
            draw_screen()
            last_resize = None

        logging.debug("Drawing screen...")
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

async def launch_worker(worker_idx, subnet_addresses, worker_wallets, token_contract, pool_address):
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
        'BACKEND': args.backend
    }

    worker_name = f'worker_{worker_idx}'
    worker_instance, worker_helpers = await launch_instance_and_record_logs(
        name=worker_name,
        gpu_type=GPU_TYPE,
        image=DOCKER_IMAGE,
        gpu_count=1,
        ports='',
        log_file=os.path.join(LOG_DIR, f"{worker_name}.log"),
        env=env,
        template_id=BASE_TEMPLATE_ID,
        cmd=DOCKER_CMD
    )
    logging.info(f"Started worker process {worker_idx} for tasks on instance {worker_instance['id']} with env {env}")

    return worker_name, worker_instance, worker_helpers

async def main():
    global sot_url, rpc_url
    processes = {}
    task_counts = {}  # Dictionary to store task counts
    pod_helpers = {}

    # Retrieve the public IP address of the machine
    public_ip = get_public_ip()
    if not public_ip:
        logging.error("Could not retrieve public IP address.")
        exit(1)

    # Reset logs
    reset_logs(LOG_DIR)
    master_wallets = generate_wallets(args.num_master_wallets)
    master_public_keys = [wallet['address'] for wallet in master_wallets]


    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    
    env = {
        'GITHUB_TOKEN': os.environ.get('GITHUB_TOKEN', ''),
        'SERVICE_TYPE': 'sot',
        'RANK': '0',
        'WORLD_SIZE': '1',
        'PUBLIC_KEYS': INPUT_JSON_PATH + '_0',
        'SOT_PRIVATE_PORT': str(SOT_PRIVATE_PORT),
    }

    logging.info(f'Environment variables: {env}')

    # Start the SOT service on a remote instance
    logging.info("Starting SOT instance...")
    
    sot_promise = launch_instance_and_record_logs(
        name="sot_instance",
        gpu_count=0,
        ports=f'{SOT_PRIVATE_PORT}/tcp',
        log_file=SOT_LOG_FILE,
        template_id=BASE_TEMPLATE_ID,
        cmd=DOCKER_CMD,
        env=env,
        input_jsons=[master_public_keys]
    )

    # Start Anvil on a remote instance
    logging.info("Starting Anvil instance...")
    anvil_instance, anvil_helpers = await launch_instance_and_record_logs(
        name="anvil_instance",
        gpu_count=0,
        ports='8545/tcp',
        log_file=ANVIL_LOG_FILE,
        cmd=ANVIL_CMD,
        template_id=BASE_TEMPLATE_ID
    )
    pod_helpers['anvil'] = anvil_helpers
    anvil_ip, anvil_port = await get_public_ip_and_port(anvil_instance['id'], private_port=8545)
    rpc_url = f"http://{anvil_ip}:{anvil_port}"
    processes['anvil'] = anvil_instance
    logging.info(f"Anvil started on {rpc_url}")

    # Wait until the Anvil port is open
    logging.info(f"Waiting for Anvil to open port {anvil_port}...")
    while not is_port_open(anvil_ip, anvil_port):
        logging.info("Anvil port not open yet. Retrying in 1 second...")
        time.sleep(1)
    logging.info("Anvil port is now open.")

    try:
        # Delete all .pt files in the state directory except for the latest version for each tensor
        delete_old_tensor_files(STATE_DIR, BLOCK_TIMESTAMPS_FILE)

        # Delete the temp directory
        delete_directory_contents(TEMP_DIR)

        # Print initial stage
        logging.info("Starting deployment...")
    
        with open(args.subnet_addresses, 'r') as file:
            subnet_addresses = json.load(file)


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


        with open(args.deployment_config, 'r') as file:
            deployment_config = json.load(file)
        
        logging.info('JSONs loaded, parsing deployment config...')

        distributor_contract_address = deployment_config['distributor']
        pool_address = deployment_config['pool']

        deployer_account = web3.eth.account.from_key(args.private_key)
        deployer_address = deployer_account.address
        pool_contract = web3.eth.contract(address=pool_address, abi=load_abi('Pool'))
        token_address = await pool_contract.functions.token().call()
        token_contract = web3.eth.contract(address=token_address, abi=load_abi('ERC20'))

        logging.info('Generating wallets')

        await fund_wallets(web3, args.private_key, master_wallets, deployer_address, token_contract, 1, 10000 * 10**18, distributor_contract_address)

        # Generate wallets for workers and fund them
        worker_wallets = generate_wallets(args.worker_count * len(subnet_addresses))
        await fund_wallets(web3, args.private_key, worker_wallets, deployer_address, token_contract, 1, 10000 * 10**18, distributor_contract_address)

        sot_instance, sot_helpers = await sot_promise
        pod_helpers['sot'] = sot_helpers
        sot_ip, sot_port = await get_public_ip_and_port(sot_instance['id'], private_port=SOT_PRIVATE_PORT)
        sot_url = f"http://{sot_ip}:{sot_port}"
        processes['sot'] = sot_instance
        logging.info(f"SOT service started on {sot_url}")

        # Wait for the SOT service to be available
        if not wait_for_sot(sot_url):
            logging.error("Error: SOT service did not become available within the timeout period.")
            sot_instance.terminate()
            exit(1)

        # Print worker initialization stage
        logging.info("Starting worker processes...")

        # Use asyncio.gather() to launch all workers concurrently
        worker_tasks = [
            launch_worker(
                worker_idx,
                subnet_addresses,
                worker_wallets,
                token_contract,
                pool_address
            )
            for worker_idx in range(args.worker_count)
        ]
        try:
            # Print master initialization stage
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
            }

            # Start master.py on a remote instance
            master_instance, master_helpers = await launch_instance_and_record_logs(
                name="master_instance",
                gpu_count=0,
                ports='',
                log_file=os.path.join(LOG_DIR, 'master.log'),
                env=env,
                template_id=BASE_TEMPLATE_ID,
                cmd=DOCKER_CMD,
                input_jsons=[master_wallets, subnet_addresses]
            )
            pod_helpers['master'] = master_helpers
            processes['master'] = master_instance
            logging.info(f"Master process started on instance {master_instance['id']}")
                
                
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
            terminate_processes()
            raise e

    except Exception as e:
        terminate_processes()
        raise e

if __name__ == "__main__":
    asyncio.run(main())
