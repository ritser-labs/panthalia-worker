import subprocess
import json
import os
import time
import argparse
import requests
import threading
import curses
from flask import Flask, request, jsonify
from common import model_args, load_abi, async_transact_with_contract_function
from web3 import AsyncWeb3, Web3
from eth_account import Account
import glob
import shutil
import asyncio
import logging
import traceback
import psutil  # For memory and VRAM usage profiling

# Define file paths and other configurations
LOG_DIR = 'logs'
STATE_DIR = os.path.join('data', 'state')
TEMP_DIR = os.path.join(STATE_DIR, 'temp')
MASTER_WALLETS_FILE = 'master_wallets.json'
MASTER_PUBLIC_KEYS_FILE = 'master_public_keys.json'
DEPLOY_SCRIPT = 'script/Deploy.s.sol'
LOG_FILE = os.path.join(LOG_DIR, 'test_run.log')
ANVIL_LOG_FILE = os.path.join(LOG_DIR, 'anvil.log')
SOT_LOG_FILE = os.path.join(LOG_DIR, 'sot.log')
BLOCK_TIMESTAMPS_FILE = os.path.join(STATE_DIR, 'block_timestamps.json')

# Configure logging to file only, initially including stdout
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
for handler in logging.getLogger().handlers:
    handler.setFormatter(formatter)

def parse_args():
    parser = argparse.ArgumentParser(description="Test run script for starting workers and master")
    parser.add_argument('--subnet_addresses', type=str, required=True, help="Path to the subnet addresses JSON file")
    parser.add_argument('--deployment_config', type=str, required=True, help="Path to the deployment configuration JSON file")
    parser.add_argument('--rpc_url', type=str, default='http://localhost:8545', help="URL of the Ethereum RPC node")
    parser.add_argument('--sot_url', type=str, required=True, help="Source of Truth URL for streaming gradient updates")
    parser.add_argument('--private_key', type=str, required=True, help="Private key of the deployer's Ethereum account")
    parser.add_argument('--group', type=int, required=True, help="Group for depositing stake")
    parser.add_argument('--local_storage_dir', type=str, default='data', help="Directory for local storage of files")
    parser.add_argument('--forge_script', type=str, default=DEPLOY_SCRIPT, help="Path to the Forge deploy script")
    parser.add_argument('--backend', type=str, default='nccl', help="Distributed backend to use (default: nccl, use 'gloo' for macOS)")
    parser.add_argument('--detailed_logs', action='store_true', help="Enable detailed logs for all processes")
    parser.add_argument('--num_master_wallets', type=int, default=70, help="Number of wallets to generate for the master process")
    return parser.parse_args()

args = parse_args()

sync_status = {}
app = Flask(__name__)

@app.route('/report_sync', methods=['GET'])
def report_sync():
    task_type = request.args.get('task_type')
    status = request.args.get('status')
    layer_idx = request.args.get('layer_idx')
    key = f"{task_type}_{layer_idx}" if layer_idx else task_type
    logging.debug(f"Received sync report for task_type={task_type}, layer_idx={layer_idx}, status={status}")
    if task_type and status:
        sync_status[key] = status
        synced_workers = sum(1 for status in sync_status.values() if status == 'synced')
        total_workers = len(sync_status)
        logging.debug(f"Synced {synced_workers}/{total_workers} workers.")
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'error', 'message': 'Missing argument'}), 400

def wait_for_sot(sot_url, timeout=1200):  # Increased timeout to 20 minutes
    """Wait for the SOT service to be available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{sot_url}/health")
            if response.status_code == 200:
                logging.debug("SOT service is available.")
                return True
        except requests.ConnectionError as e:
            logging.debug(f"Waiting for SOT service to be available... {e}")
        time.sleep(2)
    return False

def wait_for_workers_to_sync(worker_count, timeout=600):
    """Wait for all workers to sync their deposit stake."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        synced_workers = sum(1 for status in sync_status.values() if status == 'synced')
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

async def fund_wallets(web3, wallets, deployer_address, token_contract, amount_eth, amount_token):
    for wallet in wallets:
        tx = {
            'to': wallet['address'],
            'value': web3.to_wei(amount_eth, 'ether'),
            'gas': 21000,
            'gasPrice': await web3.eth.gas_price,
            'nonce': await web3.eth.get_transaction_count(deployer_address)
        }
        signed_tx = web3.eth.account.sign_transaction(tx, args.private_key)
        await web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        await web3.eth.wait_for_transaction_receipt(signed_tx.hash)

        await async_transact_with_contract_function(
            web3, token_contract, 'transfer', args.private_key, wallet['address'], amount_token
        )

def terminate_processes(processes):
    for process in processes:
        process.terminate()
    for process in processes:
        try:
            process.wait(timeout=5)  # Wait up to 5 seconds for each process to terminate
        except subprocess.TimeoutExpired:
            process.kill()  # Forcefully kill the process if it doesn't terminate in time

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

def monitor_processes(stdscr, processes, task_counts):
    logger = logging.getLogger()
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)

    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)
    selected_process = 0
    last_resize = None

    # Initialize colors
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Green for active processes
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)    # Red for inactive processes
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK) # Yellow for task counts (active)
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)   # Cyan for task counts (total)

    max_name_length = max(len(name) for name in processes.keys()) + 14  # Increased padding by 3 more characters
    right_col_width = max_name_length + 2  # Additional padding

    def draw_screen():
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        split_point = width - right_col_width  # Right column width based on max name length

        # Display logs on the left side
        process_name = list(processes.keys())[selected_process]
        log_file = os.path.join('logs', f"{process_name}.log")
        log_lines = []

        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_lines.extend(f.readlines())

        for i, line in enumerate(log_lines[-(height - 2):]):  # Leave space for instructions
            stdscr.addstr(i, 0, line[:split_point - 2])

        # Draw the separator line after all left side content is drawn
        for y in range(height):
            stdscr.addch(y, split_point - 2, curses.ACS_VLINE)

        # Order processes by name
        ordered_process_names = sorted(processes.keys(), key=lambda name: (
            name.startswith('worker_final_logits'),
            name.startswith('worker_forward'),
            name.startswith('worker_embed'),
            not name.startswith('worker'),
            name
        ))

        for i, name in enumerate(ordered_process_names):
            process = processes[name]
            is_selected = (i == selected_process)
            status = process.poll() is None
            color = curses.color_pair(1) if status else curses.color_pair(2)
            indicator = '*' if is_selected else ' '

            # Determine the task type and display accordingly
            if name == 'worker_final_logits':
                task_count = task_counts.get('final_logits', (0, 0))
                stdscr.addstr(i, split_point, f"{indicator} {name} ({task_count[0]}/{task_count[1]})", color)
            elif 'worker_forward+backward' in name:
                layer_idx = name.split('_')[-1]
                forward_task = f"forward_layer_{layer_idx}"
                backward_task = f"backward_layer_{layer_idx}"

                forward_count = task_counts.get(forward_task, (0, 0))
                backward_count = task_counts.get(backward_task, (0, 0))

                stdscr.addstr(i, split_point, f"{indicator} {name} ", color)
                stdscr.addstr(f"({forward_count[0]}/{forward_count[1]}) ", curses.color_pair(3))
                stdscr.addstr(f"({backward_count[0]}/{backward_count[1]})", curses.color_pair(4))
            elif 'worker_embed+embed_backward' in name:
                embed_task = "embed"
                embed_backward_task = "embed_backward"

                embed_count = task_counts.get(embed_task, (0, 0))
                embed_backward_count = task_counts.get(embed_backward_task, (0, 0))

                stdscr.addstr(i, split_point, f"{indicator} {name} ", color)
                stdscr.addstr(f"({embed_count[0]}/{embed_count[1]}) ", curses.color_pair(3))
                stdscr.addstr(f"({embed_backward_count[0]}/{embed_backward_count[1]})", curses.color_pair(4))
            else:
                stdscr.addstr(i, split_point, f"{indicator} {name}", color)

        stdscr.addstr(height - 1, 0, "Use arrow keys to navigate. Press 'q' to quit.", curses.A_BOLD)
        # Add the "PANTHALIA SIMULATOR V0" text at the bottom of the right column
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
        elif key == ord('q'):
            terminate_processes(list(processes.values()))
            break

        # Handle screen resize with a slight delay to prevent flashing
        if last_resize and time.time() - last_resize > 0.1:
            draw_screen()
            last_resize = None

        draw_screen()
        time.sleep(0.05)  # Frequent updates

    stdscr.keypad(False)  # Reset keypad mode before exiting
    curses.endwin()
    os._exit(0)  # Force exit the program

# Function to log memory and VRAM usage
def log_memory_vram_usage(processes, interval=5):
    logger = logging.getLogger('memory_logger')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicate logs
    logger.handlers = []

    # Add file handler if not present
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    while True:
        for name, process in processes.items():
            try:
                if process.poll() is None:  # Process is running
                    p = psutil.Process(process.pid)
                    memory_info = p.memory_info()
                    mem_usage = memory_info.rss / (1024 ** 2)  # Convert to MB
                    # VRAM usage can be logged using appropriate libraries if needed.
                    # This example logs only the main memory usage.
                    logger.info(f"{name}: Memory Usage: {mem_usage:.2f} MB")
            except psutil.NoSuchProcess:
                logger.warning(f"Process {name} with PID {process.pid} no longer exists.")
            except Exception as e:
                logger.error(f"Error logging memory for {name}: {e}")
        time.sleep(interval)


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

async def main():
    processes = {}
    task_counts = {}  # Dictionary to store task counts

    # Reset logs
    reset_logs(LOG_DIR)

    # Start anvil process
    logging.info("Starting anvil...")
    anvil_log = open(ANVIL_LOG_FILE, 'w')
    anvil_process = subprocess.Popen(['anvil'], stdout=anvil_log, stderr=anvil_log)
    processes['anvil'] = anvil_process
    logging.info(f"Anvil started with PID {anvil_process.pid}")

    try:
        # Delete all .pt files in the state directory except for the latest version for each tensor
        delete_old_tensor_files(STATE_DIR, BLOCK_TIMESTAMPS_FILE)

        # Delete the temp directory
        delete_directory_contents(TEMP_DIR)

        # Start Flask server in a separate thread
        flask_thread = threading.Thread(target=lambda: app.run(port=5002))
        flask_thread.start()

        # Print initial stage
        logging.info("Starting deployment...")

        # Set environment variables for deployment
        os.environ['SUBNET_ADDRESSES_JSON'] = args.subnet_addresses
        os.environ['PANTHALIA_DEPLOYMENT'] = args.deployment_config
        os.environ['LAYERS'] = str(model_args.n_layers)
        os.environ['SOT_URL'] = args.sot_url

        # Run Deploy.s.sol script from the correct path
        deploy_command = [
            'forge', 'script', os.path.basename(args.forge_script),
            '--broadcast', '--rpc-url', args.rpc_url,
            '--private-key', args.private_key, '-vv'
        ]
        subprocess.run(deploy_command, cwd=os.path.dirname(args.forge_script), check=True)

        web3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(args.rpc_url))

        # Print deployment stage completion
        logging.info("Deployment completed successfully.")

        # Load subnet addresses and deployment config
        with open(args.subnet_addresses, 'r') as file:
            subnet_addresses = json.load(file)

        with open(args.deployment_config, 'r') as file:
            deployment_config = json.load(file)

        pool_address = deployment_config['pool']

        web3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(args.rpc_url))
        deployer_account = web3.eth.account.from_key(args.private_key)
        deployer_address = deployer_account.address
        pool_contract = web3.eth.contract(address=pool_address, abi=load_abi('Pool'))
        token_address = await pool_contract.functions.token().call()
        token_contract = web3.eth.contract(address=token_address, abi=load_abi('ERC20'))

        # Initialize sync_status with all subnet addresses
        sync_status = {f"{task_type}_{subnet_address}" if 'layer' in task_type else task_type: 'unsynced' for task_type, subnet_address in subnet_addresses.items()}

        # Generate wallets for the master and fund them
        master_wallets = generate_wallets(args.num_master_wallets)
        await fund_wallets(web3, master_wallets, deployer_address, token_contract, 1, 10000 * 10**18)

        with open(MASTER_WALLETS_FILE, 'w') as f:
            json.dump(master_wallets, f)

        # Save the public keys of the master wallets
        master_public_keys = [wallet['address'] for wallet in master_wallets]
        with open(MASTER_PUBLIC_KEYS_FILE, 'w') as f:
            json.dump(master_public_keys, f)

        # Generate wallets for workers and fund them
        worker_wallets = generate_wallets(len(subnet_addresses))
        await fund_wallets(web3, worker_wallets, deployer_address, token_contract, 1, 10000 * 10**18)
        await set_interval_mining(web3, 1)

        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'

        # Print SOT service initialization stage
        logging.info("Starting SOT service...")

        # Start the SOT service
        sot_log = open(SOT_LOG_FILE, 'w')
        sot_process = subprocess.Popen(['python', 'sot.py', '--public_keys_file', MASTER_PUBLIC_KEYS_FILE], stdout=sot_log, stderr=sot_log)
        processes['sot'] = sot_process
        logging.info(f"SOT service started with PID {sot_process.pid}")

        # Wait for the SOT service to be available
        if not wait_for_sot(args.sot_url):
            logging.error("Error: SOT service did not become available within the timeout period.")
            sot_process.terminate()
            exit(1)

        # Print worker initialization stage
        logging.info("Starting worker processes...")

        # Start worker.py for each subnet
        task_combinations = []

        # Collect task combinations
        embed_task = None
        embed_backward_task = None
        final_logits_task = None
        forward_backward_tasks = {}

        # Iterate only over forward_layer and embed tasks, skip backward tasks
        for task_type, subnet_address in subnet_addresses.items():
            if task_type.startswith("forward_layer"):
                layer_idx = int(task_type.split('_')[-1])
                forward_backward_tasks[layer_idx] = (subnet_address, subnet_addresses.get(f"backward_layer_{layer_idx}", None))
            elif task_type == "embed":
                embed_task = subnet_address
                embed_backward_task = subnet_addresses.get("embed_backward", None)
            elif task_type == "final_logits":
                final_logits_task = subnet_address

        # Combine tasks and avoid double counting
        if embed_task:
            task_type = "embed+embed_backward" if embed_backward_task else "embed"
            task_combinations.append((task_type, None, embed_task, embed_backward_task))

        if final_logits_task:
            task_combinations.append(("final_logits", None, final_logits_task, None))

        for layer_idx, (forward_address, backward_address) in forward_backward_tasks.items():
            if forward_address:
                task_type = "forward" + ("+backward" if backward_address else "")
                task_combinations.append((task_type, layer_idx, forward_address, backward_address))

        worker_count = len(task_combinations)

        for task_type, layer_idx, address_1, address_2 in task_combinations:
            if address_2 is None:
                worker_wallet = worker_wallets.pop(0)['private_key']
            else:
                worker_wallet = worker_wallets.pop(0)['private_key'] + '+' + worker_wallets.pop(0)['private_key']
            command = [
                'python', 'worker.py',
                '--task_types', task_type,
                '--subnet_addresses', address_1 if address_2 is None else f"{address_1}+{address_2}",
                '--private_keys', worker_wallet,
                '--rpc_url', args.rpc_url,
                '--sot_url', args.sot_url,
                '--pool_address', pool_address,
                '--group', str(args.group),
                '--local_storage_dir', args.local_storage_dir,
                '--backend', args.backend,
            ]

            if layer_idx is not None:
                command.extend(['--layer_idx', str(layer_idx)])
            worker_name = f'worker_{task_type + "_" + str(layer_idx) if layer_idx is not None else task_type}'
            log_file_path = os.path.join(LOG_DIR, f"{worker_name}.log")
            log_file = open(log_file_path, 'w')
            worker_process = subprocess.Popen(command, stdout=log_file, stderr=log_file)
            processes[worker_name] = worker_process
            logging.info(f"Started worker process for tasks {task_type} with command: {' '.join(command)}")

        try:
            # Wait for all workers to sync
            if not wait_for_workers_to_sync(worker_count):
                logging.error("Error: Not all workers synced within the timeout period.")
                terminate_processes(processes.values())
                exit(1)

            # Print master initialization stage
            logging.info("Starting master process...")

            # Start master.py
            master_log = open(os.path.join(LOG_DIR, 'master.log'), 'w')
            master_command = [
                'python', 'master.py',
                '--rpc_url', args.rpc_url,
                '--wallets_file', MASTER_WALLETS_FILE,
                '--sot_url', args.sot_url,
                '--subnet_addresses', args.subnet_addresses,
            ]
            if args.detailed_logs:
                master_command.append('--detailed_logs')
            master_process = subprocess.Popen(master_command, stdout=master_log, stderr=master_log)
            processes['master'] = master_process
            logging.info(f"Started master process with command: {' '.join(master_command)}")

            # Print master started stage
            logging.info("Master process started.")

            # Start the curses interface in a new thread
            curses_thread = threading.Thread(target=curses.wrapper, args=(monitor_processes, processes, task_counts))
            curses_thread.start()

            # Start the memory and VRAM usage logger in a separate thread
            mem_logger_thread = threading.Thread(target=log_memory_vram_usage, args=(processes,))
            mem_logger_thread.daemon = True
            mem_logger_thread.start()

            # Run the task tracking in an asyncio loop
            await track_tasks(web3, subnet_addresses, pool_contract, task_counts)

        except Exception as e:
            logging.error(f"Error: {e}")
            terminate_processes(list(processes.values()))
            exit(1)

    except Exception as e:
        logging.error(f"Error: {e}")
        terminate_processes(list(processes.values()))
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())
