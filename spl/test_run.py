import subprocess
import json
import os
import time
import argparse
import requests
import threading
from flask import Flask, request, jsonify
from common import model_args, load_abi
from web3 import Web3
from eth_account import Account
import glob
import shutil
import signal
import curses

def parse_args():
    parser = argparse.ArgumentParser(description="Test run script for starting workers and master")
    parser.add_argument('--subnet_addresses', type=str, required=True, help="Path to the subnet addresses JSON file")
    parser.add_argument('--deployment_config', type=str, required=True, help="Path to the deployment configuration JSON file")
    parser.add_argument('--rpc_url', type=str, default='http://localhost:8545', help="URL of the Ethereum RPC node")
    parser.add_argument('--sot_url', type=str, required=True, help="Source of Truth URL for streaming gradient updates")
    parser.add_argument('--private_key', type=str, required=True, help="Private key of the deployer's Ethereum account")
    parser.add_argument('--group', type=int, required=True, help="Group for depositing stake")
    parser.add_argument('--local_storage_dir', type=str, default='data', help="Directory for local storage of files")
    parser.add_argument('--forge_script', type=str, default='script/Deploy.s.sol', help="Path to the Forge deploy script")
    parser.add_argument('--backend', type=str, default='nccl', help="Distributed backend to use (default: nccl, use 'gloo' for macOS)")
    parser.add_argument('--detailed_logs', action='store_true', help="Enable detailed logs for all processes")
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
    print(f"Received sync report for task_type={task_type}, layer_idx={layer_idx}, status={status}")
    if task_type and status:
        sync_status[key] = status
        synced_workers = sum(1 for status in sync_status.values() if status == 'synced')
        total_workers = len(sync_status)
        print(f"Synced {synced_workers}/{total_workers} workers.")
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
                print("SOT service is available.")
                return True
        except requests.ConnectionError as e:
            print(f"Waiting for SOT service to be available... {e}")
        time.sleep(2)
    return False

def wait_for_workers_to_sync(worker_count, timeout=600):
    """Wait for all workers to sync their deposit stake."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        synced_workers = sum(1 for status in sync_status.values() if status == 'synced')
        print(f"Synced {synced_workers}/{worker_count} workers.")
        if synced_workers >= worker_count:
            print("All workers have synced.")
            return True
        time.sleep(2)
    print("Timeout waiting for workers to sync.")
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
                print(f"Deleted old tensor file: {tensor_file}")
            except Exception as e:
                print(f"Error deleting file {tensor_file}: {e}")

def delete_directory_contents(directory):
    if os.path.exists(directory):
        try:
            shutil.rmtree(directory)
            print(f"Deleted directory: {directory}")
        except Exception as e:
            print(f"Error deleting directory {directory}: {e}")

def fund_wallets(web3, wallets, deployer_address, token_contract, amount_eth, amount_token):
    for wallet in wallets:
        tx = {
            'to': wallet['address'],
            'value': web3.to_wei(amount_eth, 'ether'),
            'gas': 21000,
            'gasPrice': web3.eth.gas_price,
            'nonce': web3.eth.get_transaction_count(deployer_address)
        }
        signed_tx = web3.eth.account.sign_transaction(tx, args.private_key)
        web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        web3.eth.wait_for_transaction_receipt(signed_tx.hash)
        
        tx = token_contract.functions.transfer(wallet['address'], amount_token).build_transaction({
            'chainId': web3.eth.chain_id,
            'gas': 100000,
            'gasPrice': web3.eth.gas_price,
            'nonce': web3.eth.get_transaction_count(deployer_address)
        })
        signed_tx = web3.eth.account.sign_transaction(tx, args.private_key)
        web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        web3.eth.wait_for_transaction_receipt(signed_tx.hash)

def terminate_processes(processes):
    for process in processes:
        process.terminate()
        process.wait()

def reset_logs(log_dir):
    if os.path.exists(log_dir):
        for file_name in os.listdir(log_dir):
            file_path = os.path.join(log_dir, file_name)
            try:
                os.remove(file_path)
                print(f"Deleted log file: {file_path}")
            except Exception as e:
                print(f"Error deleting log file {file_path}: {e}")

def monitor_processes(stdscr, processes):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)
    selected_process = 0

    # Initialize colors
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Cool color for the simulator text

    max_name_length = max(len(name) for name in processes.keys()) + 3  # Adding padding for indicator and spaces
    right_col_width = max_name_length + 2  # Additional padding

    def draw_screen():
        height, width = stdscr.getmaxyx()
        split_point = width - right_col_width  # Right column width based on max name length

        # Display logs on the left side
        process_name = list(processes.keys())[selected_process]
        log_file = os.path.join('logs', f"{process_name}.log")
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_lines = f.readlines()
        else:
            log_lines = []

        for i, line in enumerate(log_lines[-(height - 2):]):  # Leave space for instructions
            stdscr.addstr(i, 0, line[:split_point - 2])

        # Draw the separator line after all left side content is drawn
        for y in range(height):
            stdscr.addch(y, split_point - 2, curses.ACS_VLINE)

        # Display processes on the right side
        for i, (name, process) in enumerate(processes.items()):
            is_selected = (i == selected_process)
            status = process.poll() is None
            color = curses.color_pair(1) if status else curses.color_pair(2)
            indicator = '*' if is_selected else ' '
            stdscr.addstr(i, split_point, f"{indicator} {name}", color)
        
        stdscr.addstr(height - 1, 0, "Use arrow keys to navigate. Press 'q' to quit.", curses.A_BOLD)
        # Add the "PANTHALIA SIMULATOR V0" text at the bottom of the right column
        stdscr.addstr(height - 1, split_point, "PANTHALIA SIMULATOR V0", curses.color_pair(3))
        stdscr.refresh()

    while True:
        draw_screen()
        key = stdscr.getch()
        if key == curses.KEY_UP:
            selected_process = (selected_process - 1) % len(processes)
        elif key == curses.KEY_DOWN:
            selected_process = (selected_process + 1) % len(processes)
        elif key == ord('q'):
            terminate_processes(list(processes.values()))
            break
        time.sleep(0.1)

    stdscr.keypad(False)  # Reset keypad mode before exiting
    curses.endwin()
    os._exit(0)  # Force exit the program

if __name__ == "__main__":
    processes = {}
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    # Reset logs
    reset_logs(log_dir)

    # Start anvil process
    print("Starting anvil...")
    anvil_log = open(os.path.join(log_dir, 'anvil.log'), 'w')
    anvil_process = subprocess.Popen(['anvil'], stdout=anvil_log, stderr=anvil_log)
    processes['anvil'] = anvil_process
    print(f"Anvil started with PID {anvil_process.pid}")

    try:
        # Delete all .pt files in the state directory except for the latest version for each tensor
        state_dir = os.path.join(args.local_storage_dir, 'state')
        block_timestamps_file = os.path.join(state_dir, 'block_timestamps.json')
        delete_old_tensor_files(state_dir, block_timestamps_file)

        # Delete the temp directory
        temp_dir = os.path.join(state_dir, 'temp')
        delete_directory_contents(temp_dir)

        # Start Flask server in a separate thread
        flask_thread = threading.Thread(target=lambda: app.run(port=5002))
        flask_thread.start()

        # Print initial stage
        print("Starting deployment...")

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

        # Print deployment stage completion
        print("Deployment completed successfully.")

        # Load subnet addresses and deployment config
        with open(args.subnet_addresses, 'r') as file:
            subnet_addresses = json.load(file)

        with open(args.deployment_config, 'r') as file:
            deployment_config = json.load(file)

        pool_address = deployment_config['pool']

        web3 = Web3(Web3.HTTPProvider(args.rpc_url))
        deployer_account = web3.eth.account.from_key(args.private_key)
        deployer_address = deployer_account.address
        pool_contract = web3.eth.contract(address=pool_address, abi=load_abi('Pool'))
        token_address = pool_contract.functions.token().call()
        token_contract = web3.eth.contract(address=token_address, abi=load_abi('ERC20'))

        # Initialize sync_status with all subnet addresses
        sync_status = {f"{task_type}_{subnet_address}" if 'layer' in task_type else task_type: 'unsynced' for task_type, subnet_address in subnet_addresses.items()}

        # Generate wallets and fund them
        num_wallets = len(subnet_addresses)
        wallets = generate_wallets(num_wallets)
        fund_wallets(web3, wallets, deployer_address, token_contract, 1, 10000 * 10**18)

        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'

        # Print SOT service initialization stage
        print("Starting SOT service...")

        # Start the SOT service
        sot_log = open(os.path.join(log_dir, 'sot.log'), 'w')
        sot_process = subprocess.Popen(['python', 'sot.py', '--public_key', deployer_account.address], stdout=sot_log, stderr=sot_log)
        processes['sot'] = sot_process
        print(f"SOT service started with PID {sot_process.pid}")

        # Wait for the SOT service to be available
        if not wait_for_sot(args.sot_url):
            print("Error: SOT service did not become available within the timeout period.")
            sot_process.terminate()
            exit(1)

        # Print worker initialization stage
        print("Starting worker processes...")

        # Start worker.py for each subnet
        task_order = [
            'embed',
            *['forward_layer_{}'.format(i) for i in range(model_args.n_layers)],
            'final_logits',
            *['backward_layer_{}'.format(i) for i in reversed(range(model_args.n_layers))],
            'embed_backward'
        ]
        for task_type in task_order:
            if task_type in subnet_addresses:
                subnet_address = subnet_addresses[task_type]
                # Determine base_task_type and layer_idx
                if 'forward_layer' in task_type:
                    base_task_type = 'forward'
                    layer_idx = int(task_type.split('_')[-1])
                elif 'backward_layer' in task_type:
                    base_task_type = 'backward'
                    layer_idx = int(task_type.split('_')[-1])
                else:
                    base_task_type = task_type  # Use the full task type as is
                    layer_idx = None

                # Select the corresponding wallet for each worker
                wallet = wallets.pop(0)

                command = [
                    'python', 'worker.py',
                    '--task_type', base_task_type,
                    '--subnet_address', subnet_address,
                    '--private_key', wallet['private_key'],
                    '--rpc_url', args.rpc_url,
                    '--sot_url', args.sot_url,
                    '--pool_address', pool_address,
                    '--group', str(args.group),
                    '--local_storage_dir', args.local_storage_dir,
                    '--backend', args.backend,
                ]
                if layer_idx is not None:
                    command.extend(['--layer_idx', str(layer_idx)])
                log_file_path = os.path.join(log_dir, f'worker_{task_type}.log')
                log_file = open(log_file_path, 'w')
                worker_process = subprocess.Popen(command, stdout=log_file, stderr=log_file)
                processes[f'worker_{task_type}'] = worker_process
                print(f"Started worker process for task {task_type} with command: {' '.join(command)}")

        try:
            # Wait for all workers to sync
            if not wait_for_workers_to_sync(num_wallets):
                print("Error: Not all workers synced within the timeout period.")
                terminate_processes(processes.values())
                exit(1)

            # Print master initialization stage
            print("Starting master process...")

            # Start master.py
            master_log = open(os.path.join(log_dir, 'master.log'), 'w')
            master_command = [
                'python', 'master.py',
                '--rpc_url', args.rpc_url,
                '--private_key', args.private_key,
                '--sot_url', args.sot_url,
                '--subnet_addresses', args.subnet_addresses,
            ]
            if args.detailed_logs:
                master_command.append('--detailed_logs')
            master_process = subprocess.Popen(master_command, stdout=master_log, stderr=master_log)
            processes['master'] = master_process
            print(f"Started master process with command: {' '.join(master_command)}")

            # Print master started stage
            print("Master process started.")

            # Start the curses interface in a new thread
            curses.wrapper(monitor_processes, processes)

            # Wait for the master process to complete
            master_process.wait()

        finally:
            # Terminate all worker processes
            terminate_processes(processes.values())

            # Print final stage
            print("Test run completed.")

    finally:
        print("Terminating anvil process...")
        anvil_process.terminate()
        anvil_process.wait()
        print("Anvil process terminated.")
