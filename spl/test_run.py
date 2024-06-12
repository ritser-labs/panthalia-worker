import subprocess
import json
import os
import time
import argparse
import requests
from common import model_args

def parse_args():
    parser = argparse.ArgumentParser(description="Test run script for starting workers and master")
    parser.add_argument('--subnet_addresses', type=str, required=True, help="Path to the subnet addresses JSON file")
    parser.add_argument('--deployment_config', type=str, required=True, help="Path to the deployment configuration JSON file")
    parser.add_argument('--rpc_url', type=str, default='http://localhost:8545', help="URL of the Ethereum RPC node")
    parser.add_argument('--sot_url', type=str, required=True, help="Source of Truth URL for streaming gradient updates")
    parser.add_argument('--private_key', type=str, required=True, help="Private key of the worker's Ethereum account")
    parser.add_argument('--group', type=int, required=True, help="Group for depositing stake")
    parser.add_argument('--local_storage_dir', type=str, default='local_storage', help="Directory for local storage of files")
    parser.add_argument('--forge_script', type=str, default='script/Deploy.s.sol', help="Path to the Forge deploy script")
    parser.add_argument('--backend', type=str, default='nccl', help="Distributed backend to use (default: nccl, use 'gloo' for macOS)")
    return parser.parse_args()

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

args = parse_args()

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

worker_processes = []

# Print SOT service initialization stage
print("Starting SOT service...")

# Start the SOT service
sot_process = subprocess.Popen(['python', 'sot.py'])
print(f"SOT service started with PID {sot_process.pid}")

# Wait for the SOT service to be available
if not wait_for_sot(args.sot_url):
    print("Error: SOT service did not become available within the timeout period.")
    sot_process.terminate()
    exit(1)

# Print worker initialization stage
print("Starting worker processes...")

# Start worker.py for each subnet
for task_type, subnet_address in subnet_addresses.items():
    # Determine base_task_type correctly
    if 'forward_layer' in task_type:
        base_task_type = 'forward'
    elif 'backward_layer' in task_type:
        base_task_type = 'backward'
    else:
        base_task_type = task_type  # Use the full task type as is

    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    command = [
        'python', 'worker.py',
        '--task_type', base_task_type,
        '--subnet_address', subnet_address,
        '--private_key', args.private_key,
        '--rpc_url', args.rpc_url,
        '--sot_url', args.sot_url,
        '--pool_address', pool_address,
        '--group', str(args.group),
        '--local_storage_dir', args.local_storage_dir,
        '--backend', args.backend
    ]
    worker_processes.append(subprocess.Popen(command))
    print(f"Started worker process for task {task_type} with command: {' '.join(command)}")

# Print workers started stage
print("Worker processes started.")

# Give workers some time to initialize
time.sleep(10)

# Print master initialization stage
print("Starting master process...")

# Start master.py
master_command = [
    'python', 'master.py',
    '--rpc_url', args.rpc_url,
    '--private_key', args.private_key,
    '--sot_url', args.sot_url,
    '--subnet_addresses', args.subnet_addresses
]
master_process = subprocess.Popen(master_command)
print(f"Started master process with command: {' '.join(master_command)}")

# Print master started stage
print("Master process started.")

# Wait for the master process to complete
master_process.wait()

# Terminate all worker processes
for p in worker_processes:
    p.terminate()
    p.wait()

# Terminate the SOT process
sot_process.terminate()

# Print final stage
print("Test run completed.")
