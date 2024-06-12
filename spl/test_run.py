import subprocess
import json
import os
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Test run script for starting workers and master")
    parser.add_argument('--subnet_addresses', type=str, required=True, help="Path to the subnet addresses JSON file")
    parser.add_argument('--deployment_config', type=str, required=True, help="Path to the deployment configuration JSON file")
    parser.add_argument('--rpc_url', type=str, default='http://localhost:8545', help="URL of the Ethereum RPC node")
    parser.add_argument('--sot_url', type=str, required=True, help="Source of Truth URL for streaming gradient updates")
    parser.add_argument('--private_key', type=str, required=True, help="Private key of the worker's Ethereum account")
    parser.add_argument('--group', type=int, required=True, help="Group for depositing stake")
    parser.add_argument('--local_storage_dir', type=str, default='local_storage', help="Directory for local storage of files")
    return parser.parse_args()

args = parse_args()

with open(args.subnet_addresses, 'r') as file:
    subnet_addresses = json.load(file)

with open(args.deployment_config, 'r') as file:
    deployment_config = json.load(file)

pool_address = deployment_config['pool']

worker_processes = []

# Start worker.py for each subnet
for task_type, subnet_address in subnet_addresses.items():
    command = [
        'python', 'worker.py',
        '--task_type', task_type,
        '--subnet_address', subnet_address,
        '--private_key', args.private_key,
        '--rpc_url', args.rpc_url,
        '--sot_url', args.sot_url,
        '--pool_address', pool_address,
        '--group', str(args.group)
    ]
    worker_processes.append(subprocess.Popen(command))

# Give workers some time to initialize
time.sleep(10)

# Start master.py
master_command = [
    'python', 'master.py',
    '--rpc_url', args.rpc_url,
    '--private_key', args.private_key,
    '--sot_url', args.sot_url,
    '--subnet_addresses', args.subnet_addresses
]
master_process = subprocess.Popen(master_command)

# Wait for the master process to complete
master_process.wait()

# Terminate all worker processes
for p in worker_processes:
    p.terminate()
    p.wait()
