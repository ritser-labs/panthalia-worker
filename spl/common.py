import torch
import os
from tokenizer import Tokenizer
from model import ModelArgs
import json
import logging
import torch.distributed as dist

# Define the new tokenizer and model arguments
tokenizer = Tokenizer('cl100k_base')

model_args = ModelArgs(
    vocab_size=tokenizer.get_vocab_size(),
    dim=512,
    n_layers=4,
    n_heads=8,
    multiple_of=256,
    norm_eps=1e-5,
    rope_theta=500000,
    max_batch_size=32,
    max_seq_len=2048
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_to_disk(data, filename):
    torch.save(data, filename)
    print(f"Saved to {filename}")

def load_from_disk(filename):
    if os.path.exists(filename):
        data = torch.load(filename, map_location=device)
        print(f"Loaded from {filename}")
        return data
    else:
        print(f"File {filename} does not exist")
        return None

def save_layer_state_dict(state_dict, filename):
    torch.save(state_dict, filename)
    print(f"Layer state dict saved to {filename}")

def load_layer_state_dict(filename):
    if os.path.exists(filename):
        state_dict = torch.load(filename)
        print(f"Layer state dict loaded from {filename}")
        return state_dict
    else:
        print(f"File {filename} does not exist")
        return None

def extract_error_selectors(abi, web3, error_selectors):
    for item in abi:
        if item.get('type') == 'error':
            name = item['name']
            inputs = item['inputs']
            selector = web3.keccak(text=f"{name}({','.join([input['type'] for input in inputs])})")[:4].hex()
            selector = selector.lower()
            error_selectors[selector] = item
            logging.info(f"Extracted error selector {selector} for {name}")

def load_contracts(web3, subnet_addresses):
    abi_dir = 'abis'
    abis = {}
    contracts = {}
    error_selectors = {}

    for root, _, files in os.walk(abi_dir):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r') as abi_file:
                    abi = json.load(abi_file).get('abi', [])
                    contract_name = os.path.splitext(file)[0]
                    abis[contract_name] = abi
                    logging.info(f"Loaded ABI for {contract_name}")
                    extract_error_selectors(abi, web3, error_selectors)

    for task, address in subnet_addresses.items():
        if 'SubnetManager' in abis:
            contracts[task] = web3.eth.contract(address=address, abi=abis['SubnetManager'])
            logging.info(f"Loaded contract for {task} with address {address}")
        else:
            logging.error(f"ABI for SubnetManager not found")

    return abis, contracts, error_selectors

def initialize_distributed_environment(backend, master_addr='localhost', master_port=None):
    if master_port is None:
        master_port = str(12356 + os.getpid() % 10000)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
