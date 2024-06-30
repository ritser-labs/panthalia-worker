import torch
import os
from tokenizer import Tokenizer
from model import ModelArgs
import json
import logging
import torch.distributed as dist
import time
from io import BytesIO
import requests
from web3 import Web3
from enum import Enum
import web3
from collections import namedtuple

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


# Define Enums
class TaskStatus(Enum):
    SelectingSolver = 0
    SolverSelectedStakeNotRemoved = 1
    SolverSelected = 2
    SolutionSubmitted = 3
    Disputed = 4
    VerifiersSelected = 5
    Verified = 6
    ResolvedCorrect = 7
    ResolvedIncorrect = 8

class Vote(Enum):
    NoVote = 0
    SolutionCorrect = 1
    SolutionIncorrect = 2

class PoolState(Enum):
    Unlocked = 0
    Locked = 1
    SelectionsFinalizing = 2

# Define Task named tuple
Task = namedtuple('Task', [
    'status', 'submitter', 'solver', 'timeStatusChanged', 'selectionId', 'numVerifiers', 
    'selectedStakeId', 'params', 'postedSolution', 'verificationRounds', 'verifierStake', 
    'disputerStake'
])


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
            logging.debug(f"Extracted error selector {selector} for {name}")

def load_error_selectors(web3, abi_dir='abis'):
    error_selectors = {}
    for root, dirs, files in os.walk(abi_dir):
        for file in files:
            if file.endswith('.json'):
                contract_path = os.path.join(root, file)
                with open(contract_path, 'r') as abi_file:
                    abi = json.load(abi_file).get('abi', [])
                    extract_error_selectors(abi, web3, error_selectors)
    return error_selectors

def load_contracts(web3, subnet_addresses):
    abi_dir = 'abis'
    abis = load_all_abis(abi_dir)
    contracts = {}

    for task, address in subnet_addresses.items():
        contract_name = 'SubnetManager'
        if contract_name in abis:
            abi = abis[contract_name]
            contracts[task] = web3.eth.contract(address=address, abi=abi)
            logging.info(f"Loaded contract for {task} with address {address}")
        else:
            logging.error(f"Contract ABI not found for {contract_name}")

    error_selectors = load_error_selectors(web3, abi_dir)

    return abis, contracts, error_selectors

def load_all_abis(abi_dir):
    abis = {}
    for root, dirs, files in os.walk(abi_dir):
        for file in files:
            if file.endswith('.json'):
                contract_name = file.split('.')[0]
                with open(os.path.join(root, file), 'r') as abi_file:
                    abi = json.load(abi_file).get('abi', [])
                    abis[contract_name] = abi
    return abis

def load_abi(name):
    abi_dir = 'abis'
    contract_path = os.path.join(abi_dir, f'{name}.sol', f'{name}.json') 
    with open(contract_path, 'r') as abi_file:
        return json.load(abi_file).get('abi', [])

def upload_tensor(tensor, local_storage_dir):
    local_file_path = os.path.join(local_storage_dir, f'{int(time.time())}.pt')
    torch.save(tensor, local_file_path)
    return f'file://{local_file_path}'

def download_file(url):
    response = requests.get(url)
    return torch.load(BytesIO(response.content))

def initialize_distributed_environment(backend, master_addr='localhost', master_port=None):
    if master_port is None:
        master_port = str(12356 + os.getpid() % 10000)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

def handle_contract_custom_error(web3, error_selectors, e):
    try:
        error_bytes = bytes.fromhex(e.data[2:])
        selector = '0x' + error_bytes[:4].hex().lower()
        data = error_bytes[4:]

        if selector in error_selectors:
            error_info = error_selectors[selector]
            error_name = error_info['name']
            inputs = error_info['inputs']
            decoded_params = web3.codec.decode([input['type'] for input in inputs], data)
            param_str = ', '.join(f"{input['name']}: {value}" for input, value in zip(inputs, decoded_params))
            logging.error(f"Contract Custom Error {error_name}: {param_str}")
            raise ValueError(f"Contract Custom Error {error_name}: {param_str}")
        else:
            logging.error(f"Unknown error with selector {selector} and data {data.hex()}")
            raise ValueError(f"Unknown error with selector {selector} and data {data.hex()}")
    except Exception as decode_err:
        logging.error(f"Failed to decode error data: {e.data}. Error: {decode_err}")
        raise


def decode_custom_error(web3, error_selectors, error_bytes):
    try:
        selector = '0x' + error_bytes[:4].hex().lower()
        data = error_bytes[4:]

        if selector in error_selectors:
            error_info = error_selectors[selector]
            error_name = error_info['name']
            inputs = error_info['inputs']
            decoded_params = web3.codec.decode([input['type'] for input in inputs], data)
            param_str = ', '.join(f"{input['name']}: {value}" for input, value in zip(inputs, decoded_params))
            return f"Error {error_name}: {param_str}"
        else:
            return f"Unknown error with selector {selector} and data {data.hex()}"
    except Exception as e:
        logging.error(f"Error decoding message chunk: {e}")
        raise

def transact_with_contract_function(w3, contract_function, transaction_params):
    try:
        txn_hash = contract_function.transact(transaction_params)
        return w3.eth.wait_for_transaction_receipt(txn_hash)
    except web3.exceptions.ContractCustomError as e:
        error_selectors = load_error_selectors(w3)
        handle_contract_custom_error(w3, error_selectors, e)
        raise
    except Exception as e:
        logging.error(f"Unexpected error during transaction: {e}")
        raise
