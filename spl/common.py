import torch
import os
import json
import logging
import time
from io import BytesIO
import requests
from enum import Enum
from collections import namedtuple
from web3.datastructures import AttributeDict
from web3.exceptions import ContractLogicError
from .device import device
import math
import asyncio
from hexbytes import HexBytes
from eth_abi import decode
import threading
import aiohttp
from eth_account import Account

SOT_PRIVATE_PORT = 5001

# Define the new tokenizer and model arguments
current_dir = os.path.dirname(os.path.abspath(__file__))

DB_PORT = '5432'

MAX_SUBMIT_TASK_RETRY_DURATION = 300

MIN_REMAINING_TIME_SECONDS = 3

CHUNK_SIZE = 64 * 1024 * 1024

SLEEP_TIME = 1

TENSOR_NAME = 'model'

current_dir = os.path.dirname(os.path.abspath(__file__))
abi_dir = os.path.join(current_dir, 'abis')

# Global variables for transaction management by private key
pending_transactions = {}

async def wait_for_health(url, timeout=1200):  # Increased timeout to 20 minutes
    """Wait for the service to be available asynchronously."""
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < timeout:
            try:
                async with session.get(f"{url}/health") as response:
                    if response.status == 200:
                        logging.debug("Service is available.")
                        return True
            except aiohttp.ClientConnectionError as e:
                logging.debug(f"Waiting for service to be available... {e}")
            await asyncio.sleep(2)
    
    return False
# Define Enums
class TaskStatus(Enum):
    SelectingSolver = 0
    SolverSelected = 1
    SolutionSubmitted = 2
    Disputed = 3
    VerifiersSelected = 4
    Verified = 5
    ResolvedCorrect = 6
    ResolvedIncorrect = 7

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
    'status', 'submitter', 'solver', 'timeStatusChanged', 'subSelectionId', 'numVerifiers', 
    'params', 'postedSolution', 'verificationRounds', 'verifierStake', 
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

def load_error_selectors(web3):
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

    error_selectors = load_error_selectors(web3)

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

def get_future_version_number(tensor_version_interval):
    return (int(time.time()) // tensor_version_interval + 1) * tensor_version_interval

def get_current_version_number(tensor_version_interval):
    return (int(time.time()) // tensor_version_interval) * tensor_version_interval

def process_trace(trace):
    if isinstance(trace, AttributeDict):
        return {k: process_trace(v) for k, v in trace.items()}
    elif isinstance(trace, dict):
        return {k: process_trace(v) for k, v in trace.items()}
    elif isinstance(trace, list):
        return [process_trace(i) for i in trace]
    elif isinstance(trace, HexBytes):
        return trace.hex()
    else:
        return trace

async def get_debug_trace(web3, tx_hash):
    try:
        # Ensure tx_hash is in string format
        if isinstance(tx_hash, HexBytes):
            tx_hash = tx_hash.hex()
        elif isinstance(tx_hash, bytes):
            tx_hash = tx_hash.hex()
        
        trace = await web3.manager.request_blocking('debug_traceTransaction', [tx_hash])
        processed_trace = process_trace(trace)
        for key, value in processed_trace.items():
            logging.info(f"{key}: {value}")
        return processed_trace
    except Exception as e:
        logging.error(f"ERROR IN GET_DEBUG_TRACE: {e}")
        return {}

async def async_transact_with_contract_function(web3, contract, function_name, private_key, *args, value=0, attempts=5):
    global pending_transactions

    # Initialize the transaction state for the given private key if not already done
    if private_key not in pending_transactions:
        pending_transactions[private_key] = False


    # Use the condition to wait for the transaction to complete
    while pending_transactions[private_key]:
        await asyncio.sleep(SLEEP_TIME)
    # Set the transaction as pending for this private key
    pending_transactions[private_key] = True

    try:
        account = web3.eth.account.from_key(private_key)
        gas_price = await web3.eth.gas_price

        function = getattr(contract.functions, function_name)(*args)

        for attempt in range(attempts):  # Retry up to 5 times
            tx_hash = None
            try:
                nonce = await web3.eth.get_transaction_count(account.address)
                # Dynamically estimate gas
                estimated_gas = await function.estimate_gas({
                    'from': account.address,
                    'value': value
                })

                tx_params = {
                    'chainId': await web3.eth.chain_id,
                    'gas': estimated_gas,
                    'gasPrice': gas_price,
                    'nonce': nonce,
                    'value': value
                }

                tx = await function.build_transaction(tx_params)
                signed_tx = account.sign_transaction(tx)
                tx_hash = await web3.eth.send_raw_transaction(signed_tx.rawTransaction)
                receipt = await web3.eth.wait_for_transaction_receipt(tx_hash)

                if receipt['status'] == 0:
                    error_message = await decode_revert_reason(web3, await web3.eth.call({
                        'to': tx['to'],
                        'data': tx.get('input', b'')
                    }, receipt.blockNumber))
                    logging.error(f"Transaction failed: {error_message}")
                    raise ContractLogicError(f"Transaction failed: {error_message}")
                else:
                    return receipt
            except ContractLogicError as cle:
                # Extract revert reason from ContractLogicError
                try:
                    error_data = cle.args[0]  # Directly use the first argument
                    if isinstance(error_data, dict) and 'data' in error_data:
                        error_data = error_data['data']
                    decoded_error = await decode_revert_reason(web3, error_data)
                    logging.error(f"Contract logic error on attempt {attempt + 1} - {function_name}: {decoded_error}")
                except Exception as e:
                    logging.error(f"Failed to decode contract logic error - {function_name}: {cle}, original error: {e}")
                if attempt == attempts - 1:
                    raise
            except Exception as e:
                logging.error(f"Error on attempt {attempt + 1} - {function_name}: {e}")
                if attempt == attempts - 1:
                    raise

            # Wait for the next block before retrying
            await wait_for_block(web3)

    finally:
        pending_transactions[private_key] = False


async def wait_for_block(web3):
    block_filter = await web3.eth.filter('latest')
    while True:
        new_entries = await block_filter.get_new_entries()
        if new_entries:
            return new_entries[0]
        await asyncio.sleep(SLEEP_TIME)

# Global state tracking variable
current_global_state = None
# Event to notify all waiting tasks of a state change
state_change_event = asyncio.Event()
# Flag to ensure only one state-changing transaction at a time
state_changing = False


def generate_wallets(num_wallets):
    wallets = []
    for _ in range(num_wallets):
        account = Account.create()
        wallets.append({'private_key': account._private_key.hex(), 'address': account.address})
    return wallets


# Initialize a lock to prioritize tensor downloads
tensor_download_lock = asyncio.Lock()

# Initialize an event to signal tensor download in progress
tensor_download_event = asyncio.Event()

tensor_download_event.set()

async def download_with_timeout(response, chunk_size=1024 * 1024, chunk_timeout=5, download_type='batch_targets'):
    """
    Downloads data from the response stream with a timeout for each chunk.
    Pauses if a tensor download is in progress.

    Args:
        response: The aiohttp response object.
        chunk_size: The size of each chunk to download.
        chunk_timeout: Timeout for each chunk in seconds.
        download_type: Type of download ('tensor' or 'batch_targets').

    Returns:
        A BytesIO object containing the downloaded data.
    """
    start_time = time.time()
    content = BytesIO()
    
    # Get the content length from the header, if available
    content_length = response.headers.get('Content-Length', None)
    if content_length:
        total_size = int(content_length)
        logging.debug(f"Total file size (Content-Length): {total_size} bytes")
    else:
        # No Content-Length header, could be chunked transfer encoding
        total_size = None
        logging.debug("No Content-Length header. Assuming chunked transfer encoding.")
    
    downloaded_size = 0
    next_progress = 0.1

    # Fetch each chunk with a timeout
    while True:
        try:
            chunk = await asyncio.wait_for(response.content.read(chunk_size), timeout=chunk_timeout)
        except asyncio.TimeoutError:
            logging.error(f"Chunk download timed out after {chunk_timeout} seconds")
            raise

        if not chunk:
            # No more chunks left to download
            logging.debug("No more chunks to download. Download finished.")
            break

        if download_type == 'batch_targets':
            await tensor_download_event.wait()

        content.write(chunk)
        downloaded_size += len(chunk)
        #logging.debug(f"Downloaded chunk size: {len(chunk)} bytes. Total downloaded: {downloaded_size} bytes")

        # If we have the total size, we can log progress
        if total_size:
            progress = downloaded_size / total_size
            if progress >= next_progress:
                logging.info(f"Downloaded {int(progress * 100)}%")
                next_progress += 0.1

    content.seek(0)  # Reset the stream position

    # Validate that the entire content was downloaded, if we know the total size
    if total_size and downloaded_size != total_size:
        logging.error(f"Downloaded size ({downloaded_size}) does not match expected size ({total_size}).")
        raise Exception(f"Incomplete download: expected {total_size} bytes but got {downloaded_size} bytes")

    end_time = time.time()
    logging.info(f"Download completed successfully in {end_time - start_time:.2f} seconds. Total size: {downloaded_size} bytes")
    return content

async def download_file(url, retries=3, backoff=1, chunk_timeout=5, download_type='batch_targets', tensor_name=None):
    """
    Downloads a file with retry logic and prioritizes tensor downloads over batch/targets downloads.
    
    Args:
        url (str): The URL to download the file from.
        retries (int): Number of retry attempts.
        backoff (int): Backoff factor for retries.
        chunk_timeout (int): Timeout for each chunk in seconds.
        download_type (str): Type of download ('tensor' or 'batch_targets').
    
    Returns:
        torch.Tensor: The downloaded tensor.
    """
    params = {'tensor_name': tensor_name} if tensor_name else None
    for attempt in range(1, retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()

                    if download_type == 'tensor':
                        # Acquire tensor_download_lock to prioritize tensor downloads
                        async with tensor_download_lock:
                            # Signal that a tensor download is in progress
                            tensor_download_event.clear()
                            try:
                                content = await download_with_timeout(response, chunk_size=1024 * 1024, chunk_timeout=chunk_timeout, download_type=download_type)
                            finally:
                                # Clear the event after tensor download is complete
                                tensor_download_event.set()
                    elif download_type == 'batch_targets':
                        # Wait for any ongoing tensor download to finish
                        async with tensor_download_lock:
                            pass  # Simply wait until tensor_download_lock is free
                        content = await download_with_timeout(response, chunk_size=1024 * 1024, chunk_timeout=chunk_timeout, download_type=download_type)
                    else:
                        raise ValueError("Invalid download_type specified.")

                    return torch.load(content)

        except asyncio.TimeoutError:
            logging.error(f"Attempt {attempt}: Chunk download timed out.")
        except aiohttp.ClientError as e:
            logging.error(f"Attempt {attempt}: Client error: {e}")
        except Exception as e:
            logging.error(f"Attempt {attempt}: Unexpected error: {e}")

        if attempt < retries:
            await asyncio.sleep(backoff * attempt)

    raise Exception(f"Failed to download file after {retries} attempts")