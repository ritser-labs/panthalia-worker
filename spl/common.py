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


async def decode_revert_reason(web3, revert_reason):
    try:
        # Ensure the revert reason starts with '0x' and remove it
        if revert_reason.startswith('0x'):
            revert_reason = revert_reason[2:]

        if len(revert_reason) < 8:
            logging.error(f"Invalid revert reason length: {revert_reason}")
            return f"Unknown revert reason: {revert_reason}"

        selector = '0x' + revert_reason[:8]
        encoded_data = revert_reason[8:]

        # Define the selector for the 'Error(string)' type
        error_selector = '0x08c379a0'

        # Check if the selector matches 'Error(string)'
        if selector == error_selector:
            # Decode the error message
            try:
                # Error(string) has a single argument of type string
                data_bytes = bytes.fromhex(encoded_data)
                decoded = decode(['string'], data_bytes)
                decoded_reason = f"Error: {decoded[0]}"
                return decoded_reason
            except Exception as e:
                logging.error(f"Error decoding revert reason: {e}")
                return f"Error decoding revert reason: {revert_reason}"
        
        # Define the selector for the 'Panic(uint256)' type
        panic_selector = '0x4e487b71'

        # Check if the selector matches 'Panic(uint256)'
        if selector == panic_selector:
            # Decode the panic code
            try:
                # Panic(uint256) has a single argument of type uint256
                data_bytes = bytes.fromhex(encoded_data)
                decoded = decode(['uint256'], data_bytes)
                decoded_reason = f"Panic: {hex(decoded[0])}"
                return decoded_reason
            except Exception as e:
                logging.error(f"Error decoding panic code: {e}")
                return f"Error decoding panic code: {revert_reason}"
        
        error_selectors = load_error_selectors(web3)

        if selector in error_selectors:
            error_info = error_selectors[selector]
            name = error_info['name']
            types = [input['type'] for input in error_info['inputs']]
            data_bytes = bytes.fromhex(encoded_data)
            decoded = decode(types, data_bytes)
            decoded_reason = f"{name}({', '.join(map(str, decoded))})"
            return decoded_reason
        else:
            logging.error(f'Selector {selector} not found in error selectors')
            logging.error(f'Selectors: {error_selectors}')
            return f"Unknown revert reason: {revert_reason}"
    except Exception as e:
        logging.error(f"Exception in decode_revert_reason: {e}")
        return f"Error decoding revert reason: {revert_reason}"

async def approve_token_once(web3, token_contract, private_key, spender_address, amount):
    account = web3.eth.account.from_key(private_key)
    current_allowance = await token_contract.functions.allowance(account.address, spender_address).call()
    if current_allowance < amount:
        receipt = await async_transact_with_contract_function(web3, token_contract, 'approve', private_key, spender_address, amount)
        logging.info(f"Approved token transaction receipt: {receipt}")
    else:
        logging.info("Current allowance is sufficient, no need to approve more tokens.")

async def deposit_stake_without_approval(web3, pool_contract, private_key, subnet_id, group, worker_address, stake_amount, max_stakes, max_retries=10):
    stakes_deposited = await pool_contract.functions.getStakeIds(subnet_id, group, worker_address).call()
    number_of_stakes_to_deposit = max_stakes - len(stakes_deposited)
    
    logging.info(f'Depositing {number_of_stakes_to_deposit} stakes for {worker_address}...')

    if number_of_stakes_to_deposit > 0:
        for attempt in range(max_retries):  # Use max_retries variable
            try:
                await wait_for_state_change(web3, pool_contract, PoolState.Unlocked.value, private_key)
                receipt = await async_transact_with_contract_function(
                    web3, 
                    pool_contract, 
                    'depositMultipleStakes', 
                    private_key, 
                    subnet_id, 
                    group, 
                    number_of_stakes_to_deposit
                )
                logging.info(f"depositMultipleStakes transaction receipt: {receipt}")
                logging.info(f"Deposited {number_of_stakes_to_deposit} stakes for {worker_address}, total stakes: {max_stakes}")
                break  # Exit loop if successful
            except Exception as e:
                logging.error(f"Failed to deposit stakes on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise  # Rethrow the exception after the last attempt
                await asyncio.sleep(SLEEP_TIME)  # Wait before retrying

# Global state tracking variable
current_global_state = None
# Event to notify all waiting tasks of a state change
state_change_event = asyncio.Event()
# Flag to ensure only one state-changing transaction at a time
state_changing = False

async def update_current_global_state(pool):
    global current_global_state
    current_global_state = PoolState(await pool.functions.state().call())
    logging.info(f'New global state: {current_global_state.name}')
    state_change_event.set()
    state_change_event.clear()

async def wait_for_state_change(web3, pool, target_state, private_key):
    max_retries = 300
    retries = 0

    global current_global_state
    global state_change_event
    global state_changing

    await update_current_global_state(pool)

    while retries < max_retries:
        # Check if the current state matches the target state
        if current_global_state == PoolState(target_state):
            # Check if there are at least MIN_REMAINING_TIME_SECONDS remaining in the target state
            latest_block = await web3.eth.get_block('latest')
            current_time = latest_block['timestamp']
            if target_state == PoolState.Unlocked.value:
                remaining_time = (await pool.functions.lastStateChangeTime().call() + 
                                  (await pool.functions.UNLOCKED_MIN_PERIOD().call())) - current_time
                if remaining_time >= MIN_REMAINING_TIME_SECONDS:
                    logging.info(f'Pool state is now {PoolState(target_state).name} with {remaining_time} seconds remaining')
                    return
                else:
                    logging.info(f'Not enough remaining time {remaining_time} seconds in {PoolState(target_state).name}, sleeping and rechecking state')
                    await asyncio.sleep(max(remaining_time, SLEEP_TIME))
                    await update_current_global_state(pool)
            else:
                logging.info(f'Pool state is now {PoolState(target_state).name}')
                return

           

        # Try to perform the state transition if needed
        if not state_changing:
            state_changing = True
            try:
                # Perform state transition based on the current state
                if current_global_state == PoolState.Unlocked:
                    logging.info("Triggering lockGlobalState to change state to Locked")
                    await trigger_lock_global_state(web3, pool, private_key)
                elif current_global_state == PoolState.Locked:
                    logging.info("Triggering finalizeSelections to change state to SelectionsFinalizing")
                    await finalize_selections(web3, pool, private_key)
                elif current_global_state == PoolState.SelectionsFinalizing:
                    logging.info("Triggering selectStakes to change state to Unlocked")
                    await select_stakes(web3, pool, private_key)

                # Update the global state after the transaction
                await update_current_global_state(pool)
            except Exception as e:
                logging.info(f"Caught error changing state: {e}")
                await asyncio.sleep(SLEEP_TIME)
            finally:
                state_changing = False
                await update_current_global_state(pool)
        else:
            # Wait for the state to change
            await state_change_event.wait()

        retries += 1

    raise RuntimeError(f"Failed to change state to {PoolState(target_state).name} after multiple attempts")

async def finalize_selections(web3, pool, private_key):
    vrf_coordinator_address = await pool.functions.vrfCoordinator().call()
    vrf_coordinator = web3.eth.contract(address=vrf_coordinator_address, abi=load_abi('MockVRFCoordinator'))
    vrf_request_id = await pool.functions.vrfRequestId().call()
    receipt = await async_transact_with_contract_function(web3, vrf_coordinator, 'fulfillRandomWords', private_key, vrf_request_id, attempts=1)
    logging.info(f"fulfillRandomWords transaction receipt: {receipt}")

async def trigger_lock_global_state(web3, pool, private_key):
    unlocked_min_period = await pool.functions.UNLOCKED_MIN_PERIOD().call()
    last_state_change_time = await pool.functions.lastStateChangeTime().call()
    latest_block = await web3.eth.get_block('latest')
    current_time = latest_block['timestamp']
    remaining_time = (last_state_change_time + unlocked_min_period) - current_time

    if remaining_time > 0:
        logging.info(f"Waiting for {remaining_time} seconds until UNLOCKED_MIN_PERIOD is over")
        await asyncio.sleep(remaining_time + 1)
    else:
        logging.info("UNLOCKED_MIN_PERIOD is already over, proceeding with lockGlobalState")

    try:
        receipt = await async_transact_with_contract_function(web3, pool, 'lockGlobalState', private_key, attempts=1)
        logging.info(f"lockGlobalState transaction receipt: {receipt}")
    except Exception as e:
        logging.error(f"Error triggering lock global state: {e}")
        raise

async def select_stakes(web3, pool, private_key):
    try:
        receipt = await async_transact_with_contract_function(web3, pool, 'selectStakes', private_key, attempts=1)
        logging.info(f"selectStakes transaction receipt: {receipt}")
    except Exception as e:
        logging.error(f"Error triggering remove global lock: {e}")
        raise

# Add a loop to keep checking if the RPC is available
async def wait_for_rpc_available(web3, retry_interval=5, max_retries=60):
    """
    Wait until the RPC connection is available.

    :param web3: The web3 instance to check.
    :param retry_interval: Time (in seconds) to wait between retries.
    :param max_retries: Maximum number of retries before giving up.
    """
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Try to fetch the current block number as a simple check
            await web3.eth.block_number
            logging.info("RPC is available.")
            return True
        except Exception as e:
            logging.error(f"RPC not available. Retry {retry_count + 1}/{max_retries}: {e}")
            retry_count += 1
            await asyncio.sleep(retry_interval)
    logging.error(f"RPC not available after {max_retries} retries. Exiting...")
    return False

async def fund_wallets(web3, private_key, wallets, token_contract, amount_eth, amount_token, distributor_contract_address):
    logging.info('Funding wallets')
    deployer_address = web3.eth.account.from_key(private_key).address

    distributor_contract = web3.eth.contract(address=distributor_contract_address, abi=load_abi('Distributor'))

    # Distribute Ether
    recipients = [wallet['address'] for wallet in wallets]
    eth_amounts = [web3.to_wei(amount_eth, 'ether')] * len(wallets)

    distribute_eth_tx = await distributor_contract.functions.distributeEther(recipients, eth_amounts).build_transaction({
        'from': deployer_address,
        'nonce': await web3.eth.get_transaction_count(deployer_address),
        'gas': 3000000,  # Adjust as needed
        'gasPrice': await web3.eth.gas_price,
        'value': sum(eth_amounts)
    })
    signed_eth_tx = web3.eth.account.sign_transaction(distribute_eth_tx, private_key)
    eth_tx_hash = await web3.eth.send_raw_transaction(signed_eth_tx.rawTransaction)
    receipt = await web3.eth.wait_for_transaction_receipt(eth_tx_hash)
    if receipt['status'] != 1:
        raise Exception(f"Error distributing Ether: {receipt}")
    logging.info('Ether distribution completed')
    
    if not hasattr(fund_wallets, 'approval_submitted') or not fund_wallets.approval_submitted:
        # Approve the distributor contract to spend the token
        max_tokens = 1000000000000000000000000000000  # 1e27
        await async_transact_with_contract_function(
            web3,
            token_contract,
            'approve',
            private_key,
            *[distributor_contract_address, max_tokens],
        )
        logging.info('Token approval completed')
        fund_wallets.approval_submitted = True
    
    
    await async_transact_with_contract_function(
        web3,
        distributor_contract,
        'distributeTokens',
        private_key,
        *[token_contract.address, recipients, [amount_token] * len(wallets)],
    )
    logging.info('Token distribution completed')


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
    logging.info(f"Download completed successfully in {start_time - end_time:.2f} seconds. Total size: {downloaded_size} bytes")
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