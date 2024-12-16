import torch
import os
import logging
import time
from io import BytesIO
import requests
from enum import Enum
from collections import namedtuple
from .device import device
import asyncio
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
    SelectingSolver = "SelectingSolver"
    SolverSelected = "SolverSelected"
    Checking = "Checking"  # Newly added status
    ResolvedCorrect = "ResolvedCorrect"
    ResolvedIncorrect = "ResolvedIncorrect"

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