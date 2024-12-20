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
import io

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
    Checking = "Checking"
    ResolvedCorrect = "ResolvedCorrect"
    ResolvedIncorrect = "ResolvedIncorrect"
    SanityCheckPending = "SanityCheckPending"

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


class NoMoreDataException(Exception):
    pass

async def read_streamed_content(response, chunk_timeout):
    """
    Read all content from the response stream with a chunk timeout.
    If the stream ends prematurely, this function will return whatever it got.
    """
    content = io.BytesIO()
    total_chunks = 0
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(response.content.read(65536), timeout=chunk_timeout)
            except asyncio.TimeoutError:
                logging.error("read_streamed_content: Chunk download timed out.")
                raise

            if not chunk:
                # End of stream
                break
            content.write(chunk)
            total_chunks += 1

        data_size = content.getbuffer().nbytes
        logging.info(f"read_streamed_content: Finished reading. Total chunks: {total_chunks}, total size: {data_size} bytes")
        content.seek(0)
        return content.getvalue()
    except Exception as e:
        logging.error(f"read_streamed_content: Exception occurred while reading stream: {e}")
        raise


async def download_file(
    url,
    retries=3,
    backoff=1,
    chunk_timeout=20,  # Increased from 5 to allow slower downloads
    download_type='batch_targets',
    tensor_name=None
):
    """
    Download a file (e.g., model tensor or batch data) from `url` with retries,
    backoff, and a configurable chunk_timeout. Supports special handling for
    'no more data' scenarios.
    """
    params = {'tensor_name': tensor_name} if tensor_name else None

    for attempt in range(1, retries + 1):
        try:
            # Set a total=None so that only the read operations time out, not the entire request
            timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=chunk_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    # Handle 404 explicitly: might mean "no more data"
                    if response.status == 404:
                        try:
                            err_data = await response.json()
                            if 'error' in err_data and 'no more data' in err_data['error'].lower():
                                logging.info(f"No more data scenario confirmed at URL: {url}")
                                raise NoMoreDataException("No more data available from dataset.")
                        except Exception:
                            pass
                        # If not a known "no more data" scenario, raise a standard error
                        raise Exception(f"File not found (404) at {url}")

                    if response.status != 200:
                        raise Exception(f"HTTP error {response.status} from {url}")

                    # Handle tensor download lock if needed
                    if download_type == 'tensor':
                        async with tensor_download_lock:
                            tensor_download_event.clear()
                            try:
                                content = await read_streamed_content(response, chunk_timeout)
                            finally:
                                tensor_download_event.set()
                    elif download_type == 'batch_targets':
                        # Just ensure no tensor download currently blocks
                        async with tensor_download_lock:
                            pass
                        content = await read_streamed_content(response, chunk_timeout)
                    else:
                        raise ValueError("Invalid download_type. Must be 'tensor' or 'batch_targets'.")

                    if len(content) == 0:
                        logging.error(f"Downloaded file from {url} is empty on attempt {attempt}. Retrying...")
                        if attempt == retries:
                            raise Exception("Downloaded file is empty after all retries.")
                        await asyncio.sleep(backoff * attempt)
                        continue

                    # Successfully got non-empty data, load as tensor
                    return torch.load(io.BytesIO(content), map_location='cpu')

        except NoMoreDataException:
            # Immediately propagate 'no more data' scenario
            raise
        except (asyncio.TimeoutError, aiohttp.ClientError, aiohttp.ClientPayloadError) as e:
            logging.error(f"Attempt {attempt}: Network error while downloading {url}: {e}")
            if attempt == retries:
                raise Exception(f"Failed to download file after {retries} attempts due to network errors: {e}")
            await asyncio.sleep(backoff * attempt)
        except Exception as e:
            logging.error(f"Attempt {attempt}: Unexpected error: {e}")
            if attempt == retries:
                raise
            await asyncio.sleep(backoff * attempt)
