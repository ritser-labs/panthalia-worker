# spl/common.py

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
from safetensors.torch import load as safetensors_load
import aiofiles

SOT_PRIVATE_PORT = 5001

DB_PORT = '5432'

MAX_SUBMIT_TASK_RETRY_DURATION = 300

MIN_REMAINING_TIME_SECONDS = 3

CHUNK_SIZE = 64 * 1024 * 1024

SLEEP_TIME = 1

TENSOR_NAME = 'model'

current_dir = os.path.dirname(os.path.abspath(__file__))
abi_dir = os.path.join(current_dir, 'abis')

pending_transactions = {}

async def wait_for_health(url, timeout=1200):
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

class TaskStatus(Enum):
    SelectingSolver = "SelectingSolver"
    SolverSelected = "SolverSelected"
    Checking = "Checking"
    ResolvedCorrect = "ResolvedCorrect"
    ResolvedIncorrect = "ResolvedIncorrect"
    SolutionSubmitted = "SolutionSubmitted"
    ReplicationPending = "ReplicationPending"

class Vote(Enum):
    NoVote = 0
    SolutionCorrect = 1
    SolutionIncorrect = 2

class PoolState(Enum):
    Unlocked = 0
    Locked = 1
    SelectionsFinalizing = 2

Task = namedtuple('Task', [
    'status', 'submitter', 'solver', 'timeStatusChanged', 'subSelectionId', 'numVerifiers',
    'params', 'postedSolution', 'verificationRounds', 'verifierStake',
    'disputerStake'
])

def get_future_version_number(tensor_version_interval):
    return (int(time.time()) // tensor_version_interval + 1) * tensor_version_interval

def get_current_version_number(tensor_version_interval):
    """
    Return the 'rounded-down' current version based on time.
    If time.time()=1699999999.2 and interval=36 => 1699999999//36=47222222 => 47222222*36=1700000000 - 72 => etc.
    """
    now = int(time.time())
    result = (now // tensor_version_interval) * tensor_version_interval
    logging.debug(
        f"[get_current_version_number] now={now}, tensor_version_interval={tensor_version_interval}, "
        f"result={result}"
    )
    return result

current_global_state = None
state_change_event = asyncio.Event()
state_changing = False

def generate_wallets(num_wallets):
    wallets = []
    for _ in range(num_wallets):
        account = Account.create()
        wallets.append({'private_key': account._private_key.hex(), 'address': account.address})
    return wallets

tensor_download_lock = asyncio.Lock()
tensor_download_event = asyncio.Event()
tensor_download_event.set()

async def download_with_timeout(response, chunk_size=1024 * 1024, chunk_timeout=5, download_type='batch_targets'):
    start_time = time.time()
    content = BytesIO()
    content_length = response.headers.get('Content-Length', None)
    if content_length:
        total_size = int(content_length)
        logging.debug(f"Total file size (Content-Length): {total_size} bytes")
    else:
        total_size = None
        logging.debug("No Content-Length header. Possibly chunked transfer.")

    downloaded_size = 0
    next_progress = 0.1

    while True:
        try:
            chunk = await asyncio.wait_for(response.content.read(chunk_size), timeout=chunk_timeout)
        except asyncio.TimeoutError:
            logging.error(f"Chunk download timed out after {chunk_timeout} seconds")
            raise

        if not chunk:
            logging.debug("No more chunks left. Finished download.")
            break

        if download_type == 'batch_targets':
            await tensor_download_event.wait()

        content.write(chunk)
        downloaded_size += len(chunk)

        if total_size:
            progress = downloaded_size / total_size
            if progress >= next_progress:
                logging.info(f"Downloaded {int(progress*100)}%")
                next_progress += 0.1

    content.seek(0)
    if total_size and downloaded_size != total_size:
        logging.error(f"Downloaded size {downloaded_size} vs. expected {total_size}.")
        raise Exception(f"Incomplete download: expected {total_size} but got {downloaded_size} bytes")

    end_time = time.time()
    logging.info(f"Download completed in {end_time - start_time:.2f} s; total size={downloaded_size} bytes")
    return content

class NoMoreDataException(Exception):
    pass

async def read_streamed_content(response, chunk_timeout):
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
                break
            content.write(chunk)
            total_chunks += 1
        data_size = content.getbuffer().nbytes
        logging.info(f"Finished reading. Total chunks={total_chunks}, size={data_size} bytes")
        content.seek(0)
        return content.getvalue()
    except Exception as e:
        logging.error(f"read_streamed_content error: {e}")
        raise

async def download_file(
    url,
    retries=3,
    backoff=1,
    chunk_timeout=300,
    download_type='batch_targets',
    tensor_name=None,
    local_file_path=None
):
    """
    Downloads a tensor file from `url` with optional retries/backoff.
    If local_file_path is provided, writes the downloaded bytes to that file and returns the path.
    Otherwise, returns the tensor loaded via safetensors_load_file.

    :param url: The URL to download the file from.
    :param retries: Number of download attempts.
    :param backoff: Delay (in seconds) multiplier between attempts.
    :param chunk_timeout: Timeout (in seconds) when reading a chunk.
    :param download_type: Either 'tensor' or 'batch_targets' (controls locking behavior).
    :param tensor_name: Optional tensor name to send as a URL parameter.
    :param local_file_path: If provided, the path to save the downloaded file.
    :return: Either the loaded tensor (if local_file_path is None) or the local_file_path.
    :raises: Exception or NoMoreDataException on failure.
    """
    params = {'tensor_name': tensor_name} if tensor_name else None
    for attempt in range(1, retries + 1):
        try:
            timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=chunk_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 404:
                        try:
                            err_data = await response.json()
                            if 'error' in err_data and 'no more data' in err_data['error'].lower():
                                logging.info(f"No more data scenario confirmed at {url}")
                                raise NoMoreDataException("No more data from dataset.")
                        except Exception:
                            pass
                        raise Exception(f"File not found (404) at {url}")
                    if response.status != 200:
                        raise Exception(f"HTTP error {response.status} from {url}")

                    if download_type == 'tensor':
                        async with tensor_download_lock:
                            tensor_download_event.clear()
                            try:
                                content = await read_streamed_content(response, chunk_timeout)
                            finally:
                                tensor_download_event.set()
                    elif download_type == 'batch_targets':
                        async with tensor_download_lock:
                            pass
                        content = await read_streamed_content(response, chunk_timeout)
                    else:
                        raise ValueError("Invalid download_type: must be 'tensor' or 'batch_targets'")

                    if len(content) == 0:
                        logging.error(f"Downloaded file from {url} is empty on attempt {attempt}. Retrying...")
                        if attempt == retries:
                            raise Exception("Downloaded file is empty after all retries.")
                        await asyncio.sleep(backoff * attempt)
                        continue

                    # If a local file path is provided, write the downloaded content there.
                    if local_file_path is not None:
                        async with aiofiles.open(local_file_path, 'wb') as f:
                            await f.write(content)
                        return local_file_path
                    else:
                        # Otherwise, load and return the tensor using safetensors.
                        return safetensors_load(BytesIO(content))
        except NoMoreDataException:
            raise
        except (asyncio.TimeoutError, aiohttp.ClientError, aiohttp.ClientPayloadError) as e:
            logging.error(f"Attempt {attempt}: Network error while downloading {url}: {e}")
            if attempt == retries:
                raise Exception(f"Failed after {retries} attempts: {e}")
            await asyncio.sleep(backoff * attempt)
        except Exception as e:
            logging.error(f"Attempt {attempt}: Unexpected error: {e}")
            if attempt == retries:
                raise
            await asyncio.sleep(backoff * attempt)