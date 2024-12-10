# spl/worker/uploads.py
import asyncio
import requests
import logging
from tqdm import tqdm
from io import BytesIO
import torch
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor
from .config import args

logger = logging.getLogger(__name__)

def create_callback(encoder, pbar):
    def callback(monitor):
        pbar.update(monitor.bytes_read - pbar.n)
    return callback

async def upload_tensor(tensor, tensor_name, sot_url, retries=3, backoff=1):
    tensor_bytes = BytesIO()
    torch.save(tensor, tensor_bytes)
    tensor_bytes.seek(0)

    encoder = MultipartEncoder(
        fields={
            'tensor': (tensor_name, tensor_bytes, 'application/octet-stream'),
            'label': tensor_name
        }
    )

    pbar = tqdm(total=encoder.len, unit='B', unit_scale=True, desc='Uploading')
    monitor = MultipartEncoderMonitor(encoder, create_callback(encoder, pbar))

    headers = {'Content-Type': monitor.content_type}

    for attempt in range(1, retries + 1):
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f'{sot_url}/upload_tensor',
                    data=monitor,
                    headers=headers,
                    timeout=300
                )
            )
            pbar.close()

            if response.status_code == 200:
                return sot_url + response.json().get('tensor_url')
            else:
                raise RuntimeError(f"Failed to upload tensor: {response.text}")

        except requests.exceptions.Timeout:
            logger.error(f"Attempt {attempt}: Upload request timed out.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Attempt {attempt}: Upload request failed: {e}")

        if attempt < retries:
            await asyncio.sleep(backoff * attempt)

    raise RuntimeError(f"Failed to upload tensor {tensor_name} after {retries} attempts")
