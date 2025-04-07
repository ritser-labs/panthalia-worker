# spl/worker/uploads.py
import asyncio
import requests
import logging
from tqdm import tqdm
import io
import torch
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor
from .config import args
import safetensors.torch  # Import safetensors.torch to use its save function
import sys

logger = logging.getLogger(__name__)

class DummyPbar:
    def __init__(self):
        self.n = 0
    def update(self, n):
        self.n += n
    def close(self):
        pass

def create_callback(encoder, pbar):
    def callback(monitor):
        pbar.update(monitor.bytes_read - pbar.n)
    return callback

async def upload_tensor(tensor, tensor_name, sot_url, retries=3, backoff=1):
    # Prepare the dictionary to be saved
    if isinstance(tensor, torch.Tensor):
        save_dict = {"tensor": tensor}
    elif isinstance(tensor, dict):
        save_dict = tensor
    else:
        raise ValueError("tensor must be a torch.Tensor or a dict of tensors")
    
    # Use safetensors.torch.save to get raw bytes
    data_bytes = safetensors.torch.save(save_dict)
    mem_buf = io.BytesIO(data_bytes)
    mem_buf.seek(0)

    encoder_obj = MultipartEncoder(
        fields={
            'tensor': (tensor_name, mem_buf, 'application/octet-stream'),
            'label': tensor_name
        }
    )

    # Choose the progress bar based on GUI mode and stdout availability.
    if not args.gui and sys.stdout is not None and hasattr(sys.stdout, "write"):
        pbar = tqdm(total=encoder_obj.len, unit='B', unit_scale=True, desc='Uploading', file=sys.stdout)
    else:
        pbar = DummyPbar()
    
    monitor = MultipartEncoderMonitor(encoder_obj, create_callback(encoder_obj, pbar))
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
