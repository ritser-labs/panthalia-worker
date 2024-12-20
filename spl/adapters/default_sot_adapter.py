# spl/adapters/default_sot_adapter.py
import os
import logging
import asyncio
import aiofiles
import torch
import time
import random
import io
from ..common import device, TENSOR_NAME
from .sot_adapter import BaseSOTAdapter
from ..util.json import load_json, save_json
from ..common import device
from ..util.tensors import (
    initialize_tensor,
    initialize_all_tensors as utils_initialize_all_tensors,
)

CHUNK_SIZE = 1024 * 1024  # 1MB chunk size, adjust as needed

class DefaultSOTAdapter(BaseSOTAdapter):
    def __init__(self, model_adapter, dataset, state_dir, file_locks=None):
        self.model_adapter = model_adapter
        self.dataset = dataset
        self.state_dir = state_dir
        self.temp_dir = os.path.join(self.state_dir, 'temp')

        self.file_locks = file_locks if file_locks is not None else {
            'block_timestamps': asyncio.Lock(),
            'num_updates': asyncio.Lock(),
            'iteration_number': asyncio.Lock(),
            'last_future_version_number': asyncio.Lock(),
            'latest_loss': asyncio.Lock()
        }

        self.block_timestamps_file = os.path.join(self.state_dir, 'block_timestamps.json')
        self.num_updates_file = os.path.join(self.state_dir, 'num_updates.json')
        self.iteration_number_file = os.path.join(self.state_dir, 'iteration_number.json')
        self.last_future_version_file = os.path.join(self.state_dir, 'last_future_version_number.json')
        self.latest_loss_file = os.path.join(self.state_dir, 'latest_loss.json')

    def get_state_dir(self):
        return self.state_dir

    async def initialize_directories(self):
        await asyncio.to_thread(os.makedirs, self.state_dir, exist_ok=True)
        await asyncio.to_thread(os.makedirs, self.temp_dir, exist_ok=True)

    async def initialize_all_tensors(self):
        # Delegate to a utility function that uses initialize_tensor
        # and sets up necessary tensors. Adjust as needed.
        await utils_initialize_all_tensors(
            self.state_dir, 
            self.model_adapter,
            memory_logging=False, 
            file_locks=self.file_locks
        )

    async def get_batch(self):
        # The dataset yields token pairs or combined tensors.
        # Return a dict with "input_url" or None if no more data.

        try:
            token_pairs = await self.dataset.__anext__()
            if not token_pairs or len(token_pairs) == 0:
                # If no token pairs, treat as unexpected data scenario.
                logging.error("get_batch: Received empty token_pairs from dataset.")
                raise Exception("No token pairs available from dataset.")

            inputs = [torch.tensor(pair[0], dtype=torch.long) for pair in token_pairs]
            targets = [torch.tensor(pair[1], dtype=torch.long) for pair in token_pairs]

            if len(inputs) == 0 or len(targets) == 0:
                logging.error("get_batch: Empty inputs or targets after processing token pairs.")
                raise Exception("Empty batch generated.")

            batch_tensor = torch.stack(inputs)
            targets_tensor = torch.stack(targets)
            combined_tensor = torch.cat([batch_tensor, targets_tensor], dim=0)

            if combined_tensor.numel() == 0:
                logging.error("get_batch: Combined tensor is empty.")
                raise Exception("Combined tensor is empty.")

            timestamp = int(time.time())
            random_suffix = random.randint(1000, 9999)
            combined_filename = f'input_{timestamp}_{random_suffix}.pt'
            combined_path = os.path.join(self.temp_dir, combined_filename)

            await asyncio.to_thread(torch.save, combined_tensor, combined_path)

            # Double check file size:
            if not os.path.exists(combined_path) or os.path.getsize(combined_path) == 0:
                logging.error(f"get_batch: Saved file {combined_path} is empty or not found.")
                raise Exception(f"Failed to save a valid batch tensor to {combined_path}")

            return {'input_url': f'/data/state/temp/{combined_filename}'}

        except StopAsyncIteration:
            # Dataset is exhausted, no more data available.
            logging.error("get_batch: Dataset exhausted, no more data.")
            return None
        except Exception as e:
            logging.error(f"Error getting batch: {e}", exc_info=True)
        raise


    async def update_state(self, tensor_name, result_url, version_number, input_url, learning_params):
        # All logic that applies gradients and updates state goes here
        # This was previously handled inline, now moved here
        from ..common import download_file
        # Load grads
        local_path = os.path.join(self.state_dir, result_url.lstrip('/data/state/'))
        if not os.path.exists(local_path):
            raise ValueError("Gradient file not found at {local_path}")

        grads_tensor = await asyncio.to_thread(torch.load, local_path, map_location=device)

        # The update logic (e.g., applying optimizer steps) previously done in update_state
        # Move all relevant code here. For brevity, let's just assume we do what was done before:
        # load block_timestamps, num_updates, etc., apply NAG update, etc.

        block_timestamps = await load_json(self.block_timestamps_file, {}, self.file_locks['block_timestamps'])
        num_updates = await load_json(self.num_updates_file, {}, self.file_locks['num_updates'])
        iteration_number = await load_json(self.iteration_number_file, {}, self.file_locks['iteration_number'])
        last_future_version_number = await load_json(self.last_future_version_file, {}, self.file_locks['last_future_version_number'])

        current_version = block_timestamps.get(tensor_name, 0)
        if version_number != current_version:
            raise ValueError(f"Version number mismatch: {version_number} vs {current_version}")

        # Accumulate grads
        future_version_number = version_number + 1
        accumulated_grads_path = os.path.join(self.state_dir, f'accumulated_grads_{tensor_name}_{future_version_number}.pt')
        if os.path.exists(accumulated_grads_path):
            accumulated_grads = await asyncio.to_thread(torch.load, accumulated_grads_path, map_location=device)
        else:
            accumulated_grads = torch.zeros_like(grads_tensor, device=device)

        accumulated_grads += grads_tensor.to(device)
        current_updates = num_updates.get(tensor_name, 0)+1
        num_updates[tensor_name] = current_updates

        await asyncio.to_thread(torch.save, accumulated_grads, accumulated_grads_path)
        await save_json(self.num_updates_file, num_updates, self.file_locks['num_updates'])

        averaged_grads = (accumulated_grads / current_updates).to(device)
        current_iter = iteration_number.get(tensor_name,0)
        iteration_number[tensor_name] = current_iter + 1
        await save_json(self.iteration_number_file, iteration_number, self.file_locks['iteration_number'])

        lr = learning_params['learning_rate']
        beta1 = learning_params['beta1']
        beta2 = learning_params['beta2']
        epsilon = learning_params['epsilon']
        weight_decay = learning_params['weight_decay']

        current_param_path = os.path.join(self.state_dir, f'{tensor_name}_{version_number}.pt')
        if not os.path.exists(current_param_path):
            raise FileNotFoundError(f"Param file not found: {current_param_path}")
        params = await asyncio.to_thread(torch.load, current_param_path, map_location=device)

        adam_m_path = os.path.join(self.state_dir, f'{tensor_name}_adam_m_{version_number}.pt')
        if os.path.exists(adam_m_path):
            m = await asyncio.to_thread(torch.load, adam_m_path, map_location=device)
        else:
            m = torch.zeros_like(params, device=device)

        from ..util.nag import nag_update
        new_params, new_m = await asyncio.to_thread(nag_update, params, averaged_grads, m, lr, weight_decay, beta1, epsilon)

        future_param_path = os.path.join(self.state_dir, f'{tensor_name}_{future_version_number}.pt')
        future_m_path = os.path.join(self.state_dir, f'{tensor_name}_adam_m_{future_version_number}.pt')

        await asyncio.to_thread(torch.save, new_params, future_param_path)
        await asyncio.to_thread(torch.save, new_m, future_m_path)

        block_timestamps[tensor_name] = future_version_number
        if tensor_name not in last_future_version_number or last_future_version_number[tensor_name] < future_version_number:
            last_future_version_number[tensor_name] = future_version_number

        await save_json(self.block_timestamps_file, block_timestamps, self.file_locks['block_timestamps'])
        await save_json(self.last_future_version_file, last_future_version_number, self.file_locks['last_future_version_number'])

        # Clean old accumulated_grads
        for fname in os.listdir(self.state_dir):
            if fname.startswith(f'accumulated_grads_{tensor_name}_') and not fname.endswith(f'{future_version_number}.pt'):
                old_acc_path = os.path.join(self.state_dir, fname)
                if os.path.exists(old_acc_path):
                    await asyncio.to_thread(os.remove, old_acc_path)

        if os.path.exists(local_path):
            await asyncio.to_thread(os.remove, local_path)

    async def update_loss(self, loss_value, version_number):
        latest_loss = await load_json(self.latest_loss_file, {'value': None}, self.file_locks['latest_loss'])
        latest_loss['value'] = loss_value
        await save_json(self.latest_loss_file, latest_loss, self.file_locks['latest_loss'])

    async def get_loss(self):
        latest_loss = await load_json(self.latest_loss_file, {'value': None}, self.file_locks['latest_loss'])
        return {'loss': latest_loss.get('value')}

    async def upload_tensor(self, tensor_state, label):
        timestamp = int(time.time())
        random_suffix = random.randint(1000, 9999)
        filename = f'{label}_{timestamp}_{random_suffix}.pt'
        local_file_path = os.path.join(self.temp_dir, filename)
        await asyncio.to_thread(torch.save, tensor_state, local_file_path)
        return f'/data/state/temp/{filename}'

    async def get_data_file(self, filename):
        full_path = os.path.join(self.temp_dir, filename)
        if not os.path.abspath(full_path).startswith(os.path.abspath(self.temp_dir)):
            logging.error(f"get_data_file: Access denied for {full_path}.")
            return {'error': 'Access denied'}

        if not os.path.exists(full_path):
            logging.error(f"get_data_file: File not found {full_path}.")
            return {'error': 'File not found'}

        async with aiofiles.open(full_path, 'rb') as f:
            data = await f.read()

        if len(data) == 0:
            logging.error(f"get_data_file: File {full_path} is empty.")
            return {'error': 'File is empty'}

        return {'data': data, 'mime_type': 'application/octet-stream'}

    async def stream_data_file(self, filename):
        full_path = os.path.join(self.temp_dir, filename)
        if not os.path.exists(full_path):
            return {'error': 'file not found'}

        file_size = os.path.getsize(full_path)
        if file_size == 0:
            return {'error': 'empty_file'}

        # Read the entire file into memory
        async with aiofiles.open(full_path, 'rb') as f:
            data = await f.read()

        # Return raw bytes directly
        return data


    async def get_latest_state(self, tensor_name):
        # The latest version number is stored in block_timestamps
        block_timestamps = await load_json(self.block_timestamps_file, {}, self.file_locks['block_timestamps'])
        latest_version_number = block_timestamps.get(tensor_name, 0)
        state_file_path = os.path.join(self.state_dir, f'{tensor_name}_{latest_version_number}.pt')
        if not os.path.exists(state_file_path):
            raise FileNotFoundError("Tensor not found.")

        # Load into bytes
        tensor_data = io.BytesIO()
        param = await asyncio.to_thread(torch.load, state_file_path, map_location='cpu')
        torch.save(param, tensor_data)
        tensor_bytes = tensor_data.getvalue()

        return {'data': tensor_bytes, 'version_number': latest_version_number}
