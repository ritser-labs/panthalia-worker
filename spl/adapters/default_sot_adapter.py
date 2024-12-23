# spl/adapters/default_sot_adapter.py

import os
import asyncio
import logging
import random
import time
import json
import torch
from quart import Quart, request, jsonify, make_response, send_from_directory
from ..auth.api_auth import requires_authentication
# from ..util.json import load_json, save_json  # not used anymore
from ..common import (
    get_future_version_number, TENSOR_NAME
)
from ..device import device
from ..util.sot import (
    version_number_exists, get_local_file_path, apply_optimizer,
    update_block_timestamps, cleanup_old_timestamp,
    update_cleanup_timestamps, initialize_all_tensors
)
from ..db.db_adapter_client import DBAdapterClient
from ..util.docker import janky_url_replace
from .sot_adapter import BaseSOTAdapter

class DefaultSOTAdapter(BaseSOTAdapter):
    def __init__(self, model_adapter, dataset, state_dir, tensor_version_interval, hyperparams_getter=None):
        self.initialized = False
        self.db_adapter = None
        self.sot_id = None
        self.db_url = None
        self.private_key = None
        self.job_id = None
        self.perm_db = None

        self.model_adapter = model_adapter
        self.tensor_version_interval = tensor_version_interval
        self.dataset = dataset
        self.base_dir = state_dir
        self.temp_dir = os.path.join(self.base_dir, "temp")
        self.hyperparams_getter = hyperparams_getter

        self.file_locks = None
        self.update_timestamp_lock = None
        self.synced_workers = 0
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        self.app = Quart(__name__)
        self.app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024 * 1024  # 1 TB

    async def initialize(self, sot_id, db_url, private_key, job_id, perm_db, port):
        self.sot_id = sot_id
        self.db_url = janky_url_replace(db_url)
        self.private_key = private_key
        self.job_id = job_id
        self.perm_db = perm_db
        self.db_adapter = DBAdapterClient(self.db_url, self.private_key)

        # We won't use local JSON locks. We'll store everything in DB state_json
        self.file_locks = {}
        self.update_timestamp_lock = asyncio.Lock()

        await initialize_all_tensors(
            self.base_dir,
            self.tensor_version_interval,
            self.model_adapter.init_tensor,
            memory_logging=False,
            file_locks=self.file_locks
        )

        await self.dataset.initialize_dataset()
        self.initialized = True

        db_adapter_lambda = lambda: self.db_adapter
        perm_db_lambda = lambda: self.perm_db
        requires_auth = requires_authentication(db_adapter_lambda, perm_db_lambda)

        @self.app.route('/health', methods=['GET'])
        async def health_check():
            return jsonify({'status': 'healthy'}), 200

        @self.app.route('/report_sync', methods=['POST'])
        async def report_sync():
            self.synced_workers += 1
            return jsonify({'status': 'ok'})

        @self.app.route('/get_num_synced', methods=['GET'])
        async def get_num_synced():
            return jsonify(self.synced_workers)

        @self.app.route('/get_batch', methods=['POST'])
        @requires_auth
        async def get_batch():
            logging.info("Accessing /get_batch endpoint")
            try:
                token_pairs = await self.dataset.__anext__()
                if not token_pairs:
                    logging.info("No more batches available in /get_batch.")
                    return jsonify({'error': 'No more batches available'}), 404
            except StopAsyncIteration:
                logging.info("Dataset iterator exhausted (StopAsyncIteration).")
                return jsonify({'error': 'No more batches available'}), 404
            except Exception as e:
                logging.error(f"Error in /get_batch: {e}", exc_info=True)
                return jsonify({'error': 'Could not get batch'}), 500

            inputs = [torch.tensor(pair[0], dtype=torch.long) for pair in token_pairs]
            targets = [torch.tensor(pair[1], dtype=torch.long) for pair in token_pairs]

            batch_tensor = torch.stack(inputs)
            targets_tensor = torch.stack(targets)

            timestamp = int(time.time())
            random_suffix = random.randint(1000, 9999)
            combined_filename = f'input_{timestamp}_{random_suffix}.pt'
            combined_tensor = torch.cat([batch_tensor, targets_tensor], dim=0)
            await asyncio.to_thread(torch.save, combined_tensor, os.path.join(self.temp_dir, combined_filename))

            return jsonify({
                'input_url': f'/data/state/temp/{combined_filename}'
            })

        @self.app.route('/update_state', methods=['POST'])
        @requires_auth
        async def update_state():
            logging.info("Accessing /update_state endpoint")
            data = await request.get_json()
            tensor_name = data.get('tensor_name')
            result_url = data.get('result_url')
            # We used to parse `batch_url` and `targets_url`, but not anymore!

            if not tensor_name or not result_url:
                logging.error("Missing tensor_name or result_url in /update_state")
                return jsonify({'error': 'Missing tensor_name or result_url'}), 400

            # Load entire state dict from DB (unchanged):
            state_data = await self.db_adapter.get_state_for_job(self.job_id)
            block_timestamps = state_data.get("block_timestamps", {})
            num_updates = state_data.get("num_updates", {})
            last_future_version_number = state_data.get("last_future_version_number", {})
            iteration_number = state_data.get("iteration_number", {})

            future_version_number = get_future_version_number(self.tensor_version_interval)
            current_version_number = block_timestamps.get(tensor_name, 0)

            if data['version_number'] != current_version_number:
                return jsonify({'error': 'Version number mismatch'}), 409

            # Just handle the single `result_url`
            local_file_path = get_local_file_path(data.get('result_url'), request, self.base_dir)
            input_url = data.get('input_url')
            local_input_path = get_local_file_path(input_url, request, self.base_dir)
            try:
                if local_input_path and os.path.exists(local_input_path):
                    os.remove(local_input_path)
            except Exception as e:
                logging.error(f"Error removing input file: {e}", exc_info=True)

            if not local_file_path or not os.path.exists(local_file_path):
                logging.error(f"File not found at {local_file_path}")
                return jsonify({'error': 'File not found'}), 404

            tensor = torch.load(local_file_path, map_location=device)

            # Accumulate grads
            accumulated_grads_path = os.path.join(self.base_dir, f'accumulated_grads_{tensor_name}_{future_version_number}.pt')
            if os.path.exists(accumulated_grads_path):
                accumulated_grads = torch.load(accumulated_grads_path, map_location=device).to(device)
            else:
                accumulated_grads = torch.zeros_like(tensor, device=device)

            accumulated_grads += tensor.to(device)
            await asyncio.to_thread(torch.save, accumulated_grads, accumulated_grads_path)

            # update num_updates
            num_of_updates = num_updates.get(tensor_name, 0) + 1
            num_updates[tensor_name] = num_of_updates

            # do averaging
            averaged_grads = (accumulated_grads / num_of_updates).to(device)

            # get learning params from hyperparams_getter
            learning_params = {}
            if self.hyperparams_getter:
                # pass iteration_number[tensor_name] or something
                iteration_val = iteration_number.get(tensor_name, 0)
                learning_params = self.hyperparams_getter(iteration_val)

            future_tensor, m_update = await apply_optimizer(
                current_version_number,
                tensor_name,
                averaged_grads,
                learning_params.get('learning_rate', 0.001),
                learning_params.get('beta1', 0.9),
                learning_params.get('beta2', 0.999),
                learning_params.get('epsilon', 1e-8),
                learning_params.get('weight_decay', 0.0),
                learning_params.get('t', 0),
                self.base_dir
            )

            future_tensor_path = os.path.join(self.base_dir, f'{tensor_name}_{future_version_number}.pt')
            future_tensor_adam_m_path = os.path.join(self.base_dir, f'{tensor_name}_adam_m_{future_version_number}.pt')
            future_tensor_temp_path = future_tensor_path + '.tmp'
            future_tensor_adam_m_temp_path = future_tensor_adam_m_path + '.tmp'

            await asyncio.to_thread(torch.save, future_tensor, future_tensor_temp_path)
            await asyncio.to_thread(torch.save, m_update, future_tensor_adam_m_temp_path)

            os.rename(future_tensor_temp_path, future_tensor_path)
            os.rename(future_tensor_adam_m_temp_path, future_tensor_adam_m_path)

            old_block_timestamp = block_timestamps.get(tensor_name, 0)
            await cleanup_old_timestamp(tensor_name, old_block_timestamp, block_timestamps, self.base_dir)

            for filename in os.listdir(self.base_dir):
                if filename.startswith(f'accumulated_grads_{tensor_name}_') and not filename.endswith(f'{future_version_number}.pt'):
                    await asyncio.to_thread(os.remove, os.path.join(self.base_dir, filename))

            if os.path.exists(local_file_path):
                await asyncio.to_thread(os.remove, local_file_path)

            block_timestamps[tensor_name] = future_version_number

            # Save everything back in the state
            state_data["block_timestamps"] = block_timestamps
            state_data["num_updates"] = num_updates
            state_data["last_future_version_number"] = last_future_version_number
            state_data["iteration_number"] = iteration_number

            await self.db_adapter.update_state_for_job(self.job_id, state_data)
            return jsonify({'status': 'success', 'version_number': future_version_number})

        @self.app.route('/latest_state', methods=['GET'])
        async def latest_state():
            logging.info("Accessing /latest_state endpoint")
            tensor_name = request.args.get('tensor_name')
            if not tensor_name:
                return jsonify({'error': 'Missing tensor_name parameter'}), 400

            # load from DB
            state_data = await self.db_adapter.get_state_for_job(self.job_id)
            block_timestamps = state_data.get("block_timestamps", {})

            latest_version_number = request.args.get('version_number')
            if latest_version_number is None:
                latest_version_number = block_timestamps.get(tensor_name, 0)
            else:
                if not version_number_exists(int(latest_version_number), tensor_name, self.base_dir):
                    latest_version_number = block_timestamps.get(tensor_name, 0)

            state_file_path = os.path.join(self.base_dir, f'{tensor_name}_{latest_version_number}.pt')
            if not os.path.exists(state_file_path):
                return jsonify({'error': 'Tensor not found'}), 404

            response = await make_response(await send_from_directory(
                directory=self.base_dir,
                file_name=f'{tensor_name}_{latest_version_number}.pt',
                mimetype='application/octet-stream',
                as_attachment=True
            ))
            response.headers['X-Version-Number'] = str(latest_version_number)
            return response

        @self.app.route('/current_timestamp', methods=['POST'])
        async def current_timestamp():
            logging.info("Accessing /current_timestamp endpoint")
            tensor_name = request.args.get('tensor_name')
            if not tensor_name:
                return jsonify({'error': 'Missing tensor_name parameter'}), 400

            state_data = await self.db_adapter.get_state_for_job(self.job_id)
            block_timestamps = state_data.get("block_timestamps", {})
            num_updates = state_data.get("num_updates", {})
            iteration_number = state_data.get("iteration_number", {})
            last_future_version_number = state_data.get("last_future_version_number", {})

            await update_cleanup_timestamps(
                tensor_name, block_timestamps, num_updates, iteration_number,
                last_future_version_number, self.base_dir, self.db_adapter,
                self.job_id, self.tensor_version_interval, self.update_timestamp_lock, self.file_locks
            )

            latest_version_number = block_timestamps.get(tensor_name, 0)
            return jsonify({'version_number': latest_version_number})

        @self.app.route('/tensor_size', methods=['GET'])
        async def get_tensor_size():
            logging.info("Accessing /tensor_size endpoint")
            tensor_name = request.args.get('tensor_name')
            if not tensor_name:
                return jsonify({'error': 'Missing tensor_name parameter'}), 400

            state_file_path = os.path.join(self.base_dir, f'{tensor_name}.pt')
            if not os.path.exists(state_file_path):
                return jsonify({'error': 'Tensor not found'}), 404

            tensor = torch.load(state_file_path, map_location=device)
            size = tensor.numel()
            return jsonify({'size': size})

        @self.app.route('/data/state/temp/<path:filename>', methods=['GET'])
        async def get_data_file(filename):
            file_path = os.path.join(self.temp_dir, filename)
            if not os.path.abspath(file_path).startswith(os.path.abspath(self.temp_dir)):
                return jsonify({'error': 'File not found or access denied'}), 403
            if not os.path.exists(file_path):
                return jsonify({'error': 'File not found'}), 404
            response = await make_response(await send_from_directory(
                directory=self.temp_dir,
                file_name=filename,
                mimetype='application/octet-stream',
                as_attachment=True
            ))
            return response

        @self.app.route('/upload_tensor', methods=['POST'])
        async def upload_tensor():
            request_files = await request.files
            if 'tensor' not in request_files:
                return jsonify({'error': 'No tensor file provided'}), 400

            request_form = await request.form
            if 'label' not in request_form:
                return jsonify({'error': 'No label provided'}), 400

            tensor_file = request_files['tensor']
            label = request_form['label']
            update_version_number = int(time.time())
            random_suffix = random.randint(1000, 9999)
            filename = f'{label}_{update_version_number}_{random_suffix}.pt'
            local_file_path = os.path.join(self.temp_dir, filename)

            chunk_size = 1024 * 1024
            import aiofiles
            async with aiofiles.open(local_file_path, 'wb') as f:
                while True:
                    chunk = tensor_file.read(chunk_size)
                    if not chunk:
                        break
                    await f.write(chunk)

            tensor_state = torch.load(local_file_path, map_location=device)
            torch.save(tensor_state, os.path.join(self.temp_dir, filename))

            return jsonify({'message': 'Tensor uploaded successfully', 'tensor_url': f'/data/state/temp/{filename}'}), 200

        @self.app.route('/update_loss', methods=['POST'])
        async def update_loss():
            data = await request.get_json()
            if not data or 'loss' not in data:
                return jsonify({'error': 'Missing loss value'}), 400

            state_data = await self.db_adapter.get_state_for_job(self.job_id)
            if "latest_loss" not in state_data:
                state_data["latest_loss"] = {}
            state_data["latest_loss"]["value"] = data['loss']
            await self.db_adapter.update_state_for_job(self.job_id, state_data)
            logging.info(f"Updated latest loss for version {data['version_number']}: {data['loss']}")

            return jsonify({'status': 'success'}), 200

        @self.app.route('/get_loss', methods=['GET'])
        async def get_loss():
            logging.info("Accessing /get_loss endpoint")
            state_data = await self.db_adapter.get_state_for_job(self.job_id)
            loss = None
            if "latest_loss" in state_data:
                loss = state_data["latest_loss"].get("value", None)
            return jsonify({'loss': loss}), 200

        logging.info("default_sot_adapter initialized fully")

        await self.app.run_task(host='0.0.0.0', port=port)
