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
from ..util.json import load_json, save_json
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

        self.file_locks = {
            'block_timestamps': asyncio.Lock(),
            'num_updates': asyncio.Lock(),
            'iteration_number': asyncio.Lock(),
            'last_future_version_number': asyncio.Lock(),
            'latest_loss': asyncio.Lock()
        }
        self.update_timestamp_lock = asyncio.Lock()

        # Initialize all tensors
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

            if not tensor_name or not result_url:
                logging.error("Missing tensor_name or result_url in /update_state")
                return jsonify({'error': 'Missing tensor_name or result_url'}), 400

            state_dir = self.base_dir
            db_adapter = self.db_adapter
            job_id = self.job_id

            block_timestamps_file = os.path.join(state_dir, 'block_timestamps.json')
            num_updates_file = os.path.join(state_dir, 'num_updates.json')
            last_future_version_file = os.path.join(state_dir, 'last_future_version_number.json')
            iteration_number_file = os.path.join(state_dir, 'iteration_number.json')

            block_timestamps = await load_json(block_timestamps_file, {}, self.file_locks['block_timestamps'])
            num_updates = await load_json(num_updates_file, {}, self.file_locks['num_updates'])
            last_future_version_number = await load_json(last_future_version_file, {}, self.file_locks['last_future_version_number'])
            iteration_number = await load_json(iteration_number_file, {}, self.file_locks['iteration_number'])

            future_version_number = get_future_version_number(self.tensor_version_interval)

            if data['version_number'] != block_timestamps.get(tensor_name, 0):
                return jsonify({'error': 'Version number mismatch'}), 409

            old_block_timestamp = await update_block_timestamps(
                tensor_name, block_timestamps, num_updates, iteration_number,
                last_future_version_number, state_dir, db_adapter, job_id, self.tensor_version_interval,
                self.update_timestamp_lock, self.file_locks
            )

            local_file_path = get_local_file_path(data.get('result_url'), request, state_dir)
            batch_url = data.get('batch_url')
            targets_url = data.get('targets_url')
            local_batch_file_path = get_local_file_path(batch_url, request, state_dir)
            local_targets_file_path = get_local_file_path(targets_url, request, state_dir)
            try:
                if local_batch_file_path and os.path.exists(local_batch_file_path):
                    await asyncio.to_thread(os.remove, local_batch_file_path)
                if local_targets_file_path and os.path.exists(local_targets_file_path):
                    await asyncio.to_thread(os.remove, local_targets_file_path)
            except:
                pass

            if not local_file_path or not os.path.exists(local_file_path):
                logging.error(f"File not found at {local_file_path}")
                return jsonify({'error': 'File not found'}), 404

            tensor = torch.load(local_file_path, map_location=device)
            accumulated_grads_path = os.path.join(state_dir, f'accumulated_grads_{tensor_name}_{future_version_number}.pt')
            if os.path.exists(accumulated_grads_path):
                accumulated_grads = torch.load(accumulated_grads_path, map_location=device).to(device)
            else:
                accumulated_grads = torch.zeros_like(tensor, device=device)

            accumulated_grads += tensor.to(device)
            await asyncio.to_thread(torch.save, accumulated_grads, accumulated_grads_path)

            current_version_number = block_timestamps.get(tensor_name, 0)
            num_of_updates = num_updates.get(tensor_name, 0) + 1
            num_updates[tensor_name] = num_of_updates
            await save_json(num_updates_file, num_updates, self.file_locks['num_updates'])

            averaged_grads = (accumulated_grads / num_of_updates).to(device)
            learning_params = await self.hyperparams_getter(iteration_number[tensor_name])
            future_tensor, m_update = await apply_optimizer(
                current_version_number,
                tensor_name,
                averaged_grads,
                learning_params['learning_rate'],
                learning_params['beta1'],
                learning_params['beta2'],
                learning_params['epsilon'],
                learning_params['weight_decay'],
                learning_params['t'],
                state_dir
            )

            future_tensor_path = os.path.join(state_dir, f'{tensor_name}_{future_version_number}.pt')
            future_tensor_adam_m_path = os.path.join(state_dir, f'{tensor_name}_adam_m_{future_version_number}.pt')
            future_tensor_temp_path = future_tensor_path + '.tmp'
            future_tensor_adam_m_temp_path = future_tensor_adam_m_path + '.tmp'

            await asyncio.to_thread(torch.save, future_tensor, future_tensor_temp_path)
            await asyncio.to_thread(torch.save, m_update, future_tensor_adam_m_temp_path)

            os.rename(future_tensor_temp_path, future_tensor_path)
            os.rename(future_tensor_adam_m_temp_path, future_tensor_adam_m_path)

            await cleanup_old_timestamp(tensor_name, old_block_timestamp, block_timestamps, state_dir)

            for filename in os.listdir(state_dir):
                if filename.startswith(f'accumulated_grads_{tensor_name}_') and not filename.endswith(f'{future_version_number}.pt'):
                    await asyncio.to_thread(os.remove, os.path.join(state_dir, filename))

            if os.path.exists(local_file_path):
                await asyncio.to_thread(os.remove, local_file_path)

            return jsonify({'status': 'success', 'version_number': future_version_number})

        @self.app.route('/latest_state', methods=['GET'])
        async def latest_state():
            logging.info("Accessing /latest_state endpoint")
            tensor_name = request.args.get('tensor_name')
            if not tensor_name:
                return jsonify({'error': 'Missing tensor_name parameter'}), 400

            state_dir = self.base_dir
            block_timestamps_file = os.path.join(state_dir, 'block_timestamps.json')
            block_timestamps = await load_json(block_timestamps_file, {}, self.file_locks['block_timestamps'])

            latest_version_number = request.args.get('version_number')
            if latest_version_number is None or not version_number_exists(latest_version_number, tensor_name, state_dir):
                latest_version_number = block_timestamps.get(tensor_name, 0)
            else:
                latest_version_number = int(latest_version_number)

            state_file_path = os.path.join(state_dir, f'{tensor_name}_{latest_version_number}.pt')
            if not os.path.exists(state_file_path):
                return jsonify({'error': 'Tensor not found'}), 404

            response = await make_response(await send_from_directory(
                directory=state_dir,
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

            state_dir = self.base_dir
            db_adapter = self.db_adapter
            job_id = self.job_id

            block_timestamps_file = os.path.join(state_dir, 'block_timestamps.json')
            block_timestamps = await load_json(block_timestamps_file, {}, self.file_locks['block_timestamps'])
            num_updates_file = os.path.join(state_dir, 'num_updates.json')
            num_updates = await load_json(num_updates_file, {}, self.file_locks['num_updates'])
            iteration_number_file = os.path.join(state_dir, 'iteration_number.json')
            iteration_number = await load_json(iteration_number_file, {}, self.file_locks['iteration_number'])
            last_future_version_file = os.path.join(state_dir, 'last_future_version_number.json')
            last_future_version_number = await load_json(last_future_version_file, {}, self.file_locks['last_future_version_number'])

            await update_cleanup_timestamps(
                tensor_name, block_timestamps, num_updates, iteration_number,
                last_future_version_number, state_dir, db_adapter, job_id, self.tensor_version_interval,
                self.update_timestamp_lock, self.file_locks
            )

            latest_version_number = block_timestamps.get(tensor_name, 0)
            return jsonify({'version_number': latest_version_number})

        @self.app.route('/tensor_size', methods=['GET'])
        async def get_tensor_size():
            logging.info("Accessing /tensor_size endpoint")
            tensor_name = request.args.get('tensor_name')
            if not tensor_name:
                return jsonify({'error': 'Missing tensor_name parameter'}), 400

            state_dir = self.base_dir
            state_file_path = os.path.join(state_dir, f'{tensor_name}.pt')
            if not os.path.exists(state_file_path):
                return jsonify({'error': 'Tensor not found'}), 404

            tensor = torch.load(state_file_path, map_location=device)
            size = tensor.numel()
            return jsonify({'size': size})

        @self.app.route('/data/state/temp/<path:filename>', methods=['GET'])
        async def get_data_file(filename):
            temp_dir = self.temp_dir
            file_path = os.path.join(temp_dir, filename)
            if not os.path.abspath(file_path).startswith(os.path.abspath(temp_dir)):
                return jsonify({'error': 'File not found or access denied'}), 403

            if not os.path.exists(file_path):
                return jsonify({'error': 'File not found'}), 404

            response = await make_response(await send_from_directory(
                directory=temp_dir,
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

            temp_dir = self.temp_dir
            tensor_file = request_files['tensor']
            label = request_form['label']
            update_version_number = int(time.time())
            random_suffix = random.randint(1000, 9999)
            filename = f'{label}_{update_version_number}_{random_suffix}.pt'
            local_file_path = os.path.join(temp_dir, filename)

            chunk_size = 1024 * 1024
            import aiofiles
            async with aiofiles.open(local_file_path, 'wb') as f:
                while True:
                    chunk = tensor_file.read(chunk_size)
                    if not chunk:
                        break
                    await f.write(chunk)

            tensor_state = torch.load(local_file_path, map_location=device)
            torch.save(tensor_state, os.path.join(temp_dir, filename))

            return jsonify({'message': 'Tensor uploaded successfully', 'tensor_url': f'/data/state/temp/{filename}'}), 200

        @self.app.route('/update_loss', methods=['POST'])
        async def update_loss():
            data = await request.get_json()
            if not data or 'loss' not in data:
                return jsonify({'error': 'Missing loss value'}), 400

            state_dir = self.base_dir
            latest_loss_path = os.path.join(state_dir, 'latest_loss.json')

            latest_loss = await load_json(latest_loss_path, {'value': None}, self.file_locks['latest_loss'])
            latest_loss['value'] = data['loss']
            await save_json(latest_loss_path, latest_loss, self.file_locks['latest_loss'])
            logging.info(f"Updated latest loss for version {data['version_number']}: {data['loss']}")

            return jsonify({'status': 'success'}), 200

        @self.app.route('/get_loss', methods=['GET'])
        async def get_loss():
            logging.info("Accessing /get_loss endpoint")
            state_dir = self.base_dir
            latest_loss_path = os.path.join(state_dir, 'latest_loss.json')
            latest_loss = await load_json(latest_loss_path, {'value': None}, self.file_locks['latest_loss'])
            loss = latest_loss.get('value')
            return jsonify({'loss': loss}), 200

        logging.info("default_sot_adapter initialized fully")

        # Now start the server:
        # Using hypercorn or any async server would be ideal. Let's do uvicorn style:
        # But since we are inside async function, we can use self.app.run_task
        # (Quart >=0.13 supports run_task)
        await self.app.run_task(host='0.0.0.0', port=port)
