# spl/adapters/default_sot_adapter.py

import os
import asyncio
import logging
import random
import time
import json
import torch
from quart import Quart, request, jsonify, make_response, send_from_directory

from ..common import (
    get_future_version_number,
    TENSOR_NAME,
    download_file
)
from ..device import device
from ..util.sot import (
    version_number_exists,
    get_local_file_path,
    apply_optimizer,
    update_block_timestamps,
    cleanup_old_timestamp,
    update_cleanup_timestamps,
    initialize_all_tensors
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
        """
        Called once by the SOT side, passing relevant config. We optionally load an initial
        model from job.initial_state_url if it exists. Then we initialize the dataset and run
        the SOT adapter's HTTP server.
        """
        self.sot_id = sot_id
        self.db_url = janky_url_replace(db_url)
        self.private_key = private_key
        self.job_id = job_id
        self.perm_db = perm_db
        self.db_adapter = DBAdapterClient(self.db_url, self.private_key)

        self.file_locks = {}
        self.update_timestamp_lock = asyncio.Lock()

        # -----------------------------------------------------------------
        # Optionally load initial model from job.initial_state_url
        # -----------------------------------------------------------------
        job_obj = await self.db_adapter.sot_get_job(self.job_id)
        if job_obj and job_obj.initial_state_url and job_obj.initial_state_url.strip():
            init_url = job_obj.initial_state_url.strip()
            logging.info(f"[DefaultSOTAdapter.initialize] Found non-empty initial_state_url => {init_url}")
            local_path = os.path.join(self.base_dir, "model_0.pt")
            if not os.path.exists(local_path):
                logging.info(f"Downloading initial model from {init_url} -> {local_path}")
                try:
                    await download_file(init_url, local_file_path=local_path, download_type='tensor', chunk_timeout=10)
                    logging.info("Initial model downloaded successfully.")
                except Exception as e:
                    logging.error(f"Failed to download initial_state_url: {e}", exc_info=True)
            else:
                logging.info("Model 0 already exists locally, skipping re-download.")
        else:
            logging.info("[DefaultSOTAdapter.initialize] No initial_state_url found; using default init flow.")

        # -----------------------------------------------------------------
        # Ensure at least model_0.pt is present
        # -----------------------------------------------------------------
        await initialize_all_tensors(
            self.base_dir,
            self.tensor_version_interval,
            self.model_adapter.init_tensor,
            memory_logging=False,
            file_locks=self.file_locks
        )

        # Start dataset
        await self.dataset.initialize_dataset()
        self.initialized = True

        db_adapter_lambda = lambda: self.db_adapter
        perm_db_lambda = lambda: self.perm_db

        from ..auth.key_auth import requires_key_auth
        requires_auth = requires_key_auth(db_adapter_lambda, perm_db_lambda)

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
                logging.info("Dataset exhausted (StopAsyncIteration).")
                return jsonify({'error': 'No more batches available'}), 404
            except Exception as e:
                logging.error(f"Error in /get_batch: {e}", exc_info=True)
                return jsonify({'error': 'Could not get batch'}), 500

            import torch
            inputs = [torch.tensor(pair[0], dtype=torch.long) for pair in token_pairs]
            targets = [torch.tensor(pair[1], dtype=torch.long) for pair in token_pairs]
            batch_tensor = torch.stack(inputs)
            targets_tensor = torch.stack(targets)

            timestamp = int(time.time())
            random_suffix = random.randint(1000, 9999)
            combined_filename = f'input_{timestamp}_{random_suffix}.pt'
            combined_tensor = torch.cat([batch_tensor, targets_tensor], dim=0)

            await asyncio.to_thread(
                torch.save, 
                combined_tensor, 
                os.path.join(self.temp_dir, combined_filename)
            )

            return jsonify({
                'input_url': f'/data/state/temp/{combined_filename}'
            })

        @self.app.route('/update_state', methods=['POST'])
        @requires_auth
        async def update_state():
            logging.info("Accessing /update_state endpoint")
            data = await request.get_json()

            result_url = data.get('result_url')
            if not result_url:
                return jsonify({'error': 'Missing result_url'}), 400

            version_number = data.get('version_number')
            if version_number is None:
                return jsonify({'error': 'Missing version_number'}), 400

            tensor_name = TENSOR_NAME
            state_data = await self.db_adapter.get_sot_state_for_job(self.job_id)
            block_timestamps = state_data.get("block_timestamps", {})
            num_updates = state_data.get("num_updates", {})
            last_future_version_number = state_data.get("last_future_version_number", {})
            iteration_number = state_data.get("iteration_number", {})

            current_version_number = block_timestamps.get(tensor_name, 0)
            if version_number != current_version_number:
                return jsonify({'error': 'Version number mismatch'}), 409

            local_file_path = get_local_file_path(result_url, request, self.base_dir)
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

            # ===========================
            # LOAD + DEQUANTIZE
            # ===========================
            loaded_obj = torch.load(local_file_path, map_location=device)

            # If it's our int8 dict => dequantize
            if isinstance(loaded_obj, dict) and 'quantized' in loaded_obj:
                qvals = loaded_obj['quantized'].float()  # from uint8 -> float
                scale = loaded_obj['scale']
                min_val = loaded_obj['min_val']
                orig_shape = loaded_obj['orig_shape']

                # v = qvals*scale + min_val
                # then reshape
                tensor = (qvals * scale + min_val).view(orig_shape).to(device)
            else:
                # Possibly float16 or float32 or whatever
                tensor = loaded_obj
                if tensor.dtype == torch.float16:
                    tensor = tensor.float()
                tensor = tensor.to(device)

            # We can safely remove the uploaded file now
            if os.path.exists(local_file_path):
                await asyncio.to_thread(os.remove, local_file_path)

            future_version_number = get_future_version_number(self.tensor_version_interval)
            grads_path = os.path.join(
                self.base_dir, f'accumulated_grads_{tensor_name}_{future_version_number}.pt'
            )
            if os.path.exists(grads_path):
                accumulated_grads = torch.load(grads_path, map_location=device).to(device)
            else:
                accumulated_grads = torch.zeros_like(tensor, device=device)

            accumulated_grads += tensor
            await asyncio.to_thread(torch.save, accumulated_grads, grads_path)

            num_of_updates = num_updates.get(tensor_name, 0) + 1
            num_updates[tensor_name] = num_of_updates

            # Possibly get hyperparams
            if self.hyperparams_getter:
                it_val = iteration_number.get(tensor_name, 0)
                learning_params = self.hyperparams_getter(it_val)
            else:
                learning_params = {}

            lr = learning_params.get('learning_rate', 0.001)
            beta1 = learning_params.get('beta1', 0.9)
            beta2 = learning_params.get('beta2', 0.999)
            eps = learning_params.get('epsilon', 1e-8)
            wd = learning_params.get('weight_decay', 0.0)
            t_val = learning_params.get('t', 0)

            # average grads
            averaged_grads = accumulated_grads / num_of_updates

            new_params, new_m = await apply_optimizer(
                current_version_number,
                tensor_name,
                averaged_grads,
                lr,
                beta1,
                beta2,
                eps,
                wd,
                t_val,
                self.base_dir
            )

            # finalize
            future_tensor_path = os.path.join(self.base_dir, f'{tensor_name}_{future_version_number}.pt')
            future_tensor_adam_m_path = os.path.join(
                self.base_dir,
                f'{tensor_name}_adam_m_{future_version_number}.pt'
            )
            await asyncio.to_thread(torch.save, new_params, future_tensor_path + '.tmp')
            await asyncio.to_thread(torch.save, new_m, future_tensor_adam_m_path + '.tmp')

            os.rename(future_tensor_path + '.tmp', future_tensor_path)
            os.rename(future_tensor_adam_m_path + '.tmp', future_tensor_adam_m_path)

            # Cleanup old version
            old_block_timestamp = block_timestamps.get(tensor_name, 0)
            await cleanup_old_timestamp(
                tensor_name, old_block_timestamp, block_timestamps, self.base_dir
            )

            # Remove partial grads from earlier intervals
            for filename in os.listdir(self.base_dir):
                if filename.startswith(f'accumulated_grads_{tensor_name}_'):
                    if not filename.endswith(f'{future_version_number}.pt'):
                        await asyncio.to_thread(os.remove, os.path.join(self.base_dir, filename))

            # Update DB state
            block_timestamps[tensor_name] = future_version_number
            state_data["block_timestamps"] = block_timestamps
            state_data["num_updates"] = num_updates
            state_data["last_future_version_number"] = last_future_version_number
            state_data["iteration_number"] = iteration_number
            await self.db_adapter.update_sot_state_for_job(self.job_id, state_data)

            return jsonify({
                'status': 'success',
                'version_number': future_version_number
            })

        @self.app.route('/latest_state', methods=['GET'])
        async def latest_state():
            logging.info("Accessing /latest_state endpoint")
            tensor_name = TENSOR_NAME

            state_data = await self.db_adapter.get_sot_state_for_job(self.job_id)
            block_timestamps = state_data.get("block_timestamps", {})

            requested_version_str = request.args.get('version_number', None)
            if requested_version_str is None:
                requested_version_num = block_timestamps.get(tensor_name, 0)
            else:
                requested_version_num = int(requested_version_str)
                if not version_number_exists(requested_version_num, tensor_name, self.base_dir):
                    requested_version_num = block_timestamps.get(tensor_name, 0)

            path = os.path.join(self.base_dir, f'{tensor_name}_{requested_version_num}.pt')
            if not os.path.exists(path):
                return jsonify({'error': 'Tensor not found'}), 404

            response = await make_response(
                await send_from_directory(
                    directory=self.base_dir,
                    file_name=f'{tensor_name}_{requested_version_num}.pt',
                    mimetype='application/octet-stream',
                    as_attachment=True
                )
            )
            response.headers['X-Version-Number'] = str(requested_version_num)
            return response

        @self.app.route('/current_timestamp', methods=['POST'])
        async def current_timestamp():
            logging.info("Accessing /current_timestamp endpoint")
            tensor_name = TENSOR_NAME

            state_data = await self.db_adapter.get_sot_state_for_job(self.job_id)
            block_timestamps = state_data.get("block_timestamps", {})
            num_updates = state_data.get("num_updates", {})
            iteration_number = state_data.get("iteration_number", {})
            last_future_version_number = state_data.get("last_future_version_number", {})

            try:
                await update_cleanup_timestamps(
                    tensor_name,
                    block_timestamps,
                    num_updates,
                    iteration_number,
                    last_future_version_number,
                    self.base_dir,
                    self.db_adapter,
                    self.job_id,
                    self.tensor_version_interval,
                    self.update_timestamp_lock,
                    self.file_locks
                )
            except Exception as e:
                logging.error(f"update_cleanup_timestamps crashed: {e}", exc_info=True)

            final_state = await self.db_adapter.get_sot_state_for_job(self.job_id)
            final_bt = final_state.get("block_timestamps", {})
            latest_version_number = final_bt.get(tensor_name, 0)

            return jsonify({'version_number': latest_version_number})

        @self.app.route('/tensor_size', methods=['GET'])
        async def get_tensor_size():
            logging.info("Accessing /tensor_size endpoint")
            tensor_name = TENSOR_NAME
            file_path = os.path.join(self.base_dir, f'{tensor_name}.pt')
            if not os.path.exists(file_path):
                return jsonify({'error': 'Tensor not found'}), 404

            t = torch.load(file_path, map_location=device)
            size = t.numel()
            return jsonify({'size': size})

        @self.app.route('/data/state/temp/<path:filename>', methods=['GET'])
        async def get_data_file(filename):
            file_path = os.path.join(self.temp_dir, filename)
            if not os.path.abspath(file_path).startswith(os.path.abspath(self.temp_dir)):
                return jsonify({'error': 'File not found or access denied'}), 403
            if not os.path.exists(file_path):
                return jsonify({'error': 'File not found'}), 404
            response = await make_response(
                await send_from_directory(
                    directory=self.temp_dir,
                    file_name=filename,
                    mimetype='application/octet-stream',
                    as_attachment=True
                )
            )
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

            # no special logic needed here; we just store it
            return jsonify({
                'message': 'Tensor uploaded successfully',
                'tensor_url': f'/data/state/temp/{filename}'
            }), 200

        @self.app.route('/update_loss', methods=['POST'])
        async def update_loss():
            data = await request.get_json()
            if not data or 'loss' not in data:
                return jsonify({'error': 'Missing loss value'}), 400

            state_data = await self.db_adapter.get_sot_state_for_job(self.job_id)
            if "latest_loss" not in state_data:
                state_data["latest_loss"] = {}
            state_data["latest_loss"]["value"] = data['loss']
            await self.db_adapter.update_sot_state_for_job(self.job_id, state_data)
            logging.info(f"Updated latest loss for version {data['version_number']}: {data['loss']}")

            return jsonify({'status': 'success'}), 200

        @self.app.route('/get_loss', methods=['GET'])
        async def get_loss():
            logging.info("Accessing /get_loss endpoint")
            state_data = await self.db_adapter.get_sot_state_for_job(self.job_id)
            val = None
            if "latest_loss" in state_data:
                val = state_data["latest_loss"].get("value", None)
            return jsonify({'loss': val}), 200

        logging.info("default_sot_adapter initialized fully")
        await self.app.run_task(host='0.0.0.0', port=port)
