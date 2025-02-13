# spl/adapters/default_sot_adapter.py

import os
import asyncio
import logging
import random
import time
import json
import torch
import io
from quart import Quart, request, jsonify, make_response, send_from_directory

from ..common import (
    TENSOR_NAME,
    download_file,
    get_future_version_number,
    NoMoreDataException
)
from ..device import device, safetensors_device
from ..util.sot import (
    version_number_exists,
    get_local_file_path,
    initialize_all_tensors
)
from ..db.db_adapter_client import DBAdapterClient
from ..util.docker import janky_url_replace
from .sot_adapter import BaseSOTAdapter

# Import your chunked-DCT and AdamW logic
from ..util import demo
from ..util.adam import adamw_update
from safetensors.torch import save_file as safetensors_save_file
from safetensors.torch import load_file as safetensors_load_file

logging.basicConfig(level=logging.INFO)

class DefaultSOTAdapter(BaseSOTAdapter):
    """
    A SOT adapter that:
      1) Accumulates partial gradients in aggregator_sum.safetensors and aggregator_count.json
      2) On each 'time boundary' (or whenever get_future_version_number(...) is larger),
         finalizes new_params using AdamW, saves them as model_<ver>.safetensors
      3) Then chunked-DCT-encodes the final diff for distribution to workers
      4) Maintains aggregator_error for progressive encoding across versions.
    """

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

        self.aggregator_sum_path = os.path.join(self.base_dir, "aggregator_sum.safetensors")
        self.aggregator_count_path = os.path.join(self.base_dir, "aggregator_count.json")
        self.aggregator_error_path = os.path.join(self.base_dir, "aggregator_error.safetensors")

        self.file_locks = None
        self.update_timestamp_lock = None
        self.synced_workers = 0
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        self.version_calculation_lock = asyncio.Lock()
        self.versioned_diffs = {}

        self._version_task = None

        self.app = Quart(__name__)
        self.app.config['MAX_CONTENT_LENGTH'] = 1024**4  # 1 TB

    async def initialize(self, sot_id, db_url, private_key, job_id, perm_db, port):
        self.sot_id = sot_id
        self.db_url = janky_url_replace(db_url)
        self.private_key = private_key
        self.job_id = job_id
        self.perm_db = perm_db
        self.db_adapter = DBAdapterClient(self.db_url, self.private_key)

        self.file_locks = {}
        self.update_timestamp_lock = asyncio.Lock()

        # Possibly download initial model_0.safetensors if 'initial_state_url' is set
        job_obj = await self.db_adapter.sot_get_job(self.job_id)
        if job_obj and job_obj.initial_state_url and job_obj.initial_state_url.strip():
            init_url = job_obj.initial_state_url.strip()
            logging.info(f"[DefaultSOTAdapter.initialize] Found initial_state_url => {init_url}")
            local_path = os.path.join(self.base_dir, "model_0.safetensors")
            if not os.path.exists(local_path):
                logging.info(f"Downloading initial model => {local_path}")
                try:
                    await download_file(init_url, local_file_path=local_path,
                                        download_type='tensor', chunk_timeout=10)
                    logging.info("[initialize] Initial model downloaded.")
                except Exception as e:
                    logging.error(f"Failed to download initial_state_url: {e}", exc_info=True)
            else:
                logging.info("[initialize] model_0.safetensors already exists locally, skipping.")
        else:
            logging.info("[initialize] No initial_state_url found; using default init flow.")

        # Ensure model_0, model_0_adam_m, model_0_adam_v exist
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

        # ----------------------------------------------------------------------
        # Plug in basic auth
        # ----------------------------------------------------------------------
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
            logging.info("[get_batch] Worker requests a batch.")
            try:
                token_pairs = await self.dataset.__anext__()
                if not token_pairs:
                    logging.info("[get_batch] No more batches available.")
                    return jsonify({'error': 'No more batches available'}), 404
            except StopAsyncIteration:
                logging.info("[get_batch] Dataset exhausted.")
                return jsonify({'error': 'No more batches available'}), 404
            except Exception as e:
                logging.error(f"[get_batch] Error: {e}", exc_info=True)
                return jsonify({'error': 'Could not get batch'}), 500

            import torch
            inputs = [torch.tensor(pair[0], dtype=torch.long) for pair in token_pairs]
            targets = [torch.tensor(pair[1], dtype=torch.long) for pair in token_pairs]
            batch_tensor = torch.stack(inputs)
            targets_tensor = torch.stack(targets)

            timestamp = int(time.time())
            random_suffix = random.randint(1000, 9999)
            combined_filename = f'input_{timestamp}_{random_suffix}.safetensors'
            result_dict = {
                'inputs': batch_tensor,
                'targets': targets_tensor
            }

            out_path = os.path.join(self.temp_dir, combined_filename)

            safetensors_save_file(result_dict, out_path)

            return jsonify({
                'input_url': f'/data/state/temp/{combined_filename}'
            })

        @self.app.route('/update_state', methods=['POST'])
        @requires_auth
        async def update_state():
            logging.info("[update_state] Received partial gradient")
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
            current_version_number = block_timestamps.get(tensor_name, 0)

            if version_number != current_version_number:
                msg = f"Version mismatch: worker={version_number} vs. SOT={current_version_number}"
                logging.warning(f"[update_state] {msg}")
                return jsonify({'error': msg}), 409

            local_file_path = get_local_file_path(result_url, request, self.base_dir)
            if not local_file_path or not os.path.exists(local_file_path):
                logging.error(f"[update_state] gradient file not found => {local_file_path}")
                return jsonify({'error': 'File not found'}), 404

            # Load the encoded gradient
            loaded_obj = safetensors_load_file(local_file_path, device=safetensors_device)

            # Decode partial grads from chunked DCT
            partial_grads = demo.chunked_dct_decode_int8(
                loaded_obj['freq_idxs'].to(device),
                loaded_obj['freq_vals_int8'].to(device),
                loaded_obj['freq_scales'].to(device),
                loaded_obj['freq_zero_points'].to(device),
                x_shape=tuple(loaded_obj['orig_shape'].tolist()),
                chunk_shape=loaded_obj['chunk_shape'].to(device),
                norm='ortho',
                pad_count=loaded_obj.get('pad_count', 0)
            ).to(device)

            # Remove the input file if we got one
            input_url = data.get('input_url')
            if input_url:
                inp_path = get_local_file_path(input_url, request, self.base_dir)
                if inp_path and os.path.exists(inp_path):
                    try:
                        os.remove(inp_path)
                    except Exception as ex:
                        logging.error(f"[update_state] Error removing input file: {ex}", exc_info=True)

            # Update aggregator_sum
            if os.path.exists(self.aggregator_sum_path):
                loaded_dict = safetensors_load_file(self.aggregator_sum_path)
                aggregator_sum = loaded_dict["tensor"].to(device)
            else:
                aggregator_sum = partial_grads

            tmp_path = self.aggregator_sum_path + ".tmp"
            safetensors_save_file({"tensor": aggregator_sum}, tmp_path)
            os.rename(tmp_path, self.aggregator_sum_path)


            # Update aggregator_count
            updates_count = 0
            if os.path.exists(self.aggregator_count_path):
                with open(self.aggregator_count_path, "r") as f:
                    cdata = json.load(f)
                updates_count = cdata.get("num_updates", 0)
            updates_count += 1
            with open(self.aggregator_count_path + ".tmp", "w") as f:
                json.dump({"num_updates": updates_count}, f, indent=2)
            os.rename(self.aggregator_count_path + ".tmp", self.aggregator_count_path)

            logging.info(f"[update_state] aggregator_sum updated => total count={updates_count}")
            return jsonify({'status': 'accumulated'})

        @self.app.route('/get_diffs_since', methods=['GET'])
        async def get_diffs_since():
            """
            Now returns the diffs in the version range (from_version+1) ... (end_time),
            where end_time defaults to the SOT's current version if not provided.
            
            Sample JSON response:
            {
                "diffs": [...],
                "used_end_time": <int>
            }
            """
            from_version_str = request.args.get('from_version', None)
            end_time_str = request.args.get('end_time', None)

            if from_version_str is None:
                return jsonify({'error': 'missing from_version'}), 400

            try:
                from_version = int(from_version_str)
            except ValueError:
                return jsonify({'error': 'invalid from_version'}), 400

            # Grab current SOT version
            tensor_name = TENSOR_NAME
            state_data = await self.db_adapter.get_sot_state_for_job(self.job_id)
            block_timestamps = state_data.get("block_timestamps", {})
            current_version_number = block_timestamps.get(tensor_name, 0)

            if end_time_str is None:
                # If no end_time given, we interpret that as "up to now"
                end_time = current_version_number
            else:
                try:
                    end_time = int(end_time_str)
                except ValueError:
                    return jsonify({'error': 'invalid end_time'}), 400
                # If user passes an end_time bigger than the current known version,
                # you can clamp it or just let them get an empty set beyond the real version.
                if end_time > current_version_number:
                    end_time = current_version_number

            # If from_version is already >= end_time, no diffs
            if from_version >= end_time:
                return jsonify({
                    'diffs': [],
                    'used_end_time': end_time
                })

            # Gather the diffs that belong to version numbers in (from_version, end_time]
            diffs_list = []
            for ver_num, (rel_url, created_ts) in self.versioned_diffs.items():
                # we only include diff if from_version < ver_num <= end_time
                if from_version < ver_num <= end_time:
                    diffs_list.append(rel_url)

            # Sort so the diffs appear in ascending version order
            diffs_list = sorted(
                diffs_list,
                key=lambda url: int(url.split('_')[-1].split('.')[0])  # a crude parse if needed
            )

            return jsonify({
                'diffs': diffs_list,
                'used_end_time': end_time
            })

        @self.app.route('/latest_state', methods=['GET'])
        async def latest_state():
            logging.info("[latest_state] Accessing endpoint")
            tensor_name = TENSOR_NAME
            async with self.version_calculation_lock:
                state_data = await self.db_adapter.get_sot_state_for_job(self.job_id)
                block_timestamps = state_data.get("block_timestamps", {})
                requested_version_str = request.args.get('version_number', None)
                if requested_version_str is None:
                    requested_version_num = block_timestamps.get(tensor_name, 0)
                else:
                    requested_version_num = int(requested_version_str)
                    if not version_number_exists(requested_version_num, tensor_name, self.base_dir):
                        requested_version_num = block_timestamps.get(tensor_name, 0)
                path = os.path.join(self.base_dir, f'{tensor_name}_{requested_version_num}.safetensors')
                if not os.path.exists(path):
                    return jsonify({'error': 'Tensor not found'}), 404

                resp = await make_response(
                    await send_from_directory(
                        directory=self.base_dir,
                        file_name=f'{tensor_name}_{requested_version_num}.safetensors',
                        mimetype='application/octet-stream',
                        as_attachment=True
                    )
                )
                resp.headers['X-Version-Number'] = str(requested_version_num)
                return resp

        @self.app.route('/current_timestamp', methods=['POST'])
        async def current_timestamp():
            tensor_name = TENSOR_NAME
            state_data = await self.db_adapter.get_sot_state_for_job(self.job_id)
            block_timestamps = state_data.get("block_timestamps", {})
            latest_version_number = block_timestamps.get(tensor_name, 0)
            return jsonify({'version_number': latest_version_number})

        @self.app.route('/tensor_size', methods=['GET'])
        async def get_tensor_size():
            logging.info("[tensor_size] Accessing endpoint")
            tensor_name = TENSOR_NAME
            state_data = await self.db_adapter.get_sot_state_for_job(self.job_id)
            block_timestamps = state_data.get("block_timestamps", {})
            cur_ver = block_timestamps.get(tensor_name, 0)
            fpath = os.path.join(self.base_dir, f'{tensor_name}_{cur_ver}.safetensors')
            if not os.path.exists(fpath):
                return jsonify({'error': 'Tensor not found'}), 404
            t_ = safetensors_load_file(fpath, device=safetensors_device)["tensor"].to(device)
            return jsonify({'size': t_.numel()})

        @self.app.route('/data/state/<path:filename>', methods=['GET'])
        async def get_state_file(filename):
            file_path = os.path.join(self.base_dir, filename)
            if not os.path.abspath(file_path).startswith(os.path.abspath(self.base_dir)):
                return jsonify({'error': 'File not found or access denied'}), 403
            if not os.path.exists(file_path):
                return jsonify({'error': 'File not found'}), 404

            resp = await make_response(
                await send_from_directory(
                    directory=self.base_dir,
                    file_name=filename,
                    mimetype='application/octet-stream',
                    as_attachment=True
                )
            )
            return resp

        @self.app.route('/upload_tensor', methods=['POST'])
        async def upload_tensor():
            req_files = await request.files
            if 'tensor' not in req_files:
                return jsonify({'error': 'No tensor file provided'}), 400
            req_form = await request.form
            if 'label' not in req_form:
                return jsonify({'error': 'No label provided'}), 400

            tensor_file = req_files['tensor']
            label = req_form['label']
            upd_ver_num = int(time.time())
            random_suffix = random.randint(1000, 9999)
            fname = f'{label}_{upd_ver_num}_{random_suffix}.safetensors'
            local_file_path = os.path.join(self.temp_dir, fname)

            chunk_size = 1024 * 1024
            import aiofiles
            async with aiofiles.open(local_file_path, 'wb') as f:
                while True:
                    chunk = tensor_file.read(chunk_size)
                    if not chunk:
                        break
                    await f.write(chunk)

            return jsonify({
                'message': 'Tensor uploaded successfully',
                'tensor_url': f'/data/state/temp/{fname}'
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
            logging.info(f"[update_loss] Updated loss => {data['loss']}")
            return jsonify({'status': 'success'}), 200

        @self.app.route('/get_loss', methods=['GET'])
        async def get_loss():
            state_data = await self.db_adapter.get_sot_state_for_job(self.job_id)
            val = None
            if "latest_loss" in state_data:
                val = state_data["latest_loss"].get("value", None)
            return jsonify({'loss': val}), 200

        # Start the background versioning loop
        self._version_task = asyncio.create_task(self._auto_version_loop())

        logging.info("[DefaultSOTAdapter] fully initialized; starting HTTP server...")
        await self.app.run_task(host='0.0.0.0', port=port)

    async def _auto_version_loop(self):
        tensor_name = TENSOR_NAME
        while True:
            await asyncio.sleep(0.1)
            try:
                async with self.version_calculation_lock:
                    state_data = await self.db_adapter.get_sot_state_for_job(self.job_id)
                    block_timestamps = state_data.get("block_timestamps", {})
                    curr_ver = block_timestamps.get(tensor_name, 0)

                    boundary = get_future_version_number(self.tensor_version_interval)

                    if boundary > curr_ver:
                        new_params_path, diff_file_path = await self._finalize_aggregator(
                            old_version=curr_ver,
                            new_version=boundary,
                            tensor_name=tensor_name
                        )
                        if new_params_path is not None:
                            block_timestamps[tensor_name] = boundary
                            state_data["block_timestamps"] = block_timestamps
                            if "num_updates" not in state_data:
                                state_data["num_updates"] = {}
                            state_data["num_updates"][tensor_name] = 0
                            await self.db_adapter.update_sot_state_for_job(self.job_id, state_data)
                            logging.info(
                                f"[auto_version_loop] Finalized old v={curr_ver} => new v={boundary}"
                            )
            except Exception as e:
                logging.error(f"[auto_version_loop] error: {e}", exc_info=True)

            await self._cleanup_old_files()

    async def _finalize_aggregator(self, old_version, new_version, tensor_name):
        """
        Use AdamW for param updates, then encode the param diff with chunked‐DCT + aggregator_error.
        """
        if not os.path.exists(self.aggregator_sum_path):
            return None, None

        aggregator_sum = safetensors_load_file(self.aggregator_sum_path, device=safetensors_device)["tensor"].to(device)
        updates_count = 0
        if os.path.exists(self.aggregator_count_path):
            with open(self.aggregator_count_path, "r") as f:
                cdata = json.load(f)
            updates_count = cdata.get("num_updates", 0)

        if updates_count == 0:
            logging.info("[_finalize_aggregator] aggregator_count=0 => no partial grads => skip.")
            return None, None

        final_grads = aggregator_sum / float(updates_count)
        logging.info(
            f"[_finalize_aggregator] aggregator_sum.shape={aggregator_sum.shape}, "
            f"updates_count={updates_count}, aggregator_sum.norm={aggregator_sum.norm().item():.6f}"
        )
        logging.info(
            f"[_finalize_aggregator] final_grads.norm={final_grads.norm().item():.6f}"
        )

        old_params_path = os.path.join(self.base_dir, f"{tensor_name}_{old_version}.safetensors")
        if not os.path.exists(old_params_path):
            logging.error(f"[_finalize_aggregator] Missing old params => cannot finalize from v={old_version}")
            return None, None

        old_dict = safetensors_load_file(old_params_path)
        old_params_data = old_dict["tensor"].to(device)

        # Load old Adam moments
        old_m_path = os.path.join(self.base_dir, f"{tensor_name}_adam_m_{old_version}.safetensors")
        old_v_path = os.path.join(self.base_dir, f"{tensor_name}_adam_v_{old_version}.safetensors")
        if os.path.exists(old_m_path):
            m_vector_dict = safetensors_load_file(old_m_path)
            m_vector = m_vector_dict["tensor"].to(device)
        else:
            m_vector = torch.zeros_like(old_params_data)
        if os.path.exists(old_v_path):
            v_vector_dict = safetensors_load_file(old_v_path)
            v_vector = v_vector_dict["tensor"].to(device)
        else:
            v_vector = torch.zeros_like(old_params_data)

        # Pull iteration_number from DB
        state_data = await self.db_adapter.get_sot_state_for_job(self.job_id)
        iteration_number_map = state_data.get("iteration_number", {})
        iteration_val = iteration_number_map.get(tensor_name, 0)

        # Retrieve or define AdamW hyperparams
        if self.hyperparams_getter is not None:
            hyperparams = self.hyperparams_getter(iteration_val)
        else:
            hyperparams = {
                'learning_rate': 1e-3,
                'beta1': 0.9,
                'beta2': 0.999,
                'epsilon': 1e-8,
                'weight_decay': 0.01,
                't': iteration_val
            }

        # AdamW => new_params_prelim
        new_params_prelim, new_m, new_v = adamw_update(
            param_vector=old_params_data.to(device),
            grad_vector=final_grads.to(device),
            m_vector=m_vector.to(device),
            v_vector=v_vector.to(device),
            lr=hyperparams['learning_rate'],
            beta1=hyperparams['beta1'],
            beta2=hyperparams['beta2'],
            eps=hyperparams['epsilon'],
            weight_decay=hyperparams['weight_decay'],
            step=int(hyperparams['t'] + 1),
        )

        final_diff = new_params_prelim - old_params_data
        logging.info(
            f"[_finalize_aggregator] final_diff.norm={final_diff.norm().item():.6f}, "
            f"max_abs={final_diff.abs().max().item():.6f}"
        )

        # aggregator_error for chunked‐DCT
        if os.path.exists(self.aggregator_error_path):
            aggregator_error = safetensors_load_file(self.aggregator_error_path, device=safetensors_device)["tensor"].to(device)
            logging.info(f"[_finalize_aggregator] Found aggregator_error with norm={aggregator_error.norm().item():.6f}")
        else:
            aggregator_error = None

        chunk_shape = hyperparams['chunk_shape']
        k_val = hyperparams['k']
        (
            freq_idxs,
            freq_vals_int8,
            freq_scales,
            freq_zero_points,
            new_error,
            pad_count
        ) = demo.chunked_dct_encode_int8(
            x=final_diff,
            chunk_shape=chunk_shape,
            k=k_val,
            prev_error=aggregator_error,
            norm='ortho'
        )

        logging.info(
            f"[_finalize_aggregator] final_diff => diff.norm={final_diff.norm().item():.6f}, "
            f"max_abs={final_diff.abs().max().item():.6f}"
        )

        safetensors_save_file({"tensor": new_error}, self.aggregator_error_path + ".tmp")
        os.rename(self.aggregator_error_path + ".tmp", self.aggregator_error_path)
        logging.info(
            f"[_finalize_aggregator] aggregator_error updated => norm={new_error.norm().item():.6f}"
        )

        diff_dict = {
            'freq_idxs': freq_idxs,
            'freq_vals_int8': freq_vals_int8,
            'freq_scales': freq_scales,
            'freq_zero_points': freq_zero_points,
            'chunk_shape': chunk_shape,
            'orig_shape': torch.tensor(final_diff.shape, dtype=torch.int64),
            'pad_count': pad_count,
        }
        diff_file_path = os.path.join(self.base_dir, f"diff_{old_version}_to_{new_version}.safetensors")
        safetensors_save_file(diff_dict, diff_file_path)

        rel_url = f'/data/state/diff_{old_version}_to_{new_version}.safetensors'
        self.versioned_diffs[new_version] = (rel_url, time.time())

        # Decode that diff so aggregator param = old_params + exactly that decoded_diff
        decoded_diff = demo.chunked_dct_decode_int8(
            freq_idxs, freq_vals_int8, freq_scales, freq_zero_points,
            x_shape=final_diff.shape,
            chunk_shape=chunk_shape,
            norm='ortho',
            pad_count=pad_count
        ).to(device)

        new_params = old_params_data + decoded_diff

        new_params_path = os.path.join(self.base_dir, f"{tensor_name}_{new_version}.safetensors")
        safetensors_save_file({"tensor": new_params}, new_params_path + ".tmp")
        os.rename(new_params_path + ".tmp", new_params_path)

        # Save new_m, new_v
        new_m_path = os.path.join(self.base_dir, f"{tensor_name}_adam_m_{new_version}.safetensors")
        safetensors_save_file({"tensor": new_m}, new_m_path + ".tmp")
        os.rename(new_m_path + ".tmp", new_m_path)

        new_v_path = os.path.join(self.base_dir, f"{tensor_name}_adam_v_{new_version}.safetensors")
        safetensors_save_file({"tensor": new_v}, new_v_path + ".tmp")
        os.rename(new_v_path + ".tmp", new_v_path)

        logging.info(
            f"[_finalize_aggregator] Created new params => v={new_version}, "
            f"and diff => {diff_file_path}"
        )

        # Cleanup aggregator files
        try:
            os.remove(self.aggregator_sum_path)
        except:
            pass
        try:
            os.remove(self.aggregator_count_path)
        except:
            pass

        return new_params_path, diff_file_path

    async def _cleanup_old_files(self):
        """
        Periodically remove .safetensors files or diffs older than 48 hours for housekeeping.
        
        This function cleans up:
        1) Files in the base directory (e.g. model parameters, diff files),
        2) Files in the temp directory (files uploaded via upload_tensor),
        3) And entries in self.versioned_diffs whose files are older than 48 hours.
        """
        cutoff_seconds = 2 * 24 * 3600  # 48 hours in seconds
        now = time.time()

        # 1) Clean up files in the base directory
        for fname in os.listdir(self.base_dir):
            if not fname.endswith(".safetensors"):
                continue
            full_path = os.path.join(self.base_dir, fname)
            try:
                stat_info = os.stat(full_path)
                age = now - stat_info.st_mtime
                if age > cutoff_seconds:
                    os.remove(full_path)
                    self.logger.info(f"Removed old file: {full_path}")
            except Exception as e:
                self.logger.error(f"Error removing file {full_path}: {e}")

        # 2) Clean up temporary files in the temp directory (uploaded via upload_tensor)
        if hasattr(self, 'temp_dir') and os.path.isdir(self.temp_dir):
            for fname in os.listdir(self.temp_dir):
                if not fname.endswith(".safetensors"):
                    continue
                full_path = os.path.join(self.temp_dir, fname)
                try:
                    stat_info = os.stat(full_path)
                    age = now - stat_info.st_mtime
                    if age > cutoff_seconds:
                        os.remove(full_path)
                        self.logger.info(f"Removed old temp file: {full_path}")
                except Exception as e:
                    self.logger.error(f"Error removing temp file {full_path}: {e}")

        # 3) Remove old diffs from self.versioned_diffs
        for ver_key, (rel_url, created_ts) in list(self.versioned_diffs.items()):
            if (now - created_ts) > cutoff_seconds:
                base_name = os.path.basename(rel_url)
                local_diff_path = os.path.join(self.base_dir, base_name)
                if os.path.exists(local_diff_path):
                    try:
                        os.remove(local_diff_path)
                        self.logger.info(f"Removed old diff file: {local_diff_path}")
                    except Exception as e:
                        self.logger.error(f"Error removing diff file {local_diff_path}: {e}")
                del self.versioned_diffs[ver_key]