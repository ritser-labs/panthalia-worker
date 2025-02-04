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
    TENSOR_NAME,
    download_file,
    get_future_version_number
)
from ..device import device
from ..util.sot import (
    version_number_exists,
    get_local_file_path,
    apply_optimizer,
    initialize_all_tensors
)
from ..db.db_adapter_client import DBAdapterClient
from ..util.docker import janky_url_replace
from .sot_adapter import BaseSOTAdapter

from ..util import demo  # for chunked-DCT encode/decode

class DefaultSOTAdapter(BaseSOTAdapter):
    """
    A refactored SOT adapter that collects partial gradients from multiple workers
    in a single aggregator_sum.pt file and aggregator_count.json. No aggregator_{ver} directories.
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

        # We'll store aggregator sums in two files:
        #   aggregator_sum.pt (a single Tensor)
        #   aggregator_count.json => { "num_updates": <int> }
        self.aggregator_sum_path = os.path.join(self.base_dir, "aggregator_sum.pt")
        self.aggregator_count_path = os.path.join(self.base_dir, "aggregator_count.json")

        self.file_locks = None
        self.update_timestamp_lock = None
        self.synced_workers = 0
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        # Lock to ensure only one version-finalization happens at a time
        self.version_calculation_lock = asyncio.Lock()

        # Keep a map of newly generated diffs => new_version -> (rel_url, creation_time)
        self.versioned_diffs = {}

        # We'll run a background task that finalizes versions once the interval has elapsed
        self._version_task = None

        self.app = Quart(__name__)
        self.app.config['MAX_CONTENT_LENGTH'] = 1024**4  # 1 TB (arbitrary large limit)

    async def initialize(self, sot_id, db_url, private_key, job_id, perm_db, port):
        """
        Called once by the SOT side, passing relevant config. Then the SOT listens on HTTP.
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
            logging.info(f"[DefaultSOTAdapter.initialize] Found initial_state_url => {init_url}")
            local_path = os.path.join(self.base_dir, "model_0.pt")
            if not os.path.exists(local_path):
                logging.info(f"Downloading initial model => {local_path}")
                try:
                    await download_file(init_url, local_file_path=local_path, download_type='tensor', chunk_timeout=10)
                    logging.info("[initialize] Initial model downloaded.")
                except Exception as e:
                    logging.error(f"Failed to download initial_state_url: {e}", exc_info=True)
            else:
                logging.info("[initialize] model_0.pt already exists locally, skipping.")
        else:
            logging.info("[initialize] No initial_state_url found; using default init flow.")

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

        # ------------------- Register HTTP routes ------------------------

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
            combined_filename = f'input_{timestamp}_{random_suffix}.pt'
            combined_tensor = torch.cat([batch_tensor, targets_tensor], dim=0)

            out_path = os.path.join(self.temp_dir, combined_filename)
            await asyncio.to_thread(torch.save, combined_tensor, out_path)

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

            # 1) Check worker's version vs DB's current version
            tensor_name = TENSOR_NAME
            state_data = await self.db_adapter.get_sot_state_for_job(self.job_id)
            block_timestamps = state_data.get("block_timestamps", {})
            current_version_number = block_timestamps.get(tensor_name, 0)

            # => If aggregator has advanced beyond the worker's version, 409.
            if version_number != current_version_number:
                msg = f"Version mismatch: worker={version_number} vs. SOT={current_version_number}"
                logging.warning(f"[update_state] {msg}")
                return jsonify({'error': msg}), 409

            # 2) decode the partial gradient
            local_file_path = get_local_file_path(result_url, request, self.base_dir)
            if not local_file_path or not os.path.exists(local_file_path):
                logging.error(f"[update_state] gradient file not found => {local_file_path}")
                return jsonify({'error': 'File not found'}), 404

            loaded_obj = torch.load(local_file_path, map_location=device)
            os.remove(local_file_path)

            partial_grads = demo.chunked_dct_decode_int8(
                loaded_obj['freq_idxs'],
                loaded_obj['freq_vals_int8'],
                loaded_obj['freq_scales'],
                loaded_obj['freq_zero_points'],
                x_shape=loaded_obj['orig_shape'],
                chunk_shape=loaded_obj['chunk_shape'],
                norm='ortho',
                pad_count=loaded_obj.get('pad_count', 0)
            ).to(device)

            # optionally remove input file
            input_url = data.get('input_url')
            if input_url:
                inp_path = get_local_file_path(input_url, request, self.base_dir)
                if inp_path and os.path.exists(inp_path):
                    try:
                        os.remove(inp_path)
                    except Exception as ex:
                        logging.error(f"[update_state] Error removing input file: {ex}", exc_info=True)

            # 3) Add partial grads => aggregator_sum
            aggregator_sum = None
            if os.path.exists(self.aggregator_sum_path):
                aggregator_sum = torch.load(self.aggregator_sum_path, map_location=device)
                aggregator_sum = aggregator_sum + partial_grads
            else:
                aggregator_sum = partial_grads

            torch.save(aggregator_sum, self.aggregator_sum_path + ".tmp")
            os.rename(self.aggregator_sum_path + ".tmp", self.aggregator_sum_path)

            # aggregator_count => increment
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
            from_version_str = request.args.get('from_version', None)
            if from_version_str is None:
                return jsonify({'error': 'missing from_version'}), 400
            try:
                from_version = int(from_version_str)
            except ValueError:
                return jsonify({'error': 'invalid from_version'}), 400

            async with self.version_calculation_lock:
                tensor_name = TENSOR_NAME
                state_data = await self.db_adapter.get_sot_state_for_job(self.job_id)
                block_timestamps = state_data.get("block_timestamps", {})
                current_version_number = block_timestamps.get(tensor_name, 0)
                if from_version >= current_version_number:
                    return jsonify([])

                diffs_list = []
                for new_ver, (rel_url, created_ts) in self.versioned_diffs.items():
                    if from_version < new_ver <= current_version_number:
                        diffs_list.append(rel_url)

                # Sort them by ascending version
                sorted_pairs = sorted(
                    [(v, self.versioned_diffs[v][0]) for v in self.versioned_diffs],
                    key=lambda tup: tup[0]
                )
                final_list = [
                    url for v, url in sorted_pairs
                    if from_version < v <= current_version_number
                ]
                return jsonify(final_list)

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
                path = os.path.join(self.base_dir, f'{tensor_name}_{requested_version_num}.pt')
                if not os.path.exists(path):
                    return jsonify({'error': 'Tensor not found'}), 404

                resp = await make_response(
                    await send_from_directory(
                        directory=self.base_dir,
                        file_name=f'{tensor_name}_{requested_version_num}.pt',
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
            fpath = os.path.join(self.base_dir, f'{tensor_name}_{cur_ver}.pt')
            if not os.path.exists(fpath):
                return jsonify({'error': 'Tensor not found'}), 404
            t_ = torch.load(fpath, map_location=device)
            return jsonify({'size': t_.numel()})

        @self.app.route('/data/state/<path:filename>', methods=['GET'])
        async def get_state_file(filename):
            file_path = os.path.join(self.base_dir, filename)
            # basic checks
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
            fname = f'{label}_{upd_ver_num}_{random_suffix}.pt'
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

        # ------------------------------------------------------
        # Start background task to auto-finalize aggregator
        # ------------------------------------------------------
        self._version_task = asyncio.create_task(self._auto_version_loop())

        logging.info("[DefaultSOTAdapter] fully initialized; starting HTTP server...")
        await self.app.run_task(host='0.0.0.0', port=port)

    async def _auto_version_loop(self):
        tensor_name = TENSOR_NAME
        while True:
            await asyncio.sleep(0.1)  # poll every 0.1s

            try:
                async with self.version_calculation_lock:
                    state_data = await self.db_adapter.get_sot_state_for_job(self.job_id)
                    block_timestamps = state_data.get("block_timestamps", {})
                    curr_ver = block_timestamps.get(tensor_name, 0)

                    # The next boundary is a time-based integer from get_future_version_number(...).
                    boundary = get_future_version_number(self.tensor_version_interval)

                    # If the boundary is larger than curr_ver, it means we've entered a new time-slice.
                    if boundary > curr_ver:
                        # => finalize the aggregator from old version=curr_ver => new version=boundary
                        new_params_path, diff_file_path = await self._finalize_aggregator(
                            old_version=curr_ver,
                            new_version=boundary,
                            tensor_name=tensor_name
                        )
                        if new_params_path is not None:
                            # Mark the DB's "current version" as boundary
                            block_timestamps[tensor_name] = boundary
                            state_data["block_timestamps"] = block_timestamps

                            # Reset aggregator counters etc.
                            if "num_updates" not in state_data:
                                state_data["num_updates"] = {}
                            state_data["num_updates"][tensor_name] = 0

                            await self.db_adapter.update_sot_state_for_job(self.job_id, state_data)
                            logging.info(
                                f"[auto_version_loop] Finalized old v={curr_ver} => new v={boundary}"
                            )

            except Exception as e:
                logging.error(f"[auto_version_loop] error: {e}", exc_info=True)

            # optional: cleanup old files
            await self._cleanup_old_files()


    async def _finalize_aggregator(self, old_version, new_version, tensor_name):
        """
        Summarize aggregator_sum.pt + aggregator_count.json => final_grads => apply optimizer => produce new param + diff
        Then reset aggregator.
        """
        if not os.path.exists(self.aggregator_sum_path):
            #logging.info("[_finalize_aggregator] aggregator_sum.pt not found => skip.")
            return None, None

        aggregator_sum = torch.load(self.aggregator_sum_path, map_location=device)
        updates_count = 0
        if os.path.exists(self.aggregator_count_path):
            with open(self.aggregator_count_path, "r") as f:
                cdata = json.load(f)
            updates_count = cdata.get("num_updates", 0)

        if updates_count == 0:
            logging.info("[_finalize_aggregator] aggregator_count=0 => no partial grads => skip.")
            return None, None

        final_grads = aggregator_sum / float(updates_count)

        old_params_path = os.path.join(self.base_dir, f"{tensor_name}_{old_version}.pt")
        if not os.path.exists(old_params_path):
            logging.error(f"[_finalize_aggregator] Missing old params => cannot finalize from v={old_version}")
            return None, None

        # fetch hyperparams if provided
        if self.hyperparams_getter:
            it_val = 0
            hparams = self.hyperparams_getter(it_val)
            lr = hparams.get('learning_rate', 0.001)
            beta1 = hparams.get('beta1', 0.9)
            beta2 = hparams.get('beta2', 0.999)
            eps = hparams.get('epsilon', 1e-8)
            wd = hparams.get('weight_decay', 0.0)
            t_val = hparams.get('t', 0)
            chunk_shape = hparams.get('chunk_shape', 512)
            k_for_encoding = hparams.get('k', 1)
        else:
            lr = 0.001
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            wd = 0.0
            t_val = 0
            chunk_shape = 512
            k_for_encoding = 1

        # Apply optimizer => produce new_params
        new_params, new_m = await apply_optimizer(
            old_version, tensor_name,
            final_grads, lr, beta1, beta2, eps, wd, t_val,
            self.base_dir
        )

        new_params_path = os.path.join(self.base_dir, f"{tensor_name}_{new_version}.pt")
        torch.save(new_params, new_params_path + ".tmp")
        os.rename(new_params_path + ".tmp", new_params_path)

        # momentum
        future_m_path = os.path.join(self.base_dir, f"{tensor_name}_adam_m_{new_version}.pt")
        torch.save(new_m, future_m_path + ".tmp")
        os.rename(future_m_path + ".tmp", future_m_path)

        # produce chunked-DCT diff
        old_params_data = torch.load(old_params_path, map_location=device)
        diff = new_params - old_params_data
        (
            freq_idxs,
            freq_vals_int8,
            freq_scales,
            freq_zero_points,
            new_error,
            pad_count
        ) = demo.chunked_dct_encode_int8(
            diff,
            chunk_shape=chunk_shape,
            k=k_for_encoding,
            prev_error=None,
            norm='ortho'
        )

        diff_dict = {
            'freq_idxs': freq_idxs,
            'freq_vals_int8': freq_vals_int8,
            'freq_scales': freq_scales,
            'freq_zero_points': freq_zero_points,
            'chunk_shape': chunk_shape,
            'orig_shape': diff.shape,
            'pad_count': pad_count,
        }
        diff_file_path = os.path.join(self.base_dir, f"diff_{old_version}_to_{new_version}.pt")
        torch.save(diff_dict, diff_file_path)

        # record in self.versioned_diffs for /get_diffs_since
        rel_url = f'/data/state/diff_{old_version}_to_{new_version}.pt'
        self.versioned_diffs[new_version] = (rel_url, time.time())

        # reset aggregator
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
        Remove older .pt files from self.base_dir that exceed 2 days old,
        plus remove old diffs from self.versioned_diffs.
        """
        cutoff_seconds = 2 * 24 * 3600
        now = time.time()

        # 1) Clean up .pt files older than cutoff
        for fname in os.listdir(self.base_dir):
            if not fname.endswith(".pt"):
                continue
            full_path = os.path.join(self.base_dir, fname)
            try:
                stat_info = os.stat(full_path)
                age = now - stat_info.st_mtime
                if age > cutoff_seconds:
                    os.remove(full_path)
            except:
                pass

        # 2) Clean up old diffs in self.versioned_diffs
        for ver_key, (rel_url, created_ts) in list(self.versioned_diffs.items()):
            if (now - created_ts) > cutoff_seconds:
                base_name = os.path.basename(rel_url)
                local_diff_path = os.path.join(self.base_dir, base_name)
                if os.path.exists(local_diff_path):
                    try:
                        os.remove(local_diff_path)
                    except:
                        pass
                del self.versioned_diffs[ver_key]
