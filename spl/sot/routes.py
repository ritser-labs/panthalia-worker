# spl/sot/routes.py
import os
import json
import torch
import time
import random
import logging
import asyncio
import aiohttp
from quart import request, jsonify, make_response, send_from_directory
from ..auth.api_auth import requires_authentication
from ..common import (
    get_future_version_number, TENSOR_NAME, SOT_PRIVATE_PORT
)
from ..device import device
from .utils import (
    load_json, save_json, version_number_exists,
    get_local_file_path, apply_optimizer,
    update_block_timestamps, cleanup_old_timestamp,
    update_cleanup_timestamps
)

def register_routes(app):
    db_adapter = lambda: app.config['db_adapter']
    perm_db_lambda = lambda: app.config['perm_db']
    requires_auth = requires_authentication(db_adapter, perm_db_lambda)

    file_locks = app.config['file_locks']
    update_timestamp_lock = app.config['update_timestamp_lock']

    @app.route('/health', methods=['GET'])
    async def health_check():
        return jsonify({'status': 'healthy'}), 200

    @app.route('/report_sync', methods=['POST'])
    async def report_sync():
        app.config['synced_workers'] += 1
        return jsonify({'status': 'ok'})

    @app.route('/get_num_synced', methods=['GET'])
    async def get_num_synced():
        return jsonify(app.config['synced_workers'])

    @app.route('/get_batch', methods=['POST'])
    @requires_auth
    async def get_batch():
        logging.info("Accessing /get_batch endpoint")
        plugin = app.config['plugin']
        temp_dir = app.config['TEMP_DIR']
        try:
            token_pairs = await plugin.call_submodule('dataset', '__anext__')
            if not token_pairs:
                logging.info("No more batches available in /get_batch.")
                return jsonify({'error': 'No more batches available'}), 404
        except StopAsyncIteration:
            # If the plugin returns StopAsyncIteration, we treat it as no more data
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
        batch_filename = f'batch_{timestamp}_{random_suffix}.pt'
        targets_filename = f'targets_{timestamp}_{random_suffix}.pt'

        await asyncio.to_thread(torch.save, batch_tensor, os.path.join(temp_dir, batch_filename))
        await asyncio.to_thread(torch.save, targets_tensor, os.path.join(temp_dir, targets_filename))

        logging.info(f"Sending batch: {batch_filename}, targets: {targets_filename}")

        return jsonify({
            'batch_url': f'/data/state/temp/{batch_filename}',
            'targets_url': f'/data/state/temp/{targets_filename}'
        })

    @app.route('/update_state', methods=['POST'])
    @requires_auth
    async def update_state():
        logging.info("Accessing /update_state endpoint")
        data = await request.get_json()
        tensor_name = data.get('tensor_name')
        result_url = data.get('result_url')

        if not tensor_name or not result_url:
            logging.error("Missing tensor_name or result_url in /update_state")
            return jsonify({'error': 'Missing tensor_name or result_url'}), 400

        state_dir = app.config['STATE_DIR']
        db_adapter = app.config['db_adapter']
        job_id = app.config['job_id']
        plugin = app.config['plugin']

        block_timestamps_file = os.path.join(state_dir, 'block_timestamps.json')
        num_updates_file = os.path.join(state_dir, 'num_updates.json')
        last_future_version_file = os.path.join(state_dir, 'last_future_version_number.json')
        iteration_number_file = os.path.join(state_dir, 'iteration_number.json')

        block_timestamps = await load_json(block_timestamps_file, {}, file_locks['block_timestamps'])
        num_updates = await load_json(num_updates_file, {}, file_locks['num_updates'])
        last_future_version_number = await load_json(last_future_version_file, {}, file_locks['last_future_version_number'])
        iteration_number = await load_json(iteration_number_file, {}, file_locks['iteration_number'])

        future_version_number = get_future_version_number(await plugin.get('tensor_version_interval'))

        if data['version_number'] != block_timestamps.get(tensor_name, 0):
            delta = block_timestamps.get(tensor_name, 0) - data['version_number']
            logging.info(f'Delta of {delta} recorded with version number {data["version_number"]}')
            return jsonify({'error': 'Version number mismatch'}), 409

        old_block_timestamp = await update_block_timestamps(
            tensor_name, block_timestamps, num_updates, iteration_number,
            last_future_version_number, state_dir, db_adapter, job_id, plugin,
            update_timestamp_lock, file_locks
        )

        local_file_path = get_local_file_path(result_url, request, state_dir)
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
        await save_json(num_updates_file, num_updates, file_locks['num_updates'])

        averaged_grads = (accumulated_grads / num_of_updates).to(device)
        learning_params = await plugin.get_sot_learning_hyperparameters(iteration_number[tensor_name])
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

        # Cleanup old accumulated grads
        for filename in os.listdir(state_dir):
            if filename.startswith(f'accumulated_grads_{tensor_name}_') and not filename.endswith(f'{future_version_number}.pt'):
                await asyncio.to_thread(os.remove, os.path.join(state_dir, filename))

        if os.path.exists(local_file_path):
            await asyncio.to_thread(os.remove, local_file_path)
            logging.info(f"Deleted file: {local_file_path}")

        return jsonify({'status': 'success', 'version_number': future_version_number})

    @app.route('/latest_state', methods=['GET'])
    async def latest_state():
        logging.info("Accessing /latest_state endpoint")
        tensor_name = request.args.get('tensor_name')
        if not tensor_name:
            return jsonify({'error': 'Missing tensor_name parameter'}), 400

        state_dir = app.config['STATE_DIR']
        block_timestamps_file = os.path.join(state_dir, 'block_timestamps.json')
        block_timestamps = await load_json(block_timestamps_file, {}, file_locks['block_timestamps'])

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

    @app.route('/current_timestamp', methods=['POST'])
    async def current_timestamp():
        logging.info("Accessing /current_timestamp endpoint")
        tensor_name = request.args.get('tensor_name')
        if not tensor_name:
            return jsonify({'error': 'Missing tensor_name parameter'}), 400

        state_dir = app.config['STATE_DIR']
        db_adapter = app.config['db_adapter']
        job_id = app.config['job_id']
        plugin = app.config['plugin']

        block_timestamps_file = os.path.join(state_dir, 'block_timestamps.json')
        block_timestamps = await load_json(block_timestamps_file, {}, file_locks['block_timestamps'])
        num_updates_file = os.path.join(state_dir, 'num_updates.json')
        num_updates = await load_json(num_updates_file, {}, file_locks['num_updates'])
        iteration_number_file = os.path.join(state_dir, 'iteration_number.json')
        iteration_number = await load_json(iteration_number_file, {}, file_locks['iteration_number'])
        last_future_version_file = os.path.join(state_dir, 'last_future_version_number.json')
        last_future_version_number = await load_json(last_future_version_file, {}, file_locks['last_future_version_number'])

        await update_cleanup_timestamps(
            tensor_name, block_timestamps, num_updates, iteration_number,
            last_future_version_number, state_dir, db_adapter, job_id, plugin,
            app.config['update_timestamp_lock'], file_locks
        )

        latest_version_number = block_timestamps.get(tensor_name, 0)
        return jsonify({'version_number': latest_version_number})

    @app.route('/tensor_size', methods=['GET'])
    async def get_tensor_size():
        logging.info("Accessing /tensor_size endpoint")
        tensor_name = request.args.get('tensor_name')
        if not tensor_name:
            return jsonify({'error': 'Missing tensor_name parameter'}), 400

        state_dir = app.config['STATE_DIR']
        state_file_path = os.path.join(state_dir, f'{tensor_name}.pt')
        if not os.path.exists(state_file_path):
            return jsonify({'error': 'Tensor not found'}), 404

        tensor = torch.load(state_file_path, map_location=device)
        size = tensor.numel()
        return jsonify({'size': size})

    @app.route('/data/state/temp/<path:filename>', methods=['GET'])
    async def get_data_file(filename):
        temp_dir = app.config['TEMP_DIR']
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

    @app.route('/upload_tensor', methods=['POST'])
    async def upload_tensor():
        request_files = await request.files
        if 'tensor' not in request_files:
            return jsonify({'error': 'No tensor file provided'}), 400

        request_form = await request.form
        if 'label' not in request_form:
            return jsonify({'error': 'No label provided'}), 400

        temp_dir = app.config['TEMP_DIR']
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

    @app.route('/update_loss', methods=['POST'])
    async def update_loss():
        data = await request.get_json()
        if not data or 'loss' not in data:
            return jsonify({'error': 'Missing loss value'}), 400

        state_dir = app.config['STATE_DIR']
        latest_loss_path = os.path.join(state_dir, 'latest_loss.json')

        latest_loss = await load_json(latest_loss_path, {'value': None}, file_locks['latest_loss'])
        latest_loss['value'] = data['loss']
        await save_json(latest_loss_path, latest_loss, file_locks['latest_loss'])
        logging.info(f"Updated latest loss for version {data['version_number']}: {data['loss']}")

        return jsonify({'status': 'success'}), 200

    @app.route('/get_loss', methods=['GET'])
    async def get_loss():
        logging.info("Accessing /get_loss endpoint")
        state_dir = app.config['STATE_DIR']
        latest_loss_path = os.path.join(state_dir, 'latest_loss.json')
        latest_loss = await load_json(latest_loss_path, {'value': None}, file_locks['latest_loss'])
        loss = latest_loss.get('value')
        return jsonify({'loss': loss}), 200
