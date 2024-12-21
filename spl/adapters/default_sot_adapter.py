# spl/adapters/default_sot_adapter.py
import os
import asyncio
import logging
import random
import time
import torch
import json
from ..util.json import load_json, save_json
from ..common import get_future_version_number, TENSOR_NAME
from ..device import device
from ..util.sot import (
    version_number_exists, get_local_file_path, apply_optimizer,
    update_block_timestamps, cleanup_old_timestamp,
    update_cleanup_timestamps, initialize_all_tensors
)
from ..db.db_adapter_client import DBAdapterClient

class FakeRequest:
    def __init__(self, headers):
        self.headers = headers
        self.host = 'localhost:5001'  # Just a placeholder for get_local_file_path

def read_file(path):
    with open(path,'rb') as f:
        return f.read()

def read_torch_file(path):
    t=torch.load(path,map_location='cpu')
    import io
    buf=io.BytesIO()
    torch.save(t,buf)
    buf.seek(0)
    return buf.getvalue()

def parse_multipart(body,boundary):
    boundary=boundary.encode('utf-8')
    parts={}
    segments=body.split(b'--'+boundary)
    for seg in segments:
        seg=seg.strip()
        if not seg or seg==b'--':
            continue
        if b'\r\n\r\n' not in seg:
            continue
        header,data=seg.split(b'\r\n\r\n',1)
        header_lines=header.split(b'\r\n')
        name=None
        for hl in header_lines:
            hld=hl.decode('utf-8','replace')
            if 'Content-Disposition:' in hld:
                items=hld.split(';')
                for item in items:
                    item=item.strip()
                    if item.startswith('name="'):
                        name=item[6:-1]
        data=data.rstrip(b'\r\n')
        if name:
            parts[name]=data
    return parts

class DefaultSOTAdapter:
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

    async def initialize(self, sot_id, db_url, private_key, job_id, perm_db):
        self.sot_id = sot_id
        self.db_url = db_url
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
        
        logging.info("default_sot_adapter initialized fully")

    def _check_auth(self, headers):
        auth = headers.get('Authorization','')
        expected = f"Bearer {self.private_key}"
        return auth == expected

    async def handle_request(self, method, path, query, headers, body):
        if not self.initialized:
            return {'status':500,'headers':{},'body':b'{"error":"Not initialized"}'}

        if method == 'GET' and path == 'health':
            return {'status':200,'headers':{'Content-Type':'application/json'},'body':b'{"status":"healthy"}'}

        if method == 'POST' and path == 'report_sync':
            self.synced_workers += 1
            return {'status':200,'headers':{'Content-Type':'application/json'},'body':b'{"status":"ok"}'}

        if method == 'GET' and path == 'get_num_synced':
            resp = json.dumps(self.synced_workers).encode('utf-8')
            return {'status':200,'headers':{'Content-Type':'application/json'},'body':resp}

        if method == 'POST' and path == 'get_batch':
            if not self._check_auth(headers):
                return {'status':401,'headers':{},'body':b'{"error":"Unauthorized"}'}
            try:
                token_pairs = await self.dataset.__anext__()
                if not token_pairs:
                    return {'status':404,'headers':{},'body':b'{"error":"No more batches available"}'}
            except StopAsyncIteration:
                return {'status':404,'headers':{},'body':b'{"error":"No more batches available"}'}
            except Exception as e:
                logging.error(f"Error in /get_batch: {e}", exc_info=True)
                return {'status':500,'headers':{},'body':b'{"error":"Could not get batch"}'}

            inputs = [torch.tensor(pair[0], dtype=torch.long) for pair in token_pairs]
            targets = [torch.tensor(pair[1], dtype=torch.long) for pair in token_pairs]
            batch_tensor = torch.stack(inputs)
            targets_tensor = torch.stack(targets)
            timestamp = int(time.time())
            random_suffix = random.randint(1000, 9999)
            combined_filename = f'input_{timestamp}_{random_suffix}.pt'
            combined_tensor = torch.cat([batch_tensor, targets_tensor], dim=0)
            await asyncio.to_thread(torch.save, combined_tensor, os.path.join(self.temp_dir, combined_filename))
            resp_json = json.dumps({"input_url":f"/data/state/temp/{combined_filename}"}).encode('utf-8')
            return {'status':200,'headers':{'Content-Type':'application/json'},'body':resp_json}

        if method=='POST' and path=='update_state':
            if not self._check_auth(headers):
                return {'status':401,'headers':{},'body':b'{"error":"Unauthorized"}'}
            data = json.loads(body)
            tensor_name = data.get('tensor_name')
            result_url = data.get('result_url')
            if not tensor_name or not result_url:
                return {'status':400,'headers':{},'body':b'{"error":"Missing tensor_name or result_url"}'}

            state_dir = self.base_dir
            db_adapter = self.db_adapter
            job_id = self.job_id
            file_locks = self.file_locks
            update_timestamp_lock = self.update_timestamp_lock

            block_timestamps_file = os.path.join(state_dir, 'block_timestamps.json')
            num_updates_file = os.path.join(state_dir, 'num_updates.json')
            last_future_version_file = os.path.join(state_dir, 'last_future_version_number.json')
            iteration_number_file = os.path.join(state_dir, 'iteration_number.json')

            block_timestamps = await load_json(block_timestamps_file, {}, file_locks['block_timestamps'])
            num_updates = await load_json(num_updates_file, {}, file_locks['num_updates'])
            last_future_version_number = await load_json(last_future_version_file, {}, file_locks['last_future_version_number'])
            iteration_number = await load_json(iteration_number_file, {}, file_locks['iteration_number'])

            future_version_number = get_future_version_number(self.tensor_version_interval)

            if data['version_number'] != block_timestamps.get(tensor_name,0):
                return {'status':409,'headers':{},'body':b'{"error":"Version number mismatch"}'}

            old_block_timestamp = await update_block_timestamps(
                tensor_name, block_timestamps, num_updates, iteration_number,
                last_future_version_number, state_dir, db_adapter, job_id, self.tensor_version_interval,
                update_timestamp_lock, file_locks
            )

            fake_req=FakeRequest(headers)
            local_file_path = get_local_file_path(data.get('result_url'), fake_req, state_dir)
            batch_url = data.get('batch_url')
            targets_url = data.get('targets_url')
            local_batch_file_path = get_local_file_path(batch_url, fake_req, state_dir)
            local_targets_file_path = get_local_file_path(targets_url, fake_req, state_dir)
            try:
                if local_batch_file_path and os.path.exists(local_batch_file_path):
                    await asyncio.to_thread(os.remove, local_batch_file_path)
                if local_targets_file_path and os.path.exists(local_targets_file_path):
                    await asyncio.to_thread(os.remove, local_targets_file_path)
            except:
                pass

            if not local_file_path or not os.path.exists(local_file_path):
                return {'status':404,'headers':{},'body':b'{"error":"File not found"}'}

            tensor = torch.load(local_file_path, map_location=device)
            accumulated_grads_path = os.path.join(state_dir,f'accumulated_grads_{tensor_name}_{future_version_number}.pt')
            if os.path.exists(accumulated_grads_path):
                accumulated_grads = torch.load(accumulated_grads_path, map_location=device).to(device)
            else:
                accumulated_grads = torch.zeros_like(tensor, device=device)

            accumulated_grads += tensor.to(device)
            await asyncio.to_thread(torch.save, accumulated_grads, accumulated_grads_path)

            current_version_number = block_timestamps.get(tensor_name,0)
            num_of_updates = num_updates.get(tensor_name,0)+1
            num_updates[tensor_name]=num_of_updates
            await save_json(num_updates_file, num_updates, file_locks['num_updates'])

            averaged_grads=(accumulated_grads/num_of_updates).to(device)
            learning_params=await self.hyperparams_getter(iteration_number[tensor_name])
            future_tensor,m_update=await apply_optimizer(
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

            future_tensor_path=os.path.join(state_dir,f'{tensor_name}_{future_version_number}.pt')
            future_tensor_adam_m_path=os.path.join(state_dir,f'{tensor_name}_adam_m_{future_version_number}.pt')
            future_tensor_temp_path=future_tensor_path+'.tmp'
            future_tensor_adam_m_temp_path=future_tensor_adam_m_path+'.tmp'

            await asyncio.to_thread(torch.save,future_tensor,future_tensor_temp_path)
            await asyncio.to_thread(torch.save,m_update,future_tensor_adam_m_temp_path)
            os.rename(future_tensor_temp_path,future_tensor_path)
            os.rename(future_tensor_adam_m_temp_path,future_tensor_adam_m_path)

            await cleanup_old_timestamp(tensor_name,old_block_timestamp,block_timestamps,state_dir)
            for filename in os.listdir(state_dir):
                if filename.startswith(f'accumulated_grads_{tensor_name}_') and not filename.endswith(f'{future_version_number}.pt'):
                    await asyncio.to_thread(os.remove, os.path.join(state_dir, filename))

            if os.path.exists(local_file_path):
                await asyncio.to_thread(os.remove, local_file_path)
            resp=json.dumps({"status":"success","version_number":future_version_number}).encode('utf-8')
            return {'status':200,'headers':{'Content-Type':'application/json'},'body':resp}

        if method=='GET' and path=='latest_state':
            tensor_name=query.get('tensor_name')
            if not tensor_name:
                return {'status':400,'headers':{},'body':b'{"error":"Missing tensor_name parameter"}'}
            state_dir=self.base_dir
            file_locks=self.file_locks
            block_timestamps_file=os.path.join(state_dir,'block_timestamps.json')
            block_timestamps=await load_json(block_timestamps_file,{},file_locks['block_timestamps'])
            latest_version_number=query.get('version_number')
            if latest_version_number is None or not version_number_exists(latest_version_number,tensor_name,state_dir):
                latest_version_number=block_timestamps.get(tensor_name,0)
            else:
                latest_version_number=int(latest_version_number)
            state_file_path=os.path.join(state_dir,f'{tensor_name}_{latest_version_number}.pt')
            if not os.path.exists(state_file_path):
                return {'status':404,'headers':{},'body':b'{"error":"Tensor not found"}'}

            data=await asyncio.to_thread(read_torch_file, state_file_path)
            return {'status':200,'headers':{'Content-Type':'application/octet-stream','X-Version-Number':str(latest_version_number)},'body':data}

        if method=='POST' and path=='current_timestamp':
            tensor_name=query.get('tensor_name')
            if not tensor_name:
                return {'status':400,'headers':{},'body':b'{"error":"Missing tensor_name parameter"}'}
            state_dir=self.base_dir
            db_adapter=self.db_adapter
            job_id=self.job_id
            file_locks=self.file_locks
            block_timestamps_file=os.path.join(state_dir,'block_timestamps.json')
            block_timestamps=await load_json(block_timestamps_file,{},file_locks['block_timestamps'])
            num_updates_file=os.path.join(state_dir,'num_updates.json')
            num_updates=await load_json(num_updates_file,{},file_locks['num_updates'])
            iteration_number_file=os.path.join(state_dir,'iteration_number.json')
            iteration_number=await load_json(iteration_number_file,{},file_locks['iteration_number'])
            last_future_version_file=os.path.join(state_dir,'last_future_version_number.json')
            last_future_version_number=await load_json(last_future_version_file,{},file_locks['last_future_version_number'])

            await update_cleanup_timestamps(
                tensor_name, block_timestamps, num_updates, iteration_number,
                last_future_version_number, state_dir, db_adapter, job_id, self.tensor_version_interval,
                self.update_timestamp_lock, file_locks
            )
            latest_version_number=block_timestamps.get(tensor_name,0)
            resp=json.dumps({"version_number":latest_version_number}).encode('utf-8')
            return {'status':200,'headers':{'Content-Type':'application/json'},'body':resp}

        if method=='GET' and path=='tensor_size':
            tensor_name=query.get('tensor_name')
            if not tensor_name:
                return {'status':400,'headers':{},'body':b'{"error":"Missing tensor_name parameter"}'}
            state_dir=self.base_dir
            state_file_path=os.path.join(state_dir,f'{tensor_name}.pt')
            if not os.path.exists(state_file_path):
                return {'status':404,'headers':{},'body':b'{"error":"Tensor not found"}'}
            tensor=torch.load(state_file_path,map_location=device)
            size=tensor.numel()
            resp=json.dumps({"size":size}).encode('utf-8')
            return {'status':200,'headers':{'Content-Type':'application/json'},'body':resp}

        if method=='GET' and path.startswith('data/state/temp/'):
            filename=path[len('data/state/temp/'):]
            file_path=os.path.join(self.temp_dir,filename)
            if not os.path.abspath(file_path).startswith(os.path.abspath(self.temp_dir)):
                return {'status':403,'headers':{},'body':b'{"error":"File not found or access denied"}'}
            if not os.path.exists(file_path):
                return {'status':404,'headers':{},'body':b'{"error":"File not found"}'}
            data=await asyncio.to_thread(read_file,file_path)
            return {'status':200,'headers':{'Content-Type':'application/octet-stream'},'body':data}

        if method=='POST' and path=='upload_tensor':
            ctype=headers.get('Content-Type','')
            if 'multipart/form-data' not in ctype:
                return {'status':400,'headers':{},'body':b'{"error":"Not multipart"}'}
            boundary=ctype.split('boundary=')[-1]
            if not boundary:
                return {'status':400,'headers':{},'body':b'{"error":"No boundary"}'}

            parts=parse_multipart(body,boundary)
            if 'tensor' not in parts or 'label' not in parts:
                return {'status':400,'headers':{},'body':b'{"error":"No tensor file or label provided"}'}
            tensor_data=parts['tensor']
            label=parts['label'].decode('utf-8')
            update_version_number=int(time.time())
            random_suffix=random.randint(1000,9999)
            filename=f'{label}_{update_version_number}_{random_suffix}.pt'
            local_file_path=os.path.join(self.temp_dir,filename)
            import io
            t=torch.load(io.BytesIO(tensor_data),map_location=device)
            torch.save(t,local_file_path)
            resp=json.dumps({"message":"Tensor uploaded successfully","tensor_url":f"/data/state/temp/{filename}"}).encode('utf-8')
            return {'status':200,'headers':{'Content-Type':'application/json'},'body':resp}

        if method=='POST' and path=='update_loss':
            data=json.loads(body)
            if 'loss' not in data:
                return {'status':400,'headers':{},'body':b'{"error":"Missing loss value"}'}

            state_dir=self.base_dir
            latest_loss_path=os.path.join(state_dir,'latest_loss.json')
            latest_loss=await load_json(latest_loss_path,{'value':None},self.file_locks['latest_loss'])
            latest_loss['value']=data['loss']
            await save_json(latest_loss_path,latest_loss,self.file_locks['latest_loss'])
            return {'status':200,'headers':{'Content-Type':'application/json'},'body':b'{"status":"success"}'}

        if method=='GET' and path=='get_loss':
            state_dir=self.base_dir
            latest_loss_path=os.path.join(state_dir,'latest_loss.json')
            latest_loss=await load_json(latest_loss_path,{'value':None},self.file_locks['latest_loss'])
            loss=latest_loss.get('value')
            resp=json.dumps({'loss':loss}).encode('utf-8')
            return {'status':200,'headers':{'Content-Type':'application/json'},'body':resp}

        return {'status':404,'headers':{},'body':b'{"error":"Not found"}'}
