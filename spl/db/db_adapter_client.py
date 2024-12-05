# db_adapter_client.py

import logging
import aiohttp
import json
import time
import uuid
from eth_account.messages import encode_defunct
from eth_account import Account
from typing import Optional, List, Type, TypeVar, Dict, Any
from ..models import (
    Job, Plugin, Subnet, Task, TaskStatus, Perm, Sot, Instance, ServiceType, Base, PermType
)
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Generic Type for SQLAlchemy models
T = TypeVar('T', bound=Base)


class DBAdapterClient:
    def __init__(self, base_url, private_key=None):
        self.base_url = base_url.rstrip('/')
        self.private_key = private_key

    def _generate_message(self, endpoint, data=None):
        message = {
            "endpoint": endpoint,
            "nonce": str(uuid.uuid4()),
            "timestamp": int(time.time()),
            "data": data or {}
        }
        return json.dumps(message, sort_keys=True)

    def _sign_message(self, message):
        if not self.private_key:
            return None
        message_defunct = encode_defunct(text=message)
        account = Account.from_key(self.private_key)
        signed_message = account.sign_message(message_defunct)
        return signed_message.signature.hex()

    async def _authenticated_request(self, method: str, endpoint: str, data=None, params=None):
        url = f"{self.base_url}{endpoint}"
        headers = {}

        if self.private_key:
            message = self._generate_message(endpoint, data)
            signature = self._sign_message(message)
            headers = {"Authorization": f"{message}:{signature}"}

        async with aiohttp.ClientSession() as session:
            try:
                async with session.request(method, url, json=data, params=params, headers=headers) as response:
                    response.raise_for_status()
                    json_response = await response.json()
                    if not isinstance(json_response, dict) and not isinstance(json_response, list):
                        logger.error(f"Unexpected response format from {url}: {json_response}")
                        return {'error': 'Unexpected response format'}
                    return json_response
            except aiohttp.ClientError as e:
                logger.error(f"Request to {url} failed: {e}")
                return {'error': str(e)}

    def _extract_id(self, response: Dict[str, Any], id_key: str) -> Optional[int]:
        if 'error' in response:
            return None
        return response.get(id_key) or response.get('data', {}).get(id_key)

    # --- JOBS ---
    async def get_job(self, job_id: int) -> Optional[Job]:
        return await self._fetch_entity('/get_job', Job, params={'job_id': job_id})

    async def create_job(self, name: str, plugin_id: int, subnet_id: int, sot_url: str, iteration: int) -> Optional[int]:
        data = {
            'name': name,
            'plugin_id': plugin_id,
            'subnet_id': subnet_id,
            'sot_url': sot_url,
            'iteration': iteration
        }
        response = await self._authenticated_request('POST', '/create_job', data=data)
        return self._extract_id(response, 'job_id')

    async def update_job_iteration(self, job_id: int, new_iteration: int) -> bool:
        data = {
            'job_id': job_id,
            'new_iteration': new_iteration
        }
        response = await self._authenticated_request('POST', '/update_job_iteration', data=data)
        return 'success' in response
    
    async def update_job_sot_url(self, job_id: int, new_sot_url: str) -> bool:
        data = {
            'job_id': job_id,
            'new_sot_url': new_sot_url
        }
        response = await self._authenticated_request('POST', '/update_job_sot_url', data=data)
        return 'success' in response

    async def mark_job_as_done(self, job_id: int) -> bool:
        data = {
            'job_id': job_id
        }
        response = await self._authenticated_request('POST', '/mark_job_as_done', data=data)
        return 'success' in response
    
    async def get_jobs_without_instances(self) -> Optional[List[Job]]:
        response = await self._authenticated_request('GET', '/get_jobs_without_instances')
        if 'error' in response:
            logger.error(response['error'])
            return None
        return [self._deserialize(Job, job) for job in response]

    # --- PLUGINS ---
    async def get_plugin(self, plugin_id: int) -> Optional[Plugin]:
        return await self._fetch_entity('/get_plugin', Plugin, params={'plugin_id': plugin_id})

    async def create_plugin(self, name: str, code: str) -> Optional[int]:
        data = {
            'name': name,
            'code': code
        }
        response = await self._authenticated_request('POST', '/create_plugin', data=data)
        return self._extract_id(response, 'plugin_id')

    # --- SUBNETS ---
    async def get_subnet_using_address(self, address: str) -> Optional[Subnet]:
        return await self._fetch_entity('/get_subnet_using_address', Subnet, params={'address': address})

    async def get_subnet(self, subnet_id: int) -> Optional[Subnet]:
        return await self._fetch_entity('/get_subnet', Subnet, params={'subnet_id': subnet_id})

    async def create_subnet(self, dispute_period: int, solve_period: int, stake_multiplier: float) -> Optional[int]:
        data = {
            'dispute_period': dispute_period,
            'solve_period': solve_period,
            'stake_multiplier': stake_multiplier
        }
        response = await self._authenticated_request('POST', '/create_subnet', data=data)
        return self._extract_id(response, 'subnet_id')

    # --- TASKS ---
    async def get_task(self, task_id: int) -> Optional[Task]:
        return await self._fetch_entity('/get_task', Task, params={'task_id': task_id})
    
    async def get_assigned_tasks(self, subnet_id: int) -> Optional[int]:
        data = {
            'subnet_id': subnet_id
        }
        response = await self._authenticated_request(
            'GET', '/get_assigned_tasks', params=data,
            )
        return [self._deserialize(Task, task) for task in response.get('assigned_tasks')]
    
    async def get_num_orders(self, subnet_id: int, order_type: str) -> Optional[int]:
        response = await self._authenticated_request(
            'GET', '/get_num_orders', params={'subnet_id': subnet_id, 'order_type': order_type},
            )
        return response.get('num_orders')

    async def get_tasks_for_job(self, job_id: int, offset: int = 0, limit: int = 20) -> Optional[List[Task]]:
        params = {'job_id': job_id, 'offset': offset, 'limit': limit}
        response = await self._authenticated_request('GET', '/get_tasks_for_job', params=params)
        if 'error' in response:
            logger.error(response['error'])
            return None
        return [self._deserialize(Task, task) for task in response]

    async def get_task_count_for_job(self, job_id: int) -> Optional[int]:
        response = await self._authenticated_request('GET', '/get_task_count_for_job', params={'job_id': job_id})
        return response.get('task_count')

    async def get_task_count_by_status_for_job(self, job_id: int, statuses: List[str]) -> Optional[Dict[str, int]]:
        params = {'job_id': job_id, 'statuses': statuses}
        response = await self._authenticated_request('GET', '/get_task_count_by_status_for_job', params=params)
        return response

    async def get_last_task_with_status(self, job_id: int, statuses: List[str]) -> Optional[Task]:
        params = {'job_id': job_id, 'statuses': statuses}
        return await self._fetch_entity('/get_last_task_with_status', Task, params=params)

    async def create_task(self, job_id: int, job_iteration: int, status: str, params: str) -> Optional[int]:
        data = {
            'job_id': job_id,
            'job_iteration': job_iteration,
            'status': status,
            'params': params
        }
        response = await self._authenticated_request('POST', '/create_task', data=data)
        return self._extract_id(response, 'task_id')

    async def update_task_status(self, task_id: int, job_id: int, status: str, result=None, solver_address=None) -> bool:
        data = {
            'task_id': task_id,
            'job_id': job_id,
            'status': status,
            'result': result,
            'solver_address': solver_address
        }
        response = await self._authenticated_request('POST', '/update_task_status', data=data)
        return 'success' in response

    # --- ORDERS ---
    async def create_order(self, task_id: int, subnet_id: int, order_type: str, price: float) -> Optional[int]:
        data = {
            'task_id': task_id,
            'subnet_id': subnet_id,
            'order_type': order_type,
            'price': price
        }
        response = await self._authenticated_request('POST', '/create_order', data=data, )
        return self._extract_id(response, 'order_id')
    
    async def create_bids_and_tasks(self, job_id: int, num_tasks: int, price: float, params: str) -> Optional[List[Dict[str, int]]]:
        data = {
            'job_id': job_id,
            'num_tasks': num_tasks,
            'price': price,
            'params': params
        }
        response = await self._authenticated_request('POST', '/create_bids_and_tasks', data=data)
        if 'error' in response:
            logger.error(f"Error creating bids and tasks: {response['error']}")
            return None
        return response.get('created_items')


    async def delete_order(self, order_id: int) -> bool:
        data = {
            'order_id': order_id
        }
        response = await self._authenticated_request('POST', '/delete_order', data=data)
        return 'success' in response

    # --- ACCOUNTS ---
    async def create_account_key(self) -> Optional[int]:
        response = await self._authenticated_request('POST', '/create_account_key')
        return response

    async def admin_create_account_key(self, user_id: str) -> Optional[int]:
        data = {
            'user_id': user_id
        }
        return await self._authenticated_request('POST', '/admin_create_account_key', data=data)

    
    async def account_key_from_public_key(self, public_key: str) -> Optional[Dict[str, Any]]:
        params = {
            'public_key': public_key
        }
        response = await self._authenticated_request('GET', '/account_key_from_public_key', params=params)
        return response

    async def get_account_keys(self) -> Optional[List[Dict[str, Any]]]:
        response = await self._authenticated_request('GET', '/get_account_keys')
        return response

    async def delete_account_key(self, account_key_id: int) -> bool:
        data = {
            'account_key_id': account_key_id
        }
        response = await self._authenticated_request('POST', '/delete_account_key', data=data)
        return 'success' in response

    # --- STAKES ---
    async def deposit_account(self,  amount: float) -> bool:
        data = {
            'amount': amount
        }
        response = await self._authenticated_request('POST', '/deposit_account', data=data, )
        return 'success' in response

    async def withdraw_account(self, amount: float) -> bool:
        data = {
            'amount': amount
        }
        response = await self._authenticated_request('POST', '/withdraw_account', data=data, )
        return 'success' in response


    # --- PERMISSIONS ---
    async def get_perm(self, address: str, perm: int) -> Optional[Perm]:
        params = {
            'address': address,
            'perm': perm
        }
        return await self._fetch_entity('/get_perm', Perm, params=params)

    async def create_perm(self, address: str, perm: int) -> Optional[int]:
        data = {
            'address': address,
            'perm': perm
        }
        response = await self._authenticated_request('POST', '/create_perm', data=data)
        return self._extract_id(response, 'perm_id')

    async def create_perm_description(self, perm_type: str) -> Optional[int]:
        data = {
            'perm_type': perm_type,
        }
        response = await self._authenticated_request('POST', '/create_perm_description', data=data)
        return self._extract_id(response, 'perm_description_id')

    async def set_last_nonce(self, address: str, perm: int, last_nonce: str) -> bool:
        data = {
            'address': address,
            'perm': perm,
            'last_nonce': last_nonce
        }
        response = await self._authenticated_request('POST', '/set_last_nonce', data=data)
        return 'success' in response

    # --- SOTS ---
    async def get_sot(self, sot_id: int) -> Optional[Sot]:
        return await self._fetch_entity('/get_sot', Sot, params={'id': sot_id})

    async def get_sot_by_job_id(self, job_id: int) -> Optional[Sot]:
        return await self._fetch_entity('/get_sot_by_job_id', Sot, params={'job_id': job_id})

    async def create_sot(self, job_id: int, url: str) -> Optional[int]:
        data = {
            'job_id': job_id,
            'url': url
        }
        response = await self._authenticated_request('POST', '/create_sot', data=data)
        return self._extract_id(response, 'sot_id')

    async def update_sot(self, sot_id: int, url: str) -> bool:
        data = {
            'sot_id': sot_id,
            'url': url
        }
        response = await self._authenticated_request('POST', '/update_sot', data=data)
        return 'success' in response
    # --- INSTANCES ---
    async def get_instance_by_service_type(self, service_type: str, job_id: int) -> Optional[Instance]:
        params = {'service_type': service_type, 'job_id': job_id}
        return await self._fetch_entity('/get_instance_by_service_type', Instance, params=params)

    async def get_instances_by_job(self, job_id: int) -> Optional[List[Instance]]:
        response = await self._authenticated_request('GET', '/get_instances_by_job', params={'job_id': job_id})
        if 'error' in response:
            logger.error(response['error'])
            return None
        return [self._deserialize(Instance, instance) for instance in response]
    
    async def get_all_instances(self) -> Optional[List[Instance]]:
        response = await self._authenticated_request('GET', '/get_all_instances')
        if 'error' in response:
            logger.error(response['error'])
            return None
        return [self._deserialize(Instance, instance) for instance in response]

    async def create_instance(self, name: str, service_type: str, job_id: int, private_key: str, pod_id: str, process_id: str) -> Optional[int]:
        data = {
            'name': name,
            'service_type': service_type,
            'job_id': job_id,
            'private_key': private_key,
            'pod_id': pod_id,
            'process_id': process_id
        }
        response = await self._authenticated_request('POST', '/create_instance', data=data)
        return self._extract_id(response, 'instance_id')

    async def update_instance(self, instance_id: int, **kwargs) -> bool:
        data = {
            'instance_id': instance_id,
            **kwargs
        }
        response = await self._authenticated_request('POST', '/update_instance', data=data)
        return 'success' in response

    # --- STATE UPDATES ---
    async def get_total_state_updates_for_job(self, job_id: int) -> Optional[int]:
        response = await self._authenticated_request('GET', '/get_total_state_updates_for_job', params={'job_id': job_id})
        return response.get('total_state_updates')

    async def create_state_update(self, job_id: int, data: str) -> Optional[int]:
        payload = {
            'job_id': job_id,
            'data': data
        }
        response = await self._authenticated_request('POST', '/create_state_update', data=payload)
        return self._extract_id(response, 'state_update_id')

    # --- HELPERS ---
    async def _fetch_entity(self, endpoint: str, model_cls: Type[T], data=None, params=None) -> Optional[T]:
        response = await self._authenticated_request('GET', endpoint, data=data, params=params)
        if 'error' in response:
            logger.error(response['error'])
            return None
        return self._deserialize(model_cls, response.get('data', response))

    def _deserialize(self, model_cls: Type[T], data: Dict[str, Any]) -> T:
        field_names = {column.name for column in model_cls.__table__.columns}
        model_data = {k: v for k, v in data.items() if k in field_names}
        
        for column in model_cls.__table__.columns:
            if column.type.python_type == datetime and column.name in model_data:
                model_data[column.name] = self._parse_datetime(model_data[column.name])

        return model_cls(**model_data)

    def _parse_datetime(self, date_str: Optional[str]) -> Optional[datetime]:
        if not date_str:
            return None
        try:
            # parse the exact string format with timezone
            dt = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %Z')
            return dt.replace(tzinfo=timezone.utc)  # ensure it's utc
        except ValueError:
            logger.error(f"failed to parse datetime string: {date_str}")
            return None
