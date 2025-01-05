# spl/db/db_adapter_client.py

import logging
import aiohttp
import json
import time
import uuid
from eth_account.messages import encode_defunct
from eth_account import Account
from typing import Optional, List, Type, TypeVar, Dict, Any
from datetime import datetime, timezone
from typeguard import typechecked

from ..models import (
    Job, Plugin, Subnet, Task, TaskStatus, Perm, Sot, Instance, ServiceType, Base, PermType,
    PendingWithdrawal
)

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Base)

class DBAdapterClient:
    @typechecked
    def __init__(self, base_url: str, private_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.private_key = private_key

    def _generate_message(self, endpoint: str, data: Optional[dict] = None) -> str:
        message = {
            "endpoint": endpoint,
            "nonce": str(uuid.uuid4()),
            "timestamp": int(time.time()),
            "data": data or {}
        }
        return json.dumps(message, sort_keys=True)

    def _sign_message(self, message: str) -> Optional[str]:
        if not self.private_key:
            return None
        message_defunct = encode_defunct(text=message)
        account = Account.from_key(self.private_key)
        signed_message = account.sign_message(message_defunct)
        return signed_message.signature.hex()

    async def _authenticated_request(self, method: str, endpoint: str, data: Optional[dict] = None, params: Optional[dict] = None) -> Dict[str, Any]:
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
                    if not isinstance(json_response, (dict, list)):
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

    @typechecked
    async def get_job(self, job_id: int) -> Optional[Job]:
        return await self._fetch_entity('/get_job', Job, params={'job_id': job_id})

    @typechecked
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

    @typechecked
    async def update_job_iteration(self, job_id: int, new_iteration: int) -> bool:
        data = {
            'job_id': job_id,
            'new_iteration': new_iteration
        }
        response = await self._authenticated_request('POST', '/update_job_iteration', data=data)
        return 'success' in response

    @typechecked
    async def update_job_sot_url(self, job_id: int, new_sot_url: str) -> bool:
        data = {
            'job_id': job_id,
            'new_sot_url': new_sot_url
        }
        response = await self._authenticated_request('POST', '/update_job_sot_url', data=data)
        return 'success' in response

    @typechecked
    async def mark_job_as_done(self, job_id: int) -> bool:
        data = {
            'job_id': job_id
        }
        response = await self._authenticated_request('POST', '/mark_job_as_done', data=data)
        return 'success' in response

    @typechecked
    async def get_jobs_without_instances(self) -> Optional[List[Job]]:
        response = await self._authenticated_request('GET', '/get_jobs_without_instances')
        if 'error' in response:
            logger.error(response['error'])
            return None
        return [self._deserialize(Job, job) for job in response]
    
    @typechecked
    async def get_jobs_in_progress(self) -> Optional[List[Job]]:
        response = await self._authenticated_request('GET', '/get_jobs_in_progress')
        if 'error' in response:
            logger.error(response['error'])
            return None
        # each item is a job dict => deserialize
        return [self._deserialize(Job, job_dict) for job_dict in response]

    @typechecked
    async def get_plugin(self, plugin_id: int) -> Optional[Plugin]:
        return await self._fetch_entity('/get_plugin', Plugin, params={'plugin_id': plugin_id})

    @typechecked
    async def create_plugin(self, name: str, code: str) -> Optional[int]:
        data = {
            'name': name,
            'code': code
        }
        response = await self._authenticated_request('POST', '/create_plugin', data=data)
        return self._extract_id(response, 'plugin_id')

    @typechecked
    async def get_subnet_using_address(self, address: str) -> Optional[Subnet]:
        return await self._fetch_entity('/get_subnet_using_address', Subnet, params={'address': address})

    @typechecked
    async def get_subnet(self, subnet_id: int) -> Optional[Subnet]:
        return await self._fetch_entity('/get_subnet', Subnet, params={'subnet_id': subnet_id})

    @typechecked
    async def create_subnet(self, dispute_period: int, solve_period: int, stake_multiplier: float) -> Optional[int]:
        data = {
            'dispute_period': dispute_period,
            'solve_period': solve_period,
            'stake_multiplier': stake_multiplier
        }
        response = await self._authenticated_request('POST', '/create_subnet', data=data)
        return self._extract_id(response, 'subnet_id')

    @typechecked
    async def get_task(self, task_id: int) -> Optional[Task]:
        return await self._fetch_entity('/get_task', Task, params={'task_id': task_id})

    @typechecked
    async def get_assigned_tasks(self, subnet_id: int) -> Optional[List[Task]]:
        data = {
            'subnet_id': subnet_id
        }
        response = await self._authenticated_request('GET', '/get_assigned_tasks', params=data)
        if 'assigned_tasks' not in response:
            return None
        return [self._deserialize(Task, task) for task in response.get('assigned_tasks')]

    @typechecked
    async def get_num_orders(self, subnet_id: int, order_type: str, matched: Optional[bool]) -> Optional[int]:
        params = {
            'subnet_id': subnet_id,
            'order_type': order_type
        }
        if matched is not None:
            params['matched'] = str(matched).lower()

        response = await self._authenticated_request('GET', '/get_num_orders', params=params)
        return response.get('num_orders')

    @typechecked
    async def get_tasks_for_job(self, job_id: int, offset: int = 0, limit: int = 20) -> Optional[List[Task]]:
        params = {'job_id': job_id, 'offset': offset, 'limit': limit}
        response = await self._authenticated_request('GET', '/get_tasks_for_job', params=params)
        if 'error' in response:
            logger.error(response['error'])
            return None
        return [self._deserialize(Task, task) for task in response]

    @typechecked
    async def get_task_count_for_job(self, job_id: int) -> Optional[int]:
        response = await self._authenticated_request('GET', '/get_task_count_for_job', params={'job_id': job_id})
        return response.get('task_count')

    @typechecked
    async def get_task_count_by_status_for_job(self, job_id: int, statuses: List[str]) -> Optional[Dict[str, int]]:
        params = {'job_id': job_id, 'statuses': statuses}
        response = await self._authenticated_request('GET', '/get_task_count_by_status_for_job', params=params)
        return response if isinstance(response, dict) else None

    @typechecked
    async def get_last_task_with_status(self, job_id: int, statuses: List[str]) -> Optional[Task]:
        params = {'job_id': job_id, 'statuses': statuses}
        return await self._fetch_entity('/get_last_task_with_status', Task, params=params)

    @typechecked
    async def create_task(self, job_id: int, job_iteration: int, status: str, params: str) -> Optional[int]:
        data = {
            'job_id': job_id,
            'job_iteration': job_iteration,
            'status': status,
            'params': params
        }
        response = await self._authenticated_request('POST', '/create_task', data=data)
        return self._extract_id(response, 'task_id')

    @typechecked
    async def update_task_status(self, task_id: int, job_id: int, status: str, result: Optional[str] = None, solver_address: Optional[str] = None) -> bool:
        data = {
            'task_id': task_id,
            'job_id': job_id,
            'status': status,
            'result': result,
            'solver_address': solver_address
        }
        response = await self._authenticated_request('POST', '/update_task_status', data=data)
        return 'success' in response

    @typechecked
    async def submit_task_result(self, task_id: int, result: str) -> bool:
        data = {
            'task_id': task_id,
            'result': result
        }
        response = await self._authenticated_request('POST', '/submit_task_result', data=data)
        return response.get('success', False)

    @typechecked
    async def create_order(self, task_id: int | None, subnet_id: int, order_type: str, price: int, hold_id: Optional[int]) -> Optional[int]:
        data = {
            'task_id': task_id,
            'subnet_id': subnet_id,
            'order_type': order_type,
            'price': price,
            'hold_id': hold_id
        }
        response = await self._authenticated_request('POST', '/create_order', data=data)
        return self._extract_id(response, 'order_id')

    @typechecked
    async def create_bids_and_tasks(self, job_id: int, num_tasks: int, price: int, params: str, hold_id: Optional[int]) -> Optional[List[Dict[str, int]]]:
        data = {
            'job_id': job_id,
            'num_tasks': num_tasks,
            'price': price,
            'params': params,
            'hold_id': hold_id
        }
        response = await self._authenticated_request('POST', '/create_bids_and_tasks', data=data)
        if 'error' in response:
            logger.error(f"Error creating bids and tasks: {response['error']}")
            return None
        return response.get('created_items')

    @typechecked
    async def delete_order(self, order_id: int) -> bool:
        data = {
            'order_id': order_id
        }
        response = await self._authenticated_request('POST', '/delete_order', data=data)
        return 'success' in response

    @typechecked
    async def create_account_key(self) -> Optional[int]:
        response = await self._authenticated_request('POST', '/create_account_key')
        if isinstance(response, dict) and 'account_key_id' in response and isinstance(response['account_key_id'], int):
            return response['account_key_id']
        return None

    @typechecked
    async def admin_create_account_key(self, user_id: str) -> Optional[Dict[str, Any]]:
        data = {
            'user_id': user_id
        }
        response = await self._authenticated_request('POST', '/admin_create_account_key', data=data)
        if 'error' in response:
            return None
        return response

    @typechecked
    async def account_key_from_public_key(self, public_key: str) -> Optional[Dict[str, Any]]:
        params = {
            'public_key': public_key
        }
        response = await self._authenticated_request('GET', '/account_key_from_public_key', params=params)
        return response if isinstance(response, dict) else None

    @typechecked
    async def get_account_keys(self) -> Optional[List[Dict[str, Any]]]:
        response = await self._authenticated_request('GET', '/get_account_keys')
        return response if isinstance(response, list) else None

    @typechecked
    async def delete_account_key(self, account_key_id: int) -> bool:
        data = {
            'account_key_id': account_key_id
        }
        response = await self._authenticated_request('POST', '/delete_account_key', data=data)
        return 'success' in response

    @typechecked
    async def admin_deposit_account(self, user_id: str, amount: int) -> bool:
        data = {
            'user_id': user_id,
            'amount': amount
        }
        response = await self._authenticated_request('POST', '/admin_deposit_account', data=data)
        return 'success' in response

    @typechecked
    async def get_perm(self, address: str, perm: int) -> Optional[Perm]:
        params = {
            'address': address,
            'perm': perm
        }
        return await self._fetch_entity('/get_perm', Perm, params=params)

    @typechecked
    async def create_perm(self, address: str, perm: int) -> Optional[int]:
        data = {
            'address': address,
            'perm': perm
        }
        response = await self._authenticated_request('POST', '/create_perm', data=data)
        return self._extract_id(response, 'perm_id')

    @typechecked
    async def create_perm_description(self, perm_type: str) -> Optional[int]:
        data = {
            'perm_type': perm_type,
        }
        response = await self._authenticated_request('POST', '/create_perm_description', data=data)
        return self._extract_id(response, 'perm_description_id')

    @typechecked
    async def set_last_nonce(self, address: str, perm: int, last_nonce: str) -> bool:
        data = {
            'address': address,
            'perm': perm,
            'last_nonce': last_nonce
        }
        response = await self._authenticated_request('POST', '/set_last_nonce', data=data)
        return isinstance(response, dict) and 'perm' in response

    @typechecked
    async def get_sot(self, sot_id: int) -> Optional[Sot]:
        return await self._fetch_entity('/get_sot', Sot, params={'id': sot_id})

    @typechecked
    async def get_sot_by_job_id(self, job_id: int) -> Optional[Sot]:
        return await self._fetch_entity('/get_sot_by_job_id', Sot, params={'job_id': job_id})

    @typechecked
    async def create_sot(self, job_id: int, url: str | None) -> Optional[int]:
        data = {
            'job_id': job_id,
            'url': url
        }
        response = await self._authenticated_request('POST', '/create_sot', data=data)
        return self._extract_id(response, 'sot_id')

    @typechecked
    async def update_sot(self, sot_id: int, url: str | None) -> bool:
        data = {
            'sot_id': sot_id,
            'url': url
        }
        response = await self._authenticated_request('POST', '/update_sot', data=data)
        return 'success' in response

    @typechecked
    async def get_instance_by_service_type(self, service_type: str, job_id: int) -> Optional[Instance]:
        params = {'service_type': service_type, 'job_id': job_id}
        return await self._fetch_entity('/get_instance_by_service_type', Instance, params=params)

    @typechecked
    async def get_instances_by_job(self, job_id: int) -> Optional[List[Instance]]:
        response = await self._authenticated_request('GET', '/get_instances_by_job', params={'job_id': job_id})
        if 'error' in response:
            logger.error(response['error'])
            return None
        return [self._deserialize(Instance, instance) for instance in response]

    @typechecked
    async def get_all_instances(self) -> Optional[List[Instance]]:
        response = await self._authenticated_request('GET', '/get_all_instances')
        if 'error' in response:
            logger.error(response['error'])
            return None
        return [self._deserialize(Instance, instance) for instance in response]

    @typechecked
    async def create_instance(self, name: str, service_type: str, job_id: int | None, private_key: str | None, pod_id: str | None, process_id: int | None) -> Optional[int]:
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

    @typechecked
    async def update_instance(self, instance_id: int, **kwargs: Any) -> bool:
        data = {
            'instance_id': instance_id,
            **kwargs
        }
        response = await self._authenticated_request('POST', '/update_instance', data=data)
        return 'success' in response

    @typechecked
    async def get_total_state_updates_for_job(self, job_id: int) -> Optional[int]:
        response = await self._authenticated_request('GET', '/get_total_state_updates_for_job', params={'job_id': job_id})
        return response.get('total_state_updates')

    @typechecked
    async def create_state_update(self, job_id: int, data: Dict[str, Any]) -> Optional[int]:
        payload = {
            'job_id': job_id,
            'data': data
        }
        response = await self._authenticated_request('POST', '/create_state_update', data=payload)
        return self._extract_id(response, 'state_update_id')

    @typechecked
    async def get_state_for_job(self, job_id: int) -> dict:
        """
        Return the 'state_json' for this job from the DB.
        """
        logger.debug(f"[DBAdapterClient] get_state_for_job called with job_id={job_id}")
        params = {'job_id': str(job_id)}
        response = await self._authenticated_request('GET', '/get_job_state', params=params)
        logger.debug(f"[DBAdapterClient] get_state_for_job => raw server response: {response}")
        if 'error' in response:
            logger.error(f"[DBAdapterClient] get_state_for_job => error: {response['error']}")
            return {}
        if isinstance(response, dict):
            logger.debug(f"[DBAdapterClient] get_state_for_job => final: {response}")
            return response
        return {}

    @typechecked
    async def update_state_for_job(self, job_id: int, new_state_data: dict) -> bool:
        """
        Overwrite the job's state_json with new_state_data
        """
        logger.debug(f"[DBAdapterClient] update_state_for_job called with job_id={job_id}, new_state_data={new_state_data}")
        data = {
            'job_id': job_id,
            'new_state': new_state_data
        }
        response = await self._authenticated_request('POST', '/update_job_state', data=data)
        logger.debug(f"[DBAdapterClient] update_state_for_job => server response: {response}")
        if 'error' in response:
            logger.error(f"[DBAdapterClient] update_state_for_job => error: {response['error']}")
            return False
        return True

    # Internal helper for retrieving single entity
    async def _fetch_entity(self, endpoint: str, model_cls: Type[T], data: Optional[dict] = None, params: Optional[dict] = None) -> Optional[T]:
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
            from datetime import datetime
            dt = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %Z')
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            logger.error(f"failed to parse datetime string: {date_str}")
            return None

    ##
    # NEW: Withdrawals
    ##
    @typechecked
    async def create_withdrawal(self, user_id: str, amount: int) -> Optional[int]:
        data = {
            'user_id': user_id,
            'amount': amount
        }
        resp = await self._authenticated_request('POST', '/create_withdrawal', data=data)
        return self._extract_id(resp, 'withdrawal_id')

    @typechecked
    async def get_withdrawal(self, withdrawal_id: int) -> Optional[PendingWithdrawal]:
        return await self._fetch_entity('/get_withdrawal', PendingWithdrawal, params={'withdrawal_id': withdrawal_id})

    @typechecked
    async def get_withdrawals_for_user(self, user_id: str) -> List[PendingWithdrawal]:
        response = await self._authenticated_request('GET', '/get_withdrawals_for_user', params={'user_id': user_id})
        if 'error' in response:
            logger.error(response['error'])
            return []
        # parse list:
        return [self._deserialize(PendingWithdrawal, item) for item in response]

    @typechecked
    async def update_withdrawal_status(self, withdrawal_id: int, new_status: str) -> bool:
        data = {
            'withdrawal_id': withdrawal_id,
            'new_status': new_status
        }
        resp = await self._authenticated_request('POST', '/update_withdrawal_status', data=data)
        return 'success' in resp
