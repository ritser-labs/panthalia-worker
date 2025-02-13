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
    Job, Plugin, Subnet, Task, SlotType, Perm, Sot, Instance, Order, Base, PermType,
    WithdrawalRequest
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

    async def _authenticated_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[dict] = None, 
        params: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Sends an HTTP request with optional JSON `data` and `params`, adding a
        signature header if self.private_key is set. If the server returns an
        error status (>=400), logs status + body, and returns them in a dict.
        """
        url = f"{self.base_url}{endpoint}"
        headers = {}

        # If we have a private key, generate & sign a message
        if self.private_key:
            message = self._generate_message(endpoint, data)
            signature = self._sign_message(message)
            if signature:
                headers["Authorization"] = f"{message}:{signature}"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.request(
                    method, url, json=data, params=params, headers=headers
                ) as response:
                    # Check for HTTP error codes explicitly
                    if response.status >= 400:
                        body_text = await response.text()
                        logger.error(
                            "Request to %s failed: status=%s, body=%s",
                            url, response.status, body_text
                        )
                        return {
                            "error": f"HTTP {response.status}",
                            "details": body_text
                        }

                    # Otherwise parse as JSON
                    json_response = await response.json()
                    if not isinstance(json_response, (dict, list)):
                        logger.error(
                            "Unexpected response format from %s: %s",
                            url, json_response
                        )
                        return {"error": "Unexpected response format"}
                    return json_response

            except aiohttp.ClientError as e:
                # This catches network issues, DNS failures, timeouts, etc.
                logger.error("Request to %s failed: %s", url, e)
                return {"error": str(e)}


    def _extract_id(self, response: Dict[str, Any], id_key: str) -> Optional[int]:
        if 'error' in response:
            return None
        return response.get(id_key) or response.get('data', {}).get(id_key)

    @typechecked
    async def get_job(self, job_id: int) -> Optional[Job]:
        return await self._fetch_entity('/get_job', Job, params={'job_id': job_id})

    @typechecked
    async def create_job(self, name: str, plugin_id: int, sot_url: str, iteration: int, initial_state_url: str='') -> Optional[int]:
        data = {
            'name': name,
            'plugin_id': plugin_id,
            'sot_url': sot_url,
            'iteration': iteration,
            'initial_state_url': initial_state_url
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
        return [self._deserialize(Job, job_dict) for job_dict in response]

    @typechecked
    async def get_plugin(self, plugin_id: int) -> Optional[Plugin]:
        return await self._fetch_entity('/get_plugin', Plugin, params={'plugin_id': plugin_id})

    @typechecked
    async def create_plugin(self, name: str, code: str, subnet_id: int) -> Optional[int]:
        data = {
            'name': name,
            'code': code,
            'subnet_id': subnet_id
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
    async def create_subnet(self, dispute_period: int, solve_period: int, stake_multiplier: float, target_price: int=1, description: str | None = '') -> Optional[int]:
        data = {
            'dispute_period': dispute_period,
            'solve_period': solve_period,
            'stake_multiplier': stake_multiplier,
            'target_price': target_price,
            'description': description
        }
        response = await self._authenticated_request('POST', '/create_subnet', data=data)
        return self._extract_id(response, 'subnet_id')

    @typechecked
    async def set_subnet_target_price(self, subnet_id: int, target_price: int) -> bool:
        data = {
            'subnet_id': subnet_id,
            'target_price': target_price
        }
        response = await self._authenticated_request('POST', '/set_subnet_target_price', data=data)
        return 'success' in response

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
    async def get_unmatched_orders_for_job(self, job_id: int) -> list[Order]:
        """
        Calls GET /get_unmatched_orders_for_job?job_id=XYZ
        and returns a list of Order objects that are unmatched
        for the given job_id.
        """
        endpoint = "/get_unmatched_orders_for_job"
        params = {"job_id": str(job_id)}
        
        response = await self._authenticated_request("GET", endpoint, params=params)
        if "error" in response:
            logging.error(f"Error fetching unmatched orders for job {job_id}: {response['error']}")
            return []
        
        # Expecting a JSON array of orders
        orders_list = []
        for order_dict in response:
            deserialized = self._deserialize(Order, order_dict)
            orders_list.append(deserialized)
        return orders_list

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
    async def update_task_status(self, task_id: int, job_id: int, status: str) -> bool:
        data = {
            'task_id': task_id,
            'job_id': job_id,
            'status': status,
        }
        response = await self._authenticated_request('POST', '/update_task_status', data=data)
        return 'success' in response

    @typechecked
    async def submit_partial_result(self, task_id: int, partial_result: str, final: bool = False) -> bool:
        data = {
            'task_id': task_id,
            'partial_result': partial_result,
            'final': final
        }
        response = await self._authenticated_request('POST', '/submit_partial_result', data=data)
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
    async def get_account_keys(self, offset: int, limit: int) -> Optional[List[Dict[str, Any]]]:
        params = {
            'offset': offset,
            'limit': limit
        }
        response = await self._authenticated_request('GET', '/get_account_keys', params=params)
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
    async def create_perm_description(self, perm_type: str, restricted_sot_id: Optional[int] = None) -> Optional[int]:
        data = {
            'perm_type': perm_type,
            'restricted_sot_id': restricted_sot_id
        }
        response = await self._authenticated_request('POST', '/create_perm_description', data=data)
        return self._extract_id(response, 'perm_description_id')

    @typechecked
    async def get_sot_by_job_id(self, job_id: int) -> Optional[Sot]:
        return await self._fetch_entity('/get_sot_by_job_id', Sot, params={'job_id': job_id})

    @typechecked
    async def create_sot(self, job_id: int, address: str | None, url: str | None) -> Optional[int]:
        data = {
            'job_id': job_id,
            'address': address,
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
    async def get_instance(self, instance_id: int) -> Optional[Instance]:
        return await self._fetch_entity('/get_instance', Instance, params={'instance_id': instance_id})

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
    async def update_instance(self, update_data: dict) -> bool:
        # e.g. ensure at least "instance_id" is present
        if "instance_id" not in update_data:
            raise ValueError("Missing instance_id")

        response = await self._authenticated_request(
            "POST",
            "/update_instance",
            data=update_data
        )
        return "success" in response

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
    async def get_master_state_for_job(self, job_id: int) -> dict:
        """
        Fetch the master_state_json from /get_master_job_state
        """
        params = {'job_id': str(job_id)}
        response = await self._authenticated_request('GET', '/get_master_job_state', params=params)
        if 'error' in response:
            logger.error(f"get_master_state_for_job => error: {response['error']}")
            return {}
        if isinstance(response, dict):
            return response
        return {}

    @typechecked
    async def update_master_state_for_job(self, job_id: int, new_state: dict) -> bool:
        """
        POST to /update_master_job_state with job_id and new_state
        """
        data = {
            'job_id': job_id,
            'new_state': new_state
        }
        response = await self._authenticated_request('POST', '/update_master_job_state', data=data)
        if 'error' in response:
            logger.error(f"update_master_state_for_job => error: {response['error']}")
            return False
        return True

    @typechecked
    async def get_sot_state_for_job(self, job_id: int) -> dict:
        """
        Fetch the sot_state_json from /get_sot_job_state
        """
        params = {'job_id': str(job_id)}
        response = await self._authenticated_request('GET', '/sot/get_job_state', params=params)
        if 'error' in response:
            logger.error(f"get_sot_state_for_job => error: {response['error']}")
            return {}
        if isinstance(response, dict):
            return response
        return {}

    @typechecked
    async def update_sot_state_for_job(self, job_id: int, new_state: dict) -> bool:
        """
        POST to /update_sot_job_state with job_id and new_state
        """
        data = {
            'job_id': job_id,
            'new_state': new_state
        }
        response = await self._authenticated_request('POST', '/sot/update_job_state', data=data)
        if 'error' in response:
            logger.error(f"update_sot_state_for_job => error: {response['error']}")
            return False
        return True
    
    @typechecked
    async def sot_get_job(self, job_id: int) -> Optional[Job]:
        return await self._fetch_entity('/sot/get_job', Job, params={'job_id': job_id})
    
    @typechecked
    async def sot_get_sot(self, sot_id: int) -> Optional[Sot]:
        return await self._fetch_entity('/sot/get_sot', Sot, params={'sot_id': sot_id})

    @typechecked
    async def update_job_active(self, job_id: int, active: bool) -> bool:
        data = {
            'job_id': job_id,
            'active': active
        }
        response = await self._authenticated_request('POST', '/update_job_active', data=data)
        return 'success' in response

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
    async def create_withdrawal(self, amount: int) -> Optional[int]:
        data = {
            'amount': amount
        }
        resp = await self._authenticated_request('POST', '/create_withdrawal', data=data)
        return self._extract_id(resp, 'withdrawal_id')

    @typechecked
    async def get_withdrawal(self, withdrawal_id: int) -> Optional[WithdrawalRequest]:
        return await self._fetch_entity('/get_withdrawal', WithdrawalRequest, params={'withdrawal_id': withdrawal_id})

    @typechecked
    async def get_withdrawals_for_user(self) -> List[WithdrawalRequest]:
        response = await self._authenticated_request('GET', '/get_withdrawals_for_user')
        if 'error' in response:
            logger.error(response['error'])
            return []
        # parse list:
        return [self._deserialize(WithdrawalRequest, item) for item in response]

    # job queue management

    @typechecked
    async def update_job_queue_status(
        self,
        job_id: int,
        new_queued: bool,
        assigned_master_id: Optional[str] = None
    ) -> bool:
        """
        POST /update_job_queue_status

        ★ FIXED: We post 'new_queued' instead of 'queued', to match the route's required_keys. ★
        """
        data = {
            'job_id': job_id,
            'new_queued': new_queued,           # <-- changed here from 'queued': new_queued
            'assigned_master_id': assigned_master_id
        }
        response = await self._authenticated_request('POST', '/update_job_queue_status', data=data)
        return 'success' in response

    @typechecked
    async def get_unassigned_queued_jobs(self) -> Optional[List[Job]]:
        response = await self._authenticated_request('GET', '/get_unassigned_queued_jobs')
        if 'error' in response:
            logger.error(response['error'])
            return None
        if isinstance(response, list):
            return [self._deserialize(Job, job_dict) for job_dict in response]
        else:
            return []

    @typechecked
    async def get_jobs_assigned_to_master(self, master_id: str) -> Optional[List[Job]]:
        params = {'master_id': master_id}
        response = await self._authenticated_request('GET', '/get_jobs_assigned_to_master', params=params)
        if 'error' in response:
            logger.error(response['error'])
            return None
        if isinstance(response, list):
            return [self._deserialize(Job, job_dict) for job_dict in response]
        else:
            return []
    
    @typechecked
    async def get_unassigned_unqueued_active_jobs(self) -> Optional[List[Job]]:
        response = await self._authenticated_request('GET', '/get_unassigned_unqueued_active_jobs')
        if 'error' in response:
            logger.error(response['error'])
            return None
        if isinstance(response, list):
            return [self._deserialize(Job, job_dict) for job_dict in response]
        return []

    @typechecked
    async def get_free_instances_by_slot_type(self, slot_type: SlotType) -> List[dict]:
        """
        Calls GET /get_free_instances_by_slot_type?slot_type=XYZ,
        returns list of instance dicts (with .as_dict fields).
        """
        endpoint = "/get_free_instances_by_slot_type"
        params = {"slot_type": slot_type.name}
        response = await self._authenticated_request("GET", endpoint, params=params)
        if "error" in response:
            logger.error(f"get_free_instances_by_slot_type => {response['error']}")
            return []
        if isinstance(response, list):
            return response
        return []

    @typechecked
    async def reserve_instance(self, instance_id: int, job_id: int) -> bool:
        """
        Calls POST /reserve_instance with JSON {instance_id, job_id}.
        Returns True if success, else False.
        """
        endpoint = "/reserve_instance"
        data = {"instance_id": instance_id, "job_id": job_id}
        response = await self._authenticated_request("POST", endpoint, data=data)
        if "success" in response:
            return True
        return False

    @typechecked
    async def finalize_sanity_check(self, task_id: int, is_valid: bool) -> bool:
        """
        Finalizes the sanity check for a given task by POSTing to /finalize_sanity_check,
        marking the task as ResolvedCorrect or ResolvedIncorrect on the server.

        Returns:
            bool: True if the operation succeeded, False otherwise.
        """
        data = {
            "task_id": task_id,
            "is_valid": is_valid
        }
        response = await self._authenticated_request(
            method="POST",
            endpoint="/finalize_sanity_check",
            data=data
        )
        return response.get("success", False)


    @typechecked
    async def update_replicated_parent(self, child_task_id: int, parent_task_id: int) -> bool:
        """
        Sets the `replicated_parent_id` on a child task to point back to its parent task.
        Useful for forming a replication chain to track indefinite replicate tasks.

        Returns:
            bool: True if the update succeeded, False otherwise.
        """
        data = {
            "child_task_id": child_task_id,
            "parent_task_id": parent_task_id
        }
        response = await self._authenticated_request(
            method="POST",
            endpoint="/update_replicated_parent",
            data=data
        )
        return response.get("success", False)
