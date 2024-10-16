# db_adapter_client.py

import aiohttp
import asyncio
import logging
import json
from eth_account.messages import encode_defunct
from eth_account import Account
import uuid
import time
from ..models import Sot, Job, Task, Subnet, Plugin, StateUpdate, Perm, PermDescription
from typing import Dict

logger = logging.getLogger(__name__)

class DBAdapterClient:
    def __init__(self, server_url, private_key=None):
        self.server_url = server_url.rstrip('/')
        self.private_key = private_key

    def generate_message(self, endpoint, data=None):
        message = {
            "endpoint": endpoint,
            "nonce": str(uuid.uuid4()),
            "timestamp": int(time.time()),
            "data": data or {}
        }
        return json.dumps(message, sort_keys=True)

    def sign_message(self, message):
        message_defunct = encode_defunct(text=message)
        account = Account.from_key(self.private_key)
        signed_message = account.sign_message(message_defunct)
        return signed_message.signature.hex()

    async def authenticated_request(self, method, endpoint, data=None, params=None):
        url = f"{self.server_url}{endpoint}"
        headers = {}
        if method.upper() != 'GET':
            try:
                message = self.generate_message(endpoint, data)
                signature = self.sign_message(message)
                headers = {"Authorization": f"{message}:{signature}"}
            except TypeError as te:
                logger.error(f"Error generating authentication headers: {te}")
                return {'error': 'Authentication header generation failed', 'details': str(te)}

        async with aiohttp.ClientSession() as session:
            try:
                async with session.request(method, url, json=data, params=params, headers=headers) as response:
                    try:
                        response_json = await response.json()
                    except aiohttp.ContentTypeError:
                        response_text = await response.text()
                        logger.error(f"Non-JSON response from {url}: {response_text}")
                        return {'error': 'Invalid response format', 'response': response_text}

                    if response.status in (200, 201):
                        return response_json
                    else:
                        logger.error(f"Request to {url} failed with status {response.status}: {response_json}")
                        return {'error': response_json.get('error', 'Unknown error')}

            except aiohttp.ClientResponseError as e:
                logger.error(f"Response error while sending request to {url}: {e}")
                return {'error': str(e)}
            except aiohttp.ClientError as e:
                logger.error(f"Client error while sending request to {url}: {e}")
                return {'error': str(e)}
            except Exception as e:
                logger.error(f"Unexpected error while sending request to {url}: {e}")
                return {'error': str(e)}

    def convert_to_object(self, model_class, data):
        if data is None:
            return None
        return model_class(**data)

    async def get_job(self, job_id: int):
        response = await self.authenticated_request('GET', '/get_job', params={'job_id': job_id})
        return self.convert_to_object(Job, response)

    async def update_job_iteration(self, job_id: int, new_iteration: int):
        data = {'job_id': job_id, 'new_iteration': new_iteration}
        return await self.authenticated_request('POST', '/update_job_iteration', data=data)

    async def mark_job_as_done(self, job_id: int):
        data = {'job_id': job_id}
        return await self.authenticated_request('POST', '/mark_job_as_done', data=data)

    async def create_task(self, job_id: int, subnet_task_id: int, job_iteration: int, status: str):
        data = {
            'job_id': job_id,
            'subnet_task_id': subnet_task_id,
            'job_iteration': job_iteration,
            'status': status
        }
        response = await self.authenticated_request('POST', '/create_task', data=data)
        return response['task_id']

    async def create_job(self, name: str, plugin_id: int, subnet_id: int, sot_url: str, iteration: int):
        data = {
            'name': name,
            'plugin_id': plugin_id,
            'subnet_id': subnet_id,
            'sot_url': sot_url,
            'iteration': iteration
        }
        response = await self.authenticated_request('POST', '/create_job', data=data)
        return response['job_id']

    async def create_subnet(self, address: str, rpc_url: str):
        data = {
            'address': address,
            'rpc_url': rpc_url
        }
        response = await self.authenticated_request('POST', '/create_subnet', data=data)
        return response['subnet_id']

    async def create_plugin(self, name: str, code: str):
        data = {
            'name': name,
            'code': code
        }
        response = await self.authenticated_request('POST', '/create_plugin', data=data)
        return response['plugin_id']
    
    async def update_time_solved(
        self,
        subnet_task_id: int,
        job_id: int,
        time_solved: int
    ):
        data = {
            'subnet_task_id': subnet_task_id,
            'job_id': job_id,
            'time_solved': time_solved
        }
        return await self.authenticated_request('POST', '/update_time_solved', data=data)
    
    async def update_time_solver_selected(
        self,
        subnet_task_id: int,
        job_id: int,
        time_solver_selected: int
    ):
        data = {
            'subnet_task_id': subnet_task_id,
            'job_id': job_id,
            'time_solver_selected': time_solver_selected
        }
        return await self.authenticated_request('POST', '/update_time_solver_selected', data=data)

    async def update_task_status(
        self,
        subnet_task_id: int,
        job_id: int,
        status: str,
        result=None,
        solver_address=None,
    ):
        data = {
            'subnet_task_id': subnet_task_id,
            'job_id': job_id,
            'status': status,
            'result': result,
            'solver_address': solver_address
        }
        return await self.authenticated_request('POST', '/update_task_status', data=data)

    async def create_state_update(self, job_id: int, data: Dict):
        data = {
            'job_id': job_id,
            'data': data
        }
        response = await self.authenticated_request('POST', '/create_state_update', data=data)
        return response['state_update_id']

    async def get_plugin(self, plugin_id: int):
        response = await self.authenticated_request('GET', '/get_plugin', params={'plugin_id': plugin_id})
        return self.convert_to_object(Plugin, response)

    async def get_subnet_using_address(self, address: str):
        response = await self.authenticated_request('GET', '/get_subnet_using_address', params={'address': address})
        return self.convert_to_object(Subnet, response)

    async def get_task(self, subnet_task_id: int, subnet_id: int):
        response = await self.authenticated_request('GET', '/get_task', params={'subnet_task_id': subnet_task_id, 'subnet_id': subnet_id})
        return self.convert_to_object(Task, response)

    async def get_perm(self, address: str, perm: int):
        response = await self.authenticated_request('GET', '/get_perm', params={'address': address, 'perm': perm})
        return self.convert_to_object(Perm, response['perm'])

    async def set_last_nonce(self, address: str, perm: int, last_nonce: str):
        data = {
            'address': address,
            'perm': perm,
            'last_nonce': last_nonce
        }
        return await self.authenticated_request('POST', '/set_last_nonce', data=data)

    async def get_sot(self, id: int):
        response = await self.authenticated_request('GET', '/get_sot', params={'id': id})
        return self.convert_to_object(Sot, response)

    async def create_perm(self, address: str, perm: int):
        data = {
            'address': address,
            'perm': perm
        }
        response = await self.authenticated_request('POST', '/create_perm', data=data)
        return response['perm_id']

    async def create_perm_description(self, perm_type: str):
        data = {
            'perm_type': perm_type,
        }
        response = await self.authenticated_request('POST', '/create_perm_description', data=data)
        return response['perm_description_id']

    async def create_sot(self, job_id: int, url: str):
        data = {
            'job_id': job_id,
            'url': url
        }
        response = await self.authenticated_request('POST', '/create_sot', data=data)
        return response['sot_id']

    async def get_sot_by_job_id(self, job_id: int):
        response = await self.authenticated_request('GET', '/get_sot_by_job_id', params={'job_id': job_id})
        return self.convert_to_object(Sot, response)

