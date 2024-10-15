# db_adapter_client.py

import aiohttp
import asyncio
import logging
import json
from ..util.json import load_json, save_json
from eth_account.messages import encode_defunct
from ..models import PermType
from eth_account import Account
import uuid
import time

logger = logging.getLogger(__name__)

class DBAdapterClient:
    def __init__(self, server_url, private_key=None):
        """
        Initialize the client with the server URL and authentication details.

        Args:
            server_url (str): Base URL of the DB server.
            private_key (str): Private key for signing messages.
            address (str): Ethereum address associated with the private key.
            perm_type (PermType): Permission type required for operations.
        """
        self.server_url = server_url.rstrip('/')
        self.private_key = private_key

    def generate_message(self, endpoint, data=None):
        """
        Generate a signed message for authentication.

        Args:
            endpoint (str): The API endpoint being accessed.
            data (dict, optional): Additional data to include in the message.

        Returns:
            str: JSON-encoded message.
        """
        message = {
            "endpoint": endpoint,
            "nonce": str(uuid.uuid4()),
            "timestamp": int(time.time()),
            "data": data or {}
        }
        message_json = json.dumps(message, sort_keys=True)
        return message_json

    def sign_message(self, message):
        """
        Sign the message with the private key.

        Args:
            message (str): The message to sign.

        Returns:
            str: The signature in hexadecimal format.
        """
        message_defunct = encode_defunct(text=message)
        account = Account.from_key(self.private_key)
        signed_message = account.sign_message(message_defunct)
        return signed_message.signature.hex()

    async def authenticated_request(self, method, endpoint, data=None, params=None):
        """
        Send an HTTP request to the server, with authentication for non-GET requests.

        Args:
            method (str): HTTP method (GET, POST, etc.).
            endpoint (str): API endpoint.
            data (dict, optional): JSON data to send in the body.
            params (dict, optional): URL parameters.

        Returns:
            dict: JSON response from the server or error information.
        """
        url = f"{self.server_url}{endpoint}"
        
        # Only include authentication headers for non-GET requests
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
                        # Response is not JSON
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

    async def get_job(self, job_id: int):
        params = {'job_id': job_id}
        return await self.authenticated_request('GET', '/get_job', params=params)

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
        return (await self.authenticated_request('POST', '/create_task', data=data))['task_id']

    async def create_job(self, name: str, plugin_id: int, subnet_id: int, sot_url: str, iteration: int):
        data = {
            'name': name,
            'plugin_id': plugin_id,
            'subnet_id': subnet_id,
            'sot_url': sot_url,
            'iteration': iteration
        }
        return (await self.authenticated_request('POST', '/create_job', data=data))['job_id']

    async def create_subnet(self, address: str, rpc_url: str):
        data = {
            'address': address,
            'rpc_url': rpc_url
        }
        return (await self.authenticated_request('POST', '/create_subnet', data=data))['subnet_id']

    async def create_plugin(self, name: str, code: str):
        data = {
            'name': name,
            'code': code
        }
        return (await self.authenticated_request('POST', '/create_plugin', data=data))['plugin_id']

    async def update_task_status(self, subnet_task_id: int, status: str, result=None):
        data = {
            'subnet_task_id': subnet_task_id,
            'status': status,
            'result': result
        }
        return await self.authenticated_request('POST', '/update_task_status', data=data)

    async def create_state_update(self, job_id: int, state_iteration: int):
        data = {
            'job_id': job_id,
            'state_iteration': state_iteration
        }
        return (await self.authenticated_request('POST', '/create_state_update', data=data))['state_update_id']

    async def get_plugin_code(self, plugin_id: int):
        params = {'plugin_id': plugin_id}
        return (await self.authenticated_request('GET', '/get_plugin_code', params=params))['code']

    async def get_subnet_using_address(self, address: str):
        params = {'address': address}
        return await self.authenticated_request('GET', '/get_subnet_using_address', params=params)

    async def get_task(self, subnet_task_id: int, subnet_id: int):
        params = {'subnet_task_id': subnet_task_id, 'subnet_id': subnet_id}
        return await self.authenticated_request('GET', '/get_task', params=params)

    async def has_perm(self, address: str, perm: int):
        params = {'address': address, 'perm': perm}
        return (await self.authenticated_request('GET', '/has_perm', params=params))['perm']

    async def set_last_nonce(self, address: str, perm: int, last_nonce: str):
        data = {
            'address': address,
            'perm': perm,
            'last_nonce': last_nonce
        }
        return await self.authenticated_request('POST', '/set_last_nonce', data=data)

    async def get_sot(self, id: int):
        params = {'id': id}
        return await self.authenticated_request('GET', '/get_sot', params=params)

    async def create_perm(self, address: str, perm: int):
        data = {
            'address': address,
            'perm': perm
        }
        return (await self.authenticated_request('POST', '/create_perm', data=data))['perm_id']

    async def create_perm_description(self, perm_type: str):
        data = {
            'perm_type': perm_type
        }
        return (await self.authenticated_request('POST', '/create_perm_description', data=data))['perm_description_id']

    async def create_sot(self, job_id: int, url: str):
        data = {
            'job_id': job_id,
            'url': url
        }
        return (await self.authenticated_request('POST', '/create_sot', data=data))['sot_id']

    async def get_sot_by_job_id(self, job_id: int):
        params = {'job_id': job_id}
        return await self.authenticated_request('GET', '/get_sot_by_job_id', params=params)

# Example usage:
# async def main():
#     client = DBAdapterClient(
#         server_url="http://localhost:8000",
#         private_key="YOUR_PRIVATE_KEY",
#         perm_type=PermType.ModifyDb
#     )
#     job = await client.get_job(1)
#     print(job)

# if __name__ == "__main__":
#     asyncio.run(main())
