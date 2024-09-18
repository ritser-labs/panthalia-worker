import asyncio
import json
import logging
import time
import datetime
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import aiohttp
import queue
import requests
from web3 import AsyncWeb3
from web3.middleware import async_geth_poa_middleware
from web3.exceptions import ContractCustomError, TransactionNotFound
from .common import load_contracts, TaskStatus, PoolState, Task, get_master_learning_hyperparameters, async_transact_with_contract_function, TENSOR_VERSION_INTERVAL, wait_for_state_change, approve_token_once, MAX_SUBMIT_TASK_RETRY_DURATION, MAX_SELECT_SOLVER_TIME, TENSOR_NAME
from io import BytesIO
import os
from eth_account.messages import encode_defunct
from eth_account import Account
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Master:
    def __init__(self, rpc_url, wallets, sot_url, subnet_addresses, max_concurrent_iterations=2, detailed_logs=False):
        self.web3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
        self.web3.middleware_onion.inject(async_geth_poa_middleware, layer=0)
        self.sot_url = sot_url
        self.subnet_addresses = subnet_addresses
        self.max_concurrent_iterations = max_concurrent_iterations
        self.iteration = 1  # Track the number of iterations
        self.losses = []  # Initialize loss list
        self.loss_queue = queue.Queue()
        self.load_wallets(wallets)
        self.current_wallet_index = 0
        if detailed_logs:
            logging.getLogger().setLevel(logging.DEBUG)

        # Initialize contracts
        asyncio.run(self.initialize_contracts())

        # Set up the plot
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], label='Loss')
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Loss over Iterations')
        self.ax.legend()
        self.ax.grid(True)

        # Run the main process
        self.tasks = []  # Track running tasks
        asyncio.run(self.run_main())

    def load_wallets(self, wallets_string):
        with open(wallets_string, 'r') as file:
            self.wallets = json.load(file)

    def get_next_wallet(self):
        wallet = self.wallets[self.current_wallet_index]
        self.current_wallet_index = (self.current_wallet_index + 1) % len(self.wallets)
        return wallet

    async def initialize_contracts(self):
        self.abis, self.contracts, self.error_selectors = load_contracts(self.web3, self.subnet_addresses)
        if not self.contracts:
            raise ValueError("SubnetManager contracts not found. Please check the subnet_addresses configuration.")

        self.pool_address = None
        for contract in self.contracts.values():
            if hasattr(contract.functions, 'pool'):
                self.pool_address = await contract.functions.pool().call()
                break
        if not self.pool_address:
            raise ValueError("Pool contract address not found in any of the SubnetManager contracts.")

        self.pool = self.web3.eth.contract(address=self.pool_address, abi=self.abis['Pool'])

    async def approve_tokens_at_start(self):
        tasks = []
        for contract in self.contracts.values():
            token_address = await contract.functions.token().call()
            token_contract = self.web3.eth.contract(address=token_address, abi=self.abis['ERC20'])
            for wallet in self.wallets:
                task = approve_token_once(self.web3, token_contract, wallet['private_key'], contract.address, 2**256 - 1)
                tasks.append(task)
            await asyncio.gather(*tasks)
            tasks = []

    def update_plot(self, frame=None):
        while not self.loss_queue.empty():
            loss = self.loss_queue.get()
            self.losses.append(loss)
        self.line.set_xdata(range(len(self.losses)))
        self.line.set_ydata(self.losses)
        self.ax.relim()
        self.ax.autoscale_view()
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, 'loss_plot.png')
        self.fig.savefig(file_path)  # Save the figure after each update

    async def submit_task(self, task_type, params, iteration_number):
        max_duration = datetime.timedelta(seconds=MAX_SUBMIT_TASK_RETRY_DURATION)
        start_time = datetime.datetime.now()

        attempt = 0
        retry_delay = 1
        while True:
            attempt += 1
            current_time = datetime.datetime.now()
            elapsed_time = current_time - start_time

            if elapsed_time >= max_duration:
                logging.error(f"Failed to select solver within {max_duration}")
                raise RuntimeError(f"Failed to select solver within {max_duration}")

            try:
                if task_type not in self.contracts:
                    raise ValueError(f"No contract loaded for task type {task_type}")

                logging.info(f"Submitting task of type {task_type} with params: {params}")
                encoded_params = json.dumps(params).encode('utf-8')

                await wait_for_state_change(self.web3, self.pool, PoolState.Unlocked.value, self.get_next_wallet()['private_key'])
                receipt = await async_transact_with_contract_function(
                    self.web3, self.contracts[task_type], 'submitTaskRequest', self.get_next_wallet()['private_key'], encoded_params, attempts=1)
                logging.info(f"submitTaskRequest transaction receipt: {receipt}")

                logs = self.contracts[task_type].events.TaskRequestSubmitted().process_receipt(receipt)
                if not logs:
                    raise ValueError("No TaskRequestSubmitted event found in the receipt")

                task_id = logs[0]['args']['taskId']
                logging.info(f"Iteration {iteration_number} - Task submitted successfully. Task ID: {task_id}")

                return task_id
            except Exception as e:
                logging.error(f"Error submitting task on attempt {attempt}: {e}")
                logging.error(f'Retrying in {retry_delay} seconds...')
                retry_delay = min(2 * retry_delay, 60)
                await asyncio.sleep(retry_delay)

    async def select_solver(self, task_type, task_id, iteration_number):
        max_duration = datetime.timedelta(seconds=MAX_SELECT_SOLVER_TIME)
        start_time = datetime.datetime.now()

        attempt = 0
        while True:
            attempt += 1
            current_time = datetime.datetime.now()
            elapsed_time = current_time - start_time

            if elapsed_time >= max_duration:
                logging.error(f"Failed to select solver within {max_duration}")
                raise RuntimeError(f"Failed to select solver within {max_duration}")

            try:
                logging.info(f"Selecting solver for task ID: {task_id}")

                start_transact_time = time.time()
                receipt = await async_transact_with_contract_function(
                    self.web3, self.contracts[task_type], 'selectSolver', self.get_next_wallet()['private_key'], task_id, attempts=1)
                end_transact_time = time.time()

                logging.info(f"Transaction duration: {end_transact_time - start_transact_time} seconds")
                logging.info(f"Iteration {iteration_number} - selectSolver transaction receipt: {receipt}")
                return True
            except Exception as e:
                logging.error(f"Error selecting solver on attempt {attempt}: {e}")
                logging.error(f'Retrying...')
                await asyncio.sleep(0.5)


    async def get_task_result(self, task_type, task_id, iteration_number):
        try:
            task_tuple = await self.contracts[task_type].functions.getTask(task_id).call()
            task = Task(*task_tuple)
            logging.info(f"Iteration {iteration_number} - {task_type} Task status: {task.status}")
            logging.info(f"Expected status: {TaskStatus.SolutionSubmitted.value}")
            if task.status == TaskStatus.SolutionSubmitted.value or task.status == TaskStatus.ResolvedCorrect.value:
                return json.loads(task.postedSolution.decode('utf-8'))
            return None
        except Exception as e:
            logging.error(f"Error getting task result for {task_type} with task ID {task_id}: {e}")
            return None

    async def main_iteration(self, iteration_number):
        logging.info(f"Starting iteration {iteration_number}")

        learning_params = get_master_learning_hyperparameters(iteration_number)
        logging.info(f'Learning parameters for iteration {iteration_number}: {learning_params}')
        batch_url, targets_url = await self.get_batch_and_targets_url()

        logging.info(f"Iteration {iteration_number}: Starting training task")
        task_params = {
            'batch_url': batch_url,
            'targets_url': targets_url,
            **learning_params,
        }
        task_id = await self.submit_task(TENSOR_NAME, task_params, iteration_number)
        
        await wait_for_state_change(self.web3, self.pool, PoolState.SelectionsFinalizing.value, self.get_next_wallet()['private_key'])
        await wait_for_state_change(self.web3, self.pool, PoolState.Unlocked.value, self.get_next_wallet()['private_key'])

        await self.select_solver(TENSOR_NAME, task_id, iteration_number)
        result = await self.wait_for_result(TENSOR_NAME, task_id, iteration_number)
        loss_value = result['loss']
        self.loss_queue.put(loss_value)
        await self.update_latest_loss(loss_value, result['version_number'])
        await self.update_sot(learning_params, TENSOR_NAME, result, batch_url, targets_url)


        task = asyncio.create_task(self.main_iteration(self.iteration))
        self.tasks.append(task)
        self.iteration += 1

        self.update_plot()

    async def run_main(self):
        logging.info("Starting main process")
        await self.approve_tokens_at_start()

        # Start the initial set of iterations
        for _ in range(self.max_concurrent_iterations):
            task = asyncio.create_task(self.main_iteration(self.iteration))
            self.tasks.append(task)
            self.iteration += 1
        
        # Dynamically await new tasks as they are added
        while True:
            if self.tasks:
                await asyncio.wait(self.tasks, return_when=asyncio.FIRST_COMPLETED)
                self.tasks = [task for task in self.tasks if not task.done()]

    def sign_message(self, message, wallet):
        message = encode_defunct(text=message)
        account = self.web3.eth.account.from_key(wallet['private_key'])
        signed_message = account.sign_message(message)
        return signed_message.signature.hex()

    def generate_message(self, endpoint):
        nonce = str(uuid.uuid4())
        timestamp = int(time.time())
        message = {
            'endpoint': endpoint,
            'nonce': nonce,
            'timestamp': timestamp
        }
        return message
    
    async def wait_for_result(self, task_type, task_id, iteration_number):
        while True:
            result = await self.get_task_result(task_type, task_id, iteration_number)
            if result is not None:
                return result
            await asyncio.sleep(0.5)

    async def update_sot(self, learning_params, tensor_name, result, batch_url, targets_url):
        params = {
            'result_url': result['grads_url'],
            'tensor_name': tensor_name,
            'version_number': result['version_number'],
            'batch_url': batch_url,
            'targets_url': targets_url,
            **learning_params
        }

        wallet = self.get_next_wallet()
        message = json.dumps(self.generate_message('update_state'), sort_keys=True)
        signature = self.sign_message(message, wallet)
        headers = {'Authorization': f'{message}:{signature}'}

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.sot_url}/update_state", json=params, headers=headers) as response:
                if response.status != 200:
                    logging.error(f"Failed to update SOT for {tensor_name}: {await response.text()}")
                else:
                    logging.info(f"Updated SOT for {tensor_name} with result: {result}")
    
    async def update_latest_loss(self, loss_value, version_number):
        """Send the latest loss value to the SOT server."""
        payload = {'loss': loss_value, 'version_number': version_number}

        url = os.path.join(self.sot_url, 'update_loss')
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    logging.error(f"Failed to update loss value: {await response.text()}")
                else:
                    logging.info(f"Updated latest loss value to {loss_value}")

    async def get_batch_and_targets_url(self):
        wallet = self.get_next_wallet()
        url = os.path.join(self.sot_url, 'get_batch')

        retry_delay = 1
        max_retries = 400
        retries = 0
        while retries < max_retries:
            try:
                message = json.dumps(self.generate_message('get_batch'), sort_keys=True)
                signature = self.sign_message(message, wallet)
                headers = {'Authorization': f'{message}:{signature}'}
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers) as response:
                        if response.status == 200:
                            response_json = await response.json()
                            return self.sot_url + response_json['batch_url'], self.sot_url + response_json['targets_url']
                        else:
                            print(f"Request failed with status code {response.status}")
            except Exception as e:
                print(f"Request failed: {e}. Retrying in {retry_delay} seconds...")

            retries += 1
            await asyncio.sleep(retry_delay)
        
        raise Exception(f"Failed to retrieve batch and targets URL after {self.max_retries} attempts.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Master process for task submission")
    parser.add_argument('--rpc_url', type=str, required=True, help="RPC URL for Ethereum node")
    parser.add_argument('--wallets', type=str, required=True, help="URL to wallets JSON file")
    parser.add_argument('--sot_url', type=str, required=True, help="Source of Truth URL")
    parser.add_argument('--subnet_addresses', type=str, required=True, help="Path to subnet addresses JSON file")
    parser.add_argument('--max_concurrent_iterations', type=int, default=4, help="Maximum number of concurrent iterations")
    parser.add_argument('--detailed_logs', action='store_true', help="Enable detailed logs")

    args = parser.parse_args()

    with open(args.subnet_addresses, 'r') as file:
        subnet_addresses = json.load(file)

    master = Master(args.rpc_url, args.wallets, args.sot_url, subnet_addresses, max_concurrent_iterations=args.max_concurrent_iterations, detailed_logs=args.detailed_logs)
