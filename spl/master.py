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
from web3 import AsyncWeb3
from web3.middleware import async_geth_poa_middleware
from web3.exceptions import ContractCustomError, TransactionNotFound
from common import load_contracts, TaskStatus, PoolState, Task, get_learning_hyperparameters, async_transact_with_contract_function, TENSOR_VERSION_INTERVAL, wait_for_state_change, approve_token_once, MAX_SOLVER_SELECTION_DURATION
from io import BytesIO
import os
from eth_account.messages import encode_defunct
from eth_account import Account
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Master:
    def __init__(self, rpc_url, wallets_file, sot_url, subnet_addresses, desired_unresolved_tasks=2, detailed_logs=False):
        self.web3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
        self.web3.middleware_onion.inject(async_geth_poa_middleware, layer=0)
        self.sot_url = sot_url
        self.subnet_addresses = subnet_addresses
        self.desired_unresolved_tasks = desired_unresolved_tasks
        self.iteration = 1  # Track the number of iterations
        self.perplexities = []  # Initialize perplexity list
        self.perplexity_queue = queue.Queue()
        self.load_wallets(wallets_file)
        self.current_wallet_index = 0
        self.unresolved_tasks = []

        if detailed_logs:
            logging.getLogger().setLevel(logging.DEBUG)

        # Initialize contracts
        asyncio.run(self.initialize_contracts())

        # Set up the plot
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], label='Perplexity')
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Perplexity')
        self.ax.set_title('Perplexity over Iterations')
        self.ax.legend()
        self.ax.grid(True)

        # Run the main iterations in separate threads
        asyncio.run(self.run_main())

    def load_wallets(self, wallets_file):
        with open(wallets_file, 'r') as f:
            self.wallets = json.load(f)

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
        while not self.perplexity_queue.empty():
            perplexity = self.perplexity_queue.get()
            self.perplexities.append(perplexity)
        self.line.set_xdata(range(len(self.perplexities)))
        self.line.set_ydata(self.perplexities)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.savefig('perplexity_plot.png')  # Save the figure after each update

    async def submit_task(self, task_type, params, iteration_number):
        try:
            if task_type not in self.contracts:
                raise ValueError(f"No contract loaded for task type {task_type}")

            logging.info(f"Submitting task of type {task_type} with params: {params}")
            encoded_params = json.dumps(params).encode('utf-8')

            for _ in range(5):  # Retry up to 5 times
                try:
                    await wait_for_state_change(self.web3, self.pool, PoolState.Unlocked.value, self.get_next_wallet()['private_key'])
                    receipt = await async_transact_with_contract_function(
                        self.web3, self.contracts[task_type], 'submitTaskRequest', self.get_next_wallet()['private_key'], encoded_params, attempts=1)
                    logging.info(f"submitTaskRequest transaction receipt: {receipt}")

                    logs = self.contracts[task_type].events.TaskRequestSubmitted().process_receipt(receipt)
                    if not logs:
                        raise ValueError("No TaskRequestSubmitted event found in the receipt")

                    task_id = logs[0]['args']['taskId']
                    logging.info(f"Iteration {iteration_number} - Task submitted successfully. Task ID: {task_id}")

                    self.unresolved_tasks.append((task_type, task_id))

                    selection_id = await self.submit_selection_req()
                    logging.info(f"Selection ID: {selection_id}")

                    await self.select_solver(task_type, task_id, iteration_number)
                    await self.remove_solver_stake(task_type, task_id, iteration_number)

                    return task_id
                except Exception as e:
                    logging.error(f"Error submitting task: {e}. Retrying...")
                    await asyncio.sleep(1)  # Wait for a while before retrying

            raise RuntimeError("Failed to submit task after multiple attempts")
        except Exception as e:
            logging.error(f"Error submitting task: {e}")
            raise

    async def submit_selection_req(self):
        try:
            if await self.pool.functions.state().call() != PoolState.Unlocked.value:
                return await self.pool.functions.currentSelectionId().call()

            await wait_for_state_change(self.web3, self.pool, PoolState.Unlocked.value, self.get_next_wallet()['private_key'])
            logging.info("Submitting selection request")

            receipt = await async_transact_with_contract_function(
                self.web3, self.pool, 'submitSelectionReq', self.get_next_wallet()['private_key'], attempts=1)
            logging.info(f"submitSelectionReq transaction receipt: {receipt}")

            unlocked_min_period = await self.pool.functions.UNLOCKED_MIN_PERIOD().call()
            last_state_change_time = await self.pool.functions.lastStateChangeTime().call()
            current_time = await self.web3.eth.get_block('latest')['timestamp']
            remaining_time = (last_state_change_time + unlocked_min_period) - current_time

            if remaining_time > 0:
                logging.info(f"Waiting for {remaining_time} seconds until UNLOCKED_MIN_PERIOD is over")
                await asyncio.sleep(remaining_time)

            logs = self.pool.events.SelectionRequested().process_receipt(receipt)
            if not logs:
                raise ValueError("No SelectionRequested event found in the receipt")

            return logs[0]['args']['selectionId']
        except Exception as e:
            logging.error(f"Error submitting selection request: {e}")
            raise

    async def select_solver(self, task_type, task_id, iteration_number, retry_delay=1):
        max_duration = datetime.timedelta(seconds=MAX_SOLVER_SELECTION_DURATION)
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
                logging.info(f"Selecting solver for task ID: {task_id}, attempt {attempt}")
                
                start_wait_time = time.time()
                await wait_for_state_change(self.web3, self.pool, PoolState.SelectionsFinalizing.value, self.get_next_wallet()['private_key'])
                end_wait_time = time.time()
                logging.info(f"wait_for_state_change duration: {end_wait_time - start_wait_time} seconds")
                
                start_transact_time = time.time()
                receipt = await async_transact_with_contract_function(
                    self.web3, self.contracts[task_type], 'selectSolver', self.get_next_wallet()['private_key'], task_id, attempts=1)
                end_transact_time = time.time()
                logging.info(f"Transaction duration: {end_transact_time - start_transact_time} seconds")
                
                logging.info(f"Iteration {iteration_number} - selectSolver transaction receipt: {receipt}")
                return
            except Exception as e:
                logging.error(f"Error selecting solver on attempt {attempt}: {e}")
                logging.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)  # Exponential backoff

    async def remove_solver_stake(self, task_type, task_id, iteration_number):
        try:
            logging.info(f"Removing solver stake for task ID: {task_id}")

            await wait_for_state_change(self.web3, self.pool, PoolState.Unlocked.value, self.get_next_wallet()['private_key'])
            receipt = await async_transact_with_contract_function(
                self.web3, self.contracts[task_type], 'removeSolverStake', self.get_next_wallet()['private_key'], task_id, attempts=1)
            logging.info(f"Iteration {iteration_number} - removeSolverStake transaction receipt: {receipt}")
        except Exception as e:
            logging.error(f"Error removing solver stake: {e}")
            raise

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
        current_version_number = int(time.time()) // TENSOR_VERSION_INTERVAL * TENSOR_VERSION_INTERVAL
        
        model_params = await self.get_latest_model_params(current_version_number)
        batch_url, targets_url = await self.get_batch_and_targets_url()

        logging.info(f"Iteration {iteration_number}: Starting embed forward task")
        embed_result = await self.handle_embed_forward(model_params, batch_url, current_version_number, iteration_number)

        layer_inputs_url = [embed_result['result_url']]
        for layer_idx in range(model_params['n_layers']):
            logging.info(f"Iteration {iteration_number}: Starting forward task for layer {layer_idx}")
            layer_result = await self.handle_layer_forward(layer_idx, layer_inputs_url[-1], model_params, current_version_number, iteration_number)
            layer_inputs_url.append(layer_result['result_url'])

        logging.info(f"Iteration {iteration_number}: Starting final_logits task")
        final_logits_result = await self.handle_final_logits(layer_inputs_url[-1], targets_url, current_version_number, iteration_number)

        # Update perplexity list and plot
        perplexity = math.exp(final_logits_result['loss'])
        self.perplexity_queue.put(perplexity)

        error_url = final_logits_result['error_output_url']

        for layer_idx in reversed(range(model_params['n_layers'])):
            logging.info(f"Iteration {iteration_number}: Starting backward task for layer {layer_idx}")
            layer_result = await self.handle_layer_backward(layer_idx, error_url, layer_inputs_url[layer_idx], current_version_number, iteration_number)
            error_url = layer_result['error_output_url']

        logging.info(f"Iteration {iteration_number}: Starting embed backward task")
        await self.handle_embed_backward(error_url, batch_url, current_version_number, iteration_number)

        logging.info(f"Iteration {iteration_number} done, loss: {final_logits_result['loss']}")

        # Update the plot explicitly
        self.update_plot()

    async def run_main(self):
        logging.info("Starting main process")
        await self.approve_tokens_at_start()

        while True:
            await self.check_and_submit_tasks()
            await asyncio.sleep(5)  # Check every 5 seconds

    async def check_and_submit_tasks(self):
        # Filter unresolved tasks
        unresolved_tasks = []
        for task_type, task_id in self.unresolved_tasks:
            if task_type != 'embed':
                continue
            task_tuple = await self.contracts[task_type].functions.getTask(task_id).call()
            task = Task(*task_tuple)
            if task.status not in [TaskStatus.ResolvedCorrect.value, TaskStatus.ResolvedIncorrect.value]:
                unresolved_tasks.append((task_type, task_id))

        self.unresolved_tasks = unresolved_tasks

        # Submit new tasks if necessary
        if len(self.unresolved_tasks) < self.desired_unresolved_tasks:
            tasks_to_submit = self.desired_unresolved_tasks - len(self.unresolved_tasks)
            for _ in range(tasks_to_submit):
                asyncio.create_task(self.main_iteration(self.iteration))
                self.iteration += 1

    async def get_latest_model_params(self, current_version_number):
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.sot_url}/latest_model_params", params={'version_number': current_version_number}) as response:
                return await response.json()

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

    async def handle_embed_forward(self, model_params, batch_url, current_version_number, iteration_number):
        task_params = {'batch_url': batch_url, 'model_params': model_params, 'version_number': current_version_number}
        task_id = await self.submit_task('embed', task_params, iteration_number)
        result = await self.wait_for_result('embed', task_id, iteration_number)
        result['batch_url'] = batch_url
        return result

    async def handle_layer_forward(self, layer_idx, inputs_url, model_params, current_version_number, iteration_number):
        task_type = f'forward_layer_{layer_idx}'
        task_params = {'layer_idx': layer_idx, 'inputs_url': inputs_url, 'model_params': model_params, 'version_number': current_version_number}
        task_id = await self.submit_task(task_type, task_params, iteration_number)
        result = await self.wait_for_result(task_type, task_id, iteration_number)
        return result

    async def handle_final_logits(self, inputs_url, targets_url, current_version_number, iteration_number):
        task_params = {'inputs_url': inputs_url, 'targets_url': targets_url, 'version_number': current_version_number}
        task_id = await self.submit_task('final_logits', task_params, iteration_number)
        result = await self.wait_for_result('final_logits', task_id, iteration_number)
        await self.update_sot_all('final_logits', result, iteration_number=iteration_number)
        return result

    async def handle_layer_backward(self, layer_idx, error_url, inputs_url, current_version_number, iteration_number):
        task_type = f'backward_layer_{layer_idx}'
        task_params = {
            'layer_idx': layer_idx,
            'error_url': error_url,
            'inputs_url': inputs_url,
            'version_number': current_version_number
        }
        task_id = await self.submit_task(task_type, task_params, iteration_number)
        result = await self.wait_for_result(task_type, task_id, iteration_number)
        await self.update_sot_all(task_type, result, layer_idx, iteration_number)
        return result

    async def handle_embed_backward(self, error_url, batch_url, current_version_number, iteration_number):
        task_params = {
            'error_url': error_url,
            'batch_url': batch_url,
            'version_number': current_version_number
        }
        task_id = await self.submit_task('embed_backward', task_params, iteration_number)
        result = await self.wait_for_result('embed_backward', task_id, iteration_number)
        await self.update_sot_all('embed_backward', result, iteration_number=iteration_number)
        return result

    async def wait_for_result(self, task_type, task_id, iteration_number):
        while True:
            result = await self.get_task_result(task_type, task_id, iteration_number)
            if result is not None:
                return result
            await asyncio.sleep(5)

    async def update_sot(self, tensor_name, result):
        learning_params = get_learning_hyperparameters(self.iteration)
        params = {
            'result_url': result['grads_url'],
            'tensor_name': tensor_name,
            'version_number': result['version_number'],
            **learning_params
        }

        wallet = self.get_next_wallet()
        message = json.dumps(self.generate_message('update_state'), sort_keys=True)
        signature = self.sign_message(message, wallet)
        headers = {'Authorization': f'{message}:{signature}'}

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.sot_url}/update_state", json=params, headers=headers) as response:
                if response.status != 200:  # Use 'status' instead of 'status_code'
                    logging.error(f"Failed to update SOT for {tensor_name}: {await response.text()}")
                else:
                    logging.info(f"Updated SOT for {tensor_name} with result: {result}")

    async def update_sot_all(self, task_type, result, layer_idx=None, iteration_number=None):
        if task_type == 'embed_backward':
            tensor_name = 'embed'
        elif task_type == 'final_logits':
            tensor_name = 'final_logits'
        else:
            tensor_name = f'layer_{layer_idx}'
        await self.update_sot(tensor_name, result)

    async def get_batch_and_targets_url(self):
        wallet = self.get_next_wallet()
        message = json.dumps(self.generate_message('get_batch'), sort_keys=True)
        signature = self.sign_message(message, wallet)
        headers = {'Authorization': f'{message}:{signature}'}

        url = os.path.join(self.sot_url, 'get_batch')
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers) as response:
                response_json = await response.json()
                return response_json['batch_url'], response_json['targets_url']

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Master process for task submission")
    parser.add_argument('--rpc_url', type=str, required=True, help="RPC URL for Ethereum node")
    parser.add_argument('--wallets_file', type=str, required=True, help="Path to wallets JSON file")
    parser.add_argument('--sot_url', type=str, required=True, help="Source of Truth URL")
    parser.add_argument('--subnet_addresses', type=str, required=True, help="Path to subnet addresses JSON file")
    parser.add_argument('--desired_unresolved_tasks', type=int, default=2, help="Desired number of unresolved tasks to maintain")
    parser.add_argument('--detailed_logs', action='store_true', help="Enable detailed logs")

    args = parser.parse_args()

    with open(args.subnet_addresses, 'r') as file:
        subnet_addresses = json.load(file)

    master = Master(args.rpc_url, args.wallets_file, args.sot_url, subnet_addresses, desired_unresolved_tasks=args.desired_unresolved_tasks, detailed_logs=args.detailed_logs)
