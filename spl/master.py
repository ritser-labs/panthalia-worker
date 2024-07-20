import asyncio
import json
import logging
import time
import requests
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import exp
from web3 import Web3
from web3.middleware import geth_poa_middleware
from web3.exceptions import ContractCustomError, TransactionNotFound
from common import load_contracts, TaskStatus, PoolState, Task, get_learning_hyperparameters, async_transact_with_contract_function, TENSOR_VERSION_INTERVAL, decode_custom_error, wait_for_state_change
from io import BytesIO
import os
import math
import aiohttp
import threading
import queue

logging.basicConfig(level=logging.INFO)

class Master:
    def __init__(self, rpc_url, private_key, sot_url, subnet_addresses, detailed_logs=False):
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
        self.account = self.web3.eth.account.from_key(private_key)
        self.sot_url = sot_url
        self.subnet_addresses = subnet_addresses
        self.abis, self.contracts, self.error_selectors = load_contracts(self.web3, subnet_addresses)
        self.iteration = 1  # Track the number of iterations
        self.perplexities = []  # Initialize perplexity list
        self.perplexity_queue = queue.Queue()

        if not self.contracts:
            raise ValueError("SubnetManager contracts not found. Please check the subnet_addresses configuration.")

        self.pool_address = None
        for contract in self.contracts.values():
            if hasattr(contract.functions, 'pool'):
                self.pool_address = contract.functions.pool().call()
                break
        if not self.pool_address:
            raise ValueError("Pool contract address not found in any of the SubnetManager contracts.")

        self.pool = self.web3.eth.contract(address=self.pool_address, abi=self.abis['Pool'])

        if detailed_logs:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Set up the live plot
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], label='Perplexity')
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Perplexity')
        self.ax.set_title('Perplexity over Iterations')
        self.ax.legend()
        self.ax.grid(True)

        # Set up the animation
        self.anim = FuncAnimation(self.fig, self.update_plot, interval=1000)
        plt.show()

        # Start the worker thread
        self.worker_thread = threading.Thread(target=self.run_main_iterations)
        self.worker_thread.start()

    def update_plot(self, frame):
        while not self.perplexity_queue.empty():
            perplexity = self.perplexity_queue.get()
            self.perplexities.append(perplexity)
        self.line.set_xdata(range(len(self.perplexities)))
        self.line.set_ydata(self.perplexities)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def run_main_iterations(self):
        asyncio.run(self.main(args.max_simultaneous_iterations))

    async def approve_token(self, token_address, spender_address, amount):
        token_contract = self.web3.eth.contract(address=token_address, abi=self.abis['ERC20'])
        receipt = await async_transact_with_contract_function(self.web3, token_contract, 'approve', self.account._private_key, spender_address, amount, gas=100000)
        logging.info(f"Approved token transaction receipt: {receipt}")

    async def submit_task(self, task_type, params, iteration_number):
        try:
            if task_type not in self.contracts:
                raise ValueError(f"No contract loaded for task type {task_type}")

            logging.info(f"Submitting task of type {task_type} with params: {params}")
            encoded_params = json.dumps(params).encode('utf-8')

            fee = self.contracts[task_type].functions.calculateFee(0).call()
            token_address = self.contracts[task_type].functions.token().call()
            spender_address = self.contracts[task_type].address
            await self.approve_token(token_address, spender_address, fee)

            for _ in range(5):  # Retry up to 5 times
                try:
                    await wait_for_state_change(self.web3, self.pool, PoolState.Unlocked.value, self.account._private_key)
                    receipt = await async_transact_with_contract_function(self.web3, self.contracts[task_type], 'submitTaskRequest', self.account._private_key, encoded_params, gas=1000000)
                    logging.info(f"submitTaskRequest transaction receipt: {receipt}")

                    logs = self.contracts[task_type].events.TaskRequestSubmitted().process_receipt(receipt)
                    if not logs:
                        raise ValueError("No TaskRequestSubmitted event found in the receipt")

                    task_id = logs[0]['args']['taskId']
                    logging.info(f"Iteration {iteration_number} - Task submitted successfully. Task ID: {task_id}")

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
            if self.pool.functions.state().call() != PoolState.Unlocked.value:
                return self.pool.functions.currentSelectionId().call()

            await wait_for_state_change(self.web3, self.pool, PoolState.Unlocked.value, self.account._private_key)
            logging.info("Submitting selection request")

            receipt = await async_transact_with_contract_function(self.web3, self.pool, 'submitSelectionReq', self.account._private_key, gas=500000)
            logging.info(f"submitSelectionReq transaction receipt: {receipt}")

            unlocked_min_period = self.pool.functions.UNLOCKED_MIN_PERIOD().call()
            last_state_change_time = self.pool.functions.lastStateChangeTime().call()
            current_time = time.time()
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

    async def select_solver(self, task_type, task_id, iteration_number, max_retries=5, retry_delay=1):
        for attempt in range(max_retries):
            try:
                logging.info(f"Selecting solver for task ID: {task_id}, attempt {attempt + 1}/{max_retries}")

                await wait_for_state_change(self.web3, self.pool, PoolState.SelectionsFinalizing.value, self.account._private_key)
                receipt = await async_transact_with_contract_function(self.web3, self.contracts[task_type], 'selectSolver', self.account._private_key, task_id, gas=1000000)
                logging.info(f"Iteration {iteration_number} - selectSolver transaction receipt: {receipt}")
                return
            except Exception as e:
                logging.error(f"Error selecting solver on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logging.error(f"Failed to select solver after {max_retries} attempts")
                    raise

    async def remove_solver_stake(self, task_type, task_id, iteration_number):
        try:
            logging.info(f"Removing solver stake for task ID: {task_id}")

            await wait_for_state_change(self.web3, self.pool, PoolState.Unlocked.value, self.account._private_key)
            receipt = await async_transact_with_contract_function(self.web3, self.contracts[task_type], 'removeSolverStake', self.account._private_key, task_id, gas=1000000)
            logging.info(f"Iteration {iteration_number} - removeSolverStake transaction receipt: {receipt}")
        except Exception as e:
            logging.error(f"Error removing solver stake: {e}")
            raise

    async def get_task_result(self, task_type, task_id, iteration_number):
        try:
            task_tuple = self.contracts[task_type].functions.getTask(task_id).call()
            task = Task(*task_tuple)
            logging.info(f"Iteration {iteration_number} - {task_type} Task status: {task.status}")
            logging.info(f"Expected status: {TaskStatus.SolutionSubmitted.value}")
            if task.status == TaskStatus.SolutionSubmitted.value:
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

        logging.info(f"Iteration {iteration_number}: Starting final logits forward task")
        final_logits_result = await self.handle_final_logits_forward(layer_inputs_url[-1], current_version_number, iteration_number)

        logging.info(f"Iteration {iteration_number}: Starting loss computation task")
        loss_result = await self.handle_loss_computation(final_logits_result['result_url'], targets_url, current_version_number, iteration_number)

        # Update perplexity list and plot
        #perplexity = exp(loss_result['loss'])
        perplexity = loss_result['loss']
        self.perplexity_queue.put(perplexity)

        error_url = loss_result['result_url']

        logging.info(f"Iteration {iteration_number}: Starting final logits backward task")
        final_logits_result = await self.handle_final_logits_backward(error_url, layer_inputs_url[-1], current_version_number, iteration_number)

        error_url = final_logits_result['error_output_url']

        for layer_idx in reversed(range(model_params['n_layers'])):
            logging.info(f"Iteration {iteration_number}: Starting backward task for layer {layer_idx}")
            layer_result = await self.handle_layer_backward(layer_idx, error_url, layer_inputs_url[layer_idx], current_version_number, iteration_number)
            error_url = layer_result['error_output_url']

        logging.info(f"Iteration {iteration_number}: Starting embed backward task")
        await self.handle_embed_backward(error_url, batch_url, current_version_number, iteration_number)

        logging.info(f"Iteration {iteration_number} done, loss: {loss_result['loss']}")

    async def main(self, max_simultaneous_iterations):
        logging.info("Starting main process")
        tasks = set()

        for _ in range(max_simultaneous_iterations):
            tasks.add(asyncio.create_task(self.main_iteration(self.iteration)))
            self.iteration += 1

        while tasks:
            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                try:
                    await task
                except Exception as e:
                    logging.error(f"Iteration ended with error: {e}")

                new_task = asyncio.create_task(self.main_iteration(self.iteration))
                tasks.add(new_task)
                self.iteration += 1

        logging.info("All iterations completed")

    async def get_latest_model_params(self, current_version_number):
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.sot_url}/latest_model_params", params={'version_number': current_version_number}) as response:
                return await response.json()

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

    async def handle_final_logits_forward(self, inputs_url, current_version_number, iteration_number):
        task_params = {'inputs_url': inputs_url, 'version_number': current_version_number}
        task_id = await self.submit_task('final_logits', task_params, iteration_number)
        result = await self.wait_for_result('final_logits', task_id, iteration_number)
        return result

    async def handle_loss_computation(self, logits_url, targets_url, current_version_number, iteration_number):
        task_params = {'logits_url': logits_url, 'targets_url': targets_url, 'version_number': current_version_number}
        task_id = await self.submit_task('loss', task_params, iteration_number)
        result = await self.wait_for_result('loss', task_id, iteration_number)
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

    async def handle_final_logits_backward(self, error_url, inputs_url, current_version_number, iteration_number):
        task_params = {
            'error_url': error_url,
            'inputs_url': inputs_url,
            'version_number': current_version_number
        }
        task_id = await self.submit_task('final_logits_backward', task_params, iteration_number)
        result = await self.wait_for_result('final_logits_backward', task_id, iteration_number)
        await self.update_sot_all('final_logits_backward', result, iteration_number=iteration_number)
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
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.sot_url}/update_state", json=params) as response:
                if response.status != 200:  # Use 'status' instead of 'status_code'
                    logging.error(f"Failed to update SOT for {tensor_name}: {await response.text()}")
                else:
                    logging.info(f"Updated SOT for {tensor_name} with result: {result}")

    async def update_sot_all(self, task_type, result, layer_idx=None, iteration_number=None):
        if task_type == 'embed_backward':
            tensor_name = 'embed'
        elif task_type == 'final_logits_backward':
            tensor_name = 'final_logits'
        else:
            tensor_name = f'layer_{layer_idx}'
        await self.update_sot(tensor_name, result)

    async def get_batch_and_targets_url(self):
        url = os.path.join(self.sot_url, 'get_batch')
        async with aiohttp.ClientSession() as session:
            async with session.post(url) as response:
                response_json = await response.json()
                return response_json['batch_url'], response_json['targets_url']

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Master process for task submission")
    parser.add_argument('--rpc_url', type=str, required=True, help="RPC URL for Ethereum node")
    parser.add_argument('--private_key', type=str, required=True, help="Private key for Ethereum account")
    parser.add_argument('--sot_url', type=str, required=True, help="Source of Truth URL")
    parser.add_argument('--subnet_addresses', type=str, required=True, help="Path to subnet addresses JSON file")
    parser.add_argument('--detailed_logs', action='store_true', help="Enable detailed logs")
    parser.add_argument('--max_simultaneous_iterations', type=int, default=1, help="Maximum number of simultaneous iterations")

    args = parser.parse_args()

    with open(args.subnet_addresses, 'r') as file:
        subnet_addresses = json.load(file)

    master = Master(args.rpc_url, args.private_key, args.sot_url, subnet_addresses, detailed_logs=args.detailed_logs)
    
    plt.show(block=True)
