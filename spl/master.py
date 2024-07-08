import json
import logging
import time
import requests
from web3 import Web3
from web3.middleware import geth_poa_middleware
from web3.exceptions import ContractCustomError, TransactionNotFound
from common import load_contracts, handle_contract_custom_error, TaskStatus, Vote, PoolState, Task, get_learning_hyperparameters, transact_with_contract_function
from io import BytesIO
import torch
import os

logging.basicConfig(level=logging.INFO)

class Master:
    def __init__(self, rpc_url, private_key, sot_url, subnet_addresses, detailed_logs=False):
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
        self.account = self.web3.eth.account.from_key(private_key)
        self.sot_url = sot_url
        self.subnet_addresses = subnet_addresses
        self.abis, self.contracts, self.error_selectors = load_contracts(self.web3, subnet_addresses)

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
        self.vrf_coordinator_address = self.pool.functions.vrfCoordinator().call()
        self.vrf_coordinator = self.web3.eth.contract(address=self.vrf_coordinator_address, abi=self.abis['MockVRFCoordinator'])

        if detailed_logs:
            logging.getLogger().setLevel(logging.DEBUG)

    def approve_token(self, token_address, spender_address, amount):
        token_contract = self.web3.eth.contract(address=token_address, abi=self.abis['ERC20'])
        receipt = transact_with_contract_function(self.web3, token_contract, 'approve', self.account._private_key, spender_address, amount, gas=100000)
        logging.info(f"Approved token transaction receipt: {receipt}")

    def submit_task(self, task_type, params):
        try:
            if task_type not in self.contracts:
                raise ValueError(f"No contract loaded for task type {task_type}")

            logging.info(f"Submitting task of type {task_type} with params: {params}")
            encoded_params = json.dumps(params).encode('utf-8')

            fee = self.contracts[task_type].functions.calculateFee(0).call()
            token_address = self.contracts[task_type].functions.token().call()
            spender_address = self.contracts[task_type].address
            self.approve_token(token_address, spender_address, fee)

            receipt = transact_with_contract_function(self.web3, self.contracts[task_type], 'submitTaskRequest', self.account._private_key, encoded_params, gas=1000000)
            logging.info(f"submitTaskRequest transaction receipt: {receipt}")

            logs = self.contracts[task_type].events.TaskRequestSubmitted().process_receipt(receipt)
            if not logs:
                raise ValueError("No TaskRequestSubmitted event found in the receipt")

            task_id = logs[0]['args']['taskId']
            block_number = receipt['blockNumber']
            logging.info(f"Task submitted successfully. Task ID: {task_id}, Block number: {block_number}")

            selection_id = self.submit_selection_req()
            logging.info(f"Selection ID: {selection_id}")

            vrf_request_id = self.pool.functions.vrfRequestId().call()
            self.fulfill_random_words(vrf_request_id)
            self.select_solver(task_type, task_id)
            self.remove_solver_stake(task_type, task_id)

            return task_id, block_number
        except ContractCustomError as e:
            handle_contract_custom_error(self.web3, self.error_selectors, e)
        except Exception as e:
            logging.error(f"Error submitting task: {e}")
            raise

    def submit_selection_req(self):
        try:
            if self.pool.functions.state().call() != PoolState.Unlocked.value:
                return self.pool.functions.currentSelectionId().call()

            self.wait_for_state_change(PoolState.Unlocked.value)
            logging.info("Submitting selection request")

            receipt = transact_with_contract_function(self.web3, self.pool, 'submitSelectionReq', self.account._private_key, gas=500000)
            logging.info(f"submitSelectionReq transaction receipt: {receipt}")

            unlocked_min_period = self.pool.functions.UNLOCKED_MIN_PERIOD().call()
            last_state_change_time = self.pool.functions.lastStateChangeTime().call()
            current_time = time.time()
            remaining_time = (last_state_change_time + unlocked_min_period) - current_time

            if remaining_time > 0:
                logging.info(f"Waiting for {remaining_time} seconds until UNLOCKED_MIN_PERIOD is over")
                time.sleep(remaining_time)

            logs = self.pool.events.SelectionRequested().process_receipt(receipt)
            if not logs:
                raise ValueError("No SelectionRequested event found in the receipt")

            return logs[0]['args']['selectionId']
        except Exception as e:
            logging.error(f"Error submitting selection request: {e}")
            raise

    def trigger_lock_global_state(self):
        unlocked_min_period = self.pool.functions.UNLOCKED_MIN_PERIOD().call()
        last_state_change_time = self.pool.functions.lastStateChangeTime().call()
        current_time = time.time()
        remaining_time = (last_state_change_time + unlocked_min_period) - current_time

        if remaining_time > 0:
            logging.info(f"Waiting for {remaining_time} seconds until UNLOCKED_MIN_PERIOD is over")
            time.sleep(remaining_time)

        try:
            receipt = transact_with_contract_function(self.web3, self.pool, 'lockGlobalState', self.account._private_key, gas=500000)
            logging.info(f"lockGlobalState transaction receipt: {receipt}")
        except Exception as e:
            logging.error(f"Error triggering lock global state: {e}")
            raise

    def trigger_remove_global_lock(self):
        selections_finalizing_min_period = self.pool.functions.SELECTIONS_FINALIZING_MIN_PERIOD().call()
        last_state_change_time = self.pool.functions.lastStateChangeTime().call()
        current_time = time.time()
        remaining_time = (last_state_change_time + selections_finalizing_min_period) - current_time

        if remaining_time > 0:
            logging.info(f"Waiting for {remaining_time} seconds until SELECTIONS_FINALIZING_MIN_PERIOD is over")
            time.sleep(remaining_time)

        try:
            receipt = transact_with_contract_function(self.web3, self.pool, 'removeGlobalLock', self.account._private_key, gas=500000)
            logging.info(f"removeGlobalLock transaction receipt: {receipt}")
        except Exception as e:
            logging.error(f"Error triggering remove global lock: {e}")
            raise


    def wait_for_state_change(self, target_state):
        while True:
            current_state = PoolState(self.pool.functions.state().call())
            logging.info(f"Current pool state: {current_state.name}, target state: {PoolState(target_state).name}")

            if current_state == PoolState(target_state):
                break

            if current_state == PoolState.Unlocked:
                logging.info("Triggering lockGlobalState to change state to Locked")
                self.trigger_lock_global_state()
            elif current_state == PoolState.Locked:
                logging.info("Waiting for state to change from Locked to SelectionsFinalizing (handled by fulfillRandomWords)")
            elif current_state == PoolState.SelectionsFinalizing:
                logging.info("Triggering removeGlobalLock to change state to Unlocked")
                self.trigger_remove_global_lock()
            else:
                logging.info(f"Waiting for the pool state to change to {PoolState(target_state).name}")
                time.sleep(5)

    def fulfill_random_words(self, vrf_request_id):
        try:
            receipt = transact_with_contract_function(self.web3, self.vrf_coordinator, 'fulfillRandomWords', self.account._private_key, vrf_request_id, gas=500000)
            logging.info(f"fulfillRandomWords transaction receipt: {receipt}")
        except Exception as e:
            logging.error(f"Error fulfilling random words: {e}")
            raise

    def select_solver(self, task_type, task_id):
        try:
            logging.info(f"Selecting solver for task ID: {task_id}")

            self.wait_for_state_change(PoolState.SelectionsFinalizing.value)
            receipt = transact_with_contract_function(self.web3, self.contracts[task_type], 'selectSolver', self.account._private_key, task_id, gas=1000000)
            logging.info(f"selectSolver transaction receipt: {receipt}")
        except Exception as e:
            logging.error(f"Error selecting solver: {e}")
            raise

    def remove_solver_stake(self, task_type, task_id):
        try:
            logging.info(f"Removing solver stake for task ID: {task_id}")

            self.wait_for_state_change(PoolState.Unlocked.value)
            receipt = transact_with_contract_function(self.web3, self.contracts[task_type], 'removeSolverStake', self.account._private_key, task_id, gas=1000000)
            logging.info(f"removeSolverStake transaction receipt: {receipt}")
        except Exception as e:
            logging.error(f"Error removing solver stake: {e}")
            raise

    def log_transaction_failure(self, receipt):
        try:
            tx = self.web3.eth.get_transaction(receipt['transactionHash'])
            error_message = self.web3.eth.call({
                'to': tx['to'],
                'data': tx['input']
            }, receipt['blockNumber'])
            decoded_error_message = self.web3.codec.decode_abi(['string'], error_message)
            logging.error(f"Transaction failed with error message: {decoded_error_message}")
        except (TransactionNotFound, ValueError) as e:
            logging.error(f"Error retrieving transaction details: {e}")

    def get_task_result(self, task_type, task_id):
        try:
            task_tuple = self.contracts[task_type].functions.getTask(task_id).call()
            task = Task(*task_tuple)
            logging.info(f"{task_type} Task status: {task.status}")
            logging.info(f"Expected status: {TaskStatus.SolutionSubmitted.value}")
            if task.status == TaskStatus.SolutionSubmitted.value:
                return json.loads(task.postedSolution.decode('utf-8'))
            return None
        except Exception as e:
            logging.error(f"Error getting task result for {task_type} with task ID {task_id}: {e}")
            return None

    def main(self):
        logging.info("Starting main process")
        model_params = self.get_latest_model_params()

        batch_url = self.get_batch_url()

        logging.info("Starting embed forward task")
        embed_result = self.handle_embed_forward(model_params, batch_url)

        layer_inputs_url = [embed_result['result_url']]
        for layer_idx in range(model_params['n_layers']):
            logging.info(f"Starting forward task for layer {layer_idx}")
            layer_result = self.handle_layer_forward(layer_idx, layer_inputs_url[-1], model_params)
            layer_inputs_url.append(layer_result['result_url'])

        logging.info("Starting final logits forward task")
        final_logits_result = self.handle_final_logits_forward(layer_inputs_url[-1])

        logging.info("Starting loss computation task")
        loss_result = self.handle_loss_computation(final_logits_result['result_url'])

        error_url = loss_result['result_url']

        logging.info("Starting final logits backward task")
        final_logits_backward_result = self.handle_final_logits_backward(error_url, layer_inputs_url[-1], model_params)
        self.update_sot('final_logits_backward', final_logits_backward_result, final_logits_backward_result['block_number'])

        for layer_idx in reversed(range(model_params['n_layers'])):
            logging.info(f"Starting backward task for layer {layer_idx}")
            layer_result = self.handle_layer_backward(layer_idx, error_url, layer_inputs_url[layer_idx], model_params)
            error_url = layer_result['error_output_url']
            self.update_sot(f'backward_layer_{layer_idx}', layer_result, layer_result['block_number'])

        logging.info("Starting embed backward task")
        embed_backward_result = self.handle_embed_backward(error_url, batch_url)
        self.update_sot('embed_backward', embed_backward_result, embed_backward_result['block_number'])

    def get_latest_model_params(self):
        response = requests.get(f"{self.sot_url}/latest_model_params")
        return response.json()

    def handle_embed_forward(self, model_params, batch_url):
        task_params = {'batch_url': batch_url, 'model_params': model_params}
        task_id, block_number = self.submit_task('embed', task_params)
        result = self.wait_for_result('embed', task_id)
        result['batch_url'] = batch_url
        result['block_number'] = block_number
        return result

    def handle_layer_forward(self, layer_idx, inputs_url, model_params):
        task_type = f'forward_layer_{layer_idx}'
        task_params = {'layer_idx': layer_idx, 'inputs_url': inputs_url, 'model_params': model_params}
        task_id, block_number = self.submit_task(task_type, task_params)
        result = self.wait_for_result(task_type, task_id)
        result['block_number'] = block_number
        return result

    def handle_final_logits_forward(self, inputs_url):
        task_params = {'inputs_url': inputs_url}
        task_id, block_number = self.submit_task('final_logits', task_params)
        result = self.wait_for_result('final_logits', task_id)
        result['block_number'] = block_number
        return result

    def handle_loss_computation(self, logits_url):
        targets_url = self.get_targets_url()
        task_params = {'logits_url': logits_url, 'targets_url': targets_url}
        task_id, block_number = self.submit_task('loss', task_params)
        result = self.wait_for_result('loss', task_id)
        result['block_number'] = block_number
        return result

    def handle_layer_backward(self, layer_idx, error_url, inputs_url, model_params):
        learning_params = get_learning_hyperparameters()
        task_type = f'backward_layer_{layer_idx}'
        task_params = {
            'layer_idx': layer_idx,
            'error_url': error_url,
            'inputs_url': inputs_url,
            **learning_params
        }
        task_id, block_number = self.submit_task(task_type, task_params)
        result = self.wait_for_result(task_type, task_id)
        result['block_number'] = block_number
        self.update_sot_with_sparse(task_type, result, block_number)
        return result

    def handle_final_logits_backward(self, error_url, inputs_url, model_params):
        learning_params = get_learning_hyperparameters()
        task_params = {
            'error_url': error_url,
            'inputs_url': inputs_url,
            **learning_params
        }
        task_id, block_number = self.submit_task('final_logits_backward', task_params)
        result = self.wait_for_result('final_logits_backward', task_id)
        result['block_number'] = block_number
        self.update_sot_with_sparse('final_logits_backward', result, block_number)
        return result

    def handle_embed_backward(self, error_url, batch_url):
        learning_params = get_learning_hyperparameters()
        task_params = {
            'error_url': error_url,
            'batch_url': batch_url,
            **learning_params
        }
        task_id, block_number = self.submit_task('embed_backward', task_params)
        result = self.wait_for_result('embed_backward', task_id)
        result['block_number'] = block_number
        self.update_sot_with_sparse('embed_backward', result, block_number)
        return result

    def wait_for_result(self, task_type, task_id):
        while True:
            result = self.get_task_result(task_type, task_id)
            if result is not None:
                return result
            time.sleep(5)

    def update_sot(self, task_type, result, block_number):
        response = requests.post(f"{self.sot_url}/update_state", json={'task_type': task_type, 'result_url': result['grads_url'], 'block_number': block_number})
        if response.status_code != 200:
            logging.error(f"Failed to update SOT for {task_type}: {response.text}")
        else:
            logging.info(f"Updated SOT for {task_type} with result: {result}")

    def update_sot_with_sparse(self, task_type, result, block_number):
        self.update_sot(task_type, result, block_number)
        self.update_adam_state(task_type, result['adam_m_url'], result['adam_v_url'], block_number)

    def update_adam_state(self, task_type, adam_m_url, adam_v_url, block_number):
        response = requests.post(f"{self.sot_url}/update_state", json={'task_type': f'{task_type}_adam_m', 'result_url': adam_m_url, 'block_number': block_number})
        if response.status_code != 200:
            logging.error(f"Failed to update Adam state for {task_type}: {response.text}")
        else:
            logging.info(f"Updated Adam state for {task_type}")
        response = requests.post(f"{self.sot_url}/update_state", json={'task_type': f'{task_type}_adam_v', 'result_url': adam_v_url, 'block_number': block_number})
        if response.status_code != 200:
            logging.error(f"Failed to update Adam state for {task_type}: {response.text}")
        else:
            logging.info(f"Updated Adam state for {task_type}")

    def get_batch_url(self):
        url = os.path.join(self.sot_url, 'get_batch')
        response = requests.get(url)
        return response.json()['batch_url']

    def get_targets_url(self):
        url = os.path.join(self.sot_url, 'get_targets')
        response = requests.get(url)
        return response.json()['targets_url']

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Master process for task submission")
    parser.add_argument('--rpc_url', type=str, required=True, help="RPC URL for Ethereum node")
    parser.add_argument('--private_key', type=str, required=True, help="Private key for Ethereum account")
    parser.add_argument('--sot_url', type=str, required=True, help="Source of Truth URL")
    parser.add_argument('--subnet_addresses', type=str, required=True, help="Path to subnet addresses JSON file")
    parser.add_argument('--detailed_logs', action='store_true', help="Enable detailed logs")

    args = parser.parse_args()

    with open(args.subnet_addresses, 'r') as file:
        subnet_addresses = json.load(file)

    master = Master(args.rpc_url, args.private_key, args.sot_url, subnet_addresses, detailed_logs=args.detailed_logs)
    master.main()
