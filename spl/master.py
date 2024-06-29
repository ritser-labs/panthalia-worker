import json
import logging
import time
import requests
from web3 import Web3
from web3.middleware import geth_poa_middleware
from web3.exceptions import ContractCustomError, TransactionNotFound
from common import load_contracts, handle_contract_custom_error
from io import BytesIO
import torch

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
        nonce = self.web3.eth.get_transaction_count(self.account.address)
        gas_price = self.web3.eth.gas_price

        approve_txn = token_contract.functions.approve(spender_address, amount).build_transaction({
            'chainId': self.web3.eth.chain_id,
            'gas': 100000,
            'gasPrice': gas_price,
            'nonce': nonce
        })

        signed_approve_txn = self.web3.eth.account.sign_transaction(approve_txn, private_key=self.account._private_key)
        tx_hash = self.web3.eth.send_raw_transaction(signed_approve_txn.rawTransaction)
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        logging.info(f"Token approval transaction receipt: {receipt}")

    def submit_task(self, task_type, params):
        try:
            if task_type not in self.contracts:
                raise ValueError(f"No contract loaded for task type {task_type}")

            logging.info(f"Submitting task of type {task_type} with params: {params}")
            encoded_params = json.dumps(params).encode('utf-8')
            logging.info(f"Encoded params: {encoded_params}")

            if not hasattr(self.contracts[task_type].functions, 'submitTaskRequest'):
                raise ValueError(f"'submitTaskRequest' method not found in contract for task type {task_type}")

            placeholder_task_id = 0
            fee = self.contracts[task_type].functions.calculateFee(placeholder_task_id).call()

            subnet_manager_contract = self.contracts[task_type]
            token_address = subnet_manager_contract.functions.token().call()
            spender_address = self.contracts[task_type].address
            self.approve_token(token_address, spender_address, fee)

            nonce = self.web3.eth.get_transaction_count(self.account.address)
            gas_price = self.web3.eth.gas_price
            transaction = self.contracts[task_type].functions.submitTaskRequest(encoded_params).build_transaction({
                'chainId': self.web3.eth.chain_id,
                'gas': 1000000,
                'gasPrice': gas_price,
                'nonce': nonce
            })

            signed_txn = self.web3.eth.account.sign_transaction(transaction, private_key=self.account._private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            logging.info(f"Transaction receipt: {receipt}")

            if receipt['status'] == 0:
                self.log_transaction_failure(receipt)
                raise ValueError(f"Transaction failed with status 0. Transaction hash: {receipt['transactionHash']}, block number: {receipt['blockNumber']}")

            logs = self.contracts[task_type].events.TaskRequestSubmitted().process_receipt(receipt)
            logging.info(f"Event logs: {logs}")

            if not logs:
                raise ValueError("No TaskRequestSubmitted event found in the receipt")

            task_id = logs[0]['args']['taskId']
            block_number = receipt['blockNumber']
            logging.info(f"Task submitted successfully. Task ID: {task_id}, Block number: {block_number}")

            # Call submitSelectionReq and wait for UNLOCKED_MIN_PERIOD
            selection_id = self.submit_selection_req()
            logging.info(f"Selection ID: {selection_id}")

            # Retrieve vrfRequestId from Pool contract
            vrf_request_id = self.pool.functions.vrfRequestId().call()

            # Call fulfillRandomWords function
            self.fulfill_random_words(vrf_request_id)

            # Call select_solver function
            self.select_solver(task_type, task_id)

            return task_id, block_number
        except ContractCustomError as e:
            handle_contract_custom_error(self.web3, self.error_selectors, e)
        except Exception as e:
            logging.error(f"Error submitting task: {e}")
            raise

    def submit_selection_req(self):
        try:
            # Wait until the pool state is Unlocked and the unlocked minimum period is over
            while self.pool.functions.state().call() != 0 or self.pool.functions.lastStateChangeTime().call() + self.pool.functions.UNLOCKED_MIN_PERIOD().call() >= time.time():
                logging.info("Waiting for pool state to be Unlocked and UNLOCKED_MIN_PERIOD to be over")
                time.sleep(5)

            nonce = self.web3.eth.get_transaction_count(self.account.address)
            gas_price = self.web3.eth.gas_price
            transaction = self.pool.functions.submitSelectionReq().build_transaction({
                'chainId': self.web3.eth.chain_id,
                'gas': 500000,
                'gasPrice': gas_price,
                'nonce': nonce
            })

            signed_txn = self.web3.eth.account.sign_transaction(transaction, private_key=self.account._private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            logging.info(f"submitSelectionReq transaction receipt: {receipt}")

            if receipt['status'] == 0:
                self.log_transaction_failure(receipt)
                raise ValueError(f"submitSelectionReq transaction failed with status 0. Transaction hash: {receipt['transactionHash']}, block number: {receipt['blockNumber']}")

            # Calculate remaining time for UNLOCKED_MIN_PERIOD
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

            selection_id = logs[0]['args']['selectionId']
            return selection_id
        except Exception as e:
            logging.error(f"Error submitting selection request: {e}")
            raise


    def fulfill_random_words(self, vrf_request_id):
        try:
            nonce = self.web3.eth.get_transaction_count(self.account.address)
            gas_price = self.web3.eth.gas_price
            transaction = self.vrf_coordinator.functions.fulfillRandomWords(vrf_request_id).build_transaction({
                'chainId': self.web3.eth.chain_id,
                'gas': 500000,
                'gasPrice': gas_price,
                'nonce': nonce
            })

            signed_txn = self.web3.eth.account.sign_transaction(transaction, private_key=self.account._private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            logging.info(f"fulfillRandomWords transaction receipt: {receipt}")

            if receipt['status'] == 0:
                self.log_transaction_failure(receipt)
                raise ValueError(f"fulfillRandomWords transaction failed with status 0. Transaction hash: {receipt['transactionHash']}, block number: {receipt['blockNumber']}")
        except Exception as e:
            logging.error(f"Error fulfilling random words: {e}")
            raise

    def select_solver(self, task_type, task_id):
        try:
            if task_type not in self.contracts:
                raise ValueError(f"No contract loaded for task type {task_type}")

            logging.info(f"Selecting solver for task ID: {task_id}")

            nonce = self.web3.eth.get_transaction_count(self.account.address)
            gas_price = self.web3.eth.gas_price
            transaction = self.contracts[task_type].functions.selectSolver(task_id).build_transaction({
                'chainId': self.web3.eth.chain_id,
                'gas': 1000000,
                'gasPrice': gas_price,
                'nonce': nonce
            })

            signed_txn = self.web3.eth.account.sign_transaction(transaction, private_key=self.account._private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            logging.info(f"Select solver transaction receipt: {receipt}")

            if receipt['status'] == 0:
                self.log_transaction_failure(receipt)
                raise ValueError(f"Select solver transaction failed with status 0. Transaction hash: {receipt['transactionHash']}, block number: {receipt['blockNumber']}")
        except Exception as e:
            logging.error(f"Error selecting solver: {e}")
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
            task = self.contracts[task_type].functions.getTask(task_id).call()
            if task[0] == 4:  # Assuming TaskStatus.ResolvedCorrect is 4
                result_data = json.loads(task[6].decode('utf-8'))  # Decode the result field
                return result_data
            return None
        except Exception as e:
            logging.error(f"Error getting task result for {task_type} with task ID {task_id}: {e}")
            return None

    def main(self):
        logging.info("Starting main process")
        model_params = self.get_latest_model_params()

        logging.info("Starting embed forward task")
        embed_result = self.handle_embed_forward(model_params)
        self.update_sot('embed', embed_result, embed_result['block_number'])

        layer_inputs_url = embed_result['result_url']
        for layer_idx in range(model_params['n_layers']):
            logging.info(f"Starting forward task for layer {layer_idx}")
            layer_result = self.handle_layer_forward(layer_idx, layer_inputs_url, model_params)
            layer_inputs_url = layer_result['result_url']
            self.update_sot(f'forward_layer_{layer_idx}', layer_result, layer_result['block_number'])

        logging.info("Starting final logits forward task")
        final_logits_result = self.handle_final_logits_forward(layer_inputs_url)
        self.update_sot('final_logits', final_logits_result, final_logits_result['block_number'])

        logging.info("Starting loss computation task")
        loss_result = self.handle_loss_computation(final_logits_result['result_url'])
        self.update_sot('loss', loss_result, loss_result['block_number'])

        error_url = loss_result['result_url']
        for layer_idx in reversed(range(model_params['n_layers'])):
            logging.info(f"Starting backward task for layer {layer_idx}")
            layer_result = self.handle_layer_backward(layer_idx, error_url, model_params)
            error_url = layer_result['error_output_url']
            self.update_sot(f'backward_layer_{layer_idx}', layer_result, layer_result['block_number'])

        logging.info("Starting final logits backward task")
        final_logits_backward_result = self.handle_final_logits_backward(error_url, final_logits_result['result_url'], model_params)
        self.update_sot('final_logits_backward', final_logits_backward_result, final_logits_backward_result['block_number'])

        logging.info("Starting embed backward task")
        embed_backward_result = self.handle_embed_backward(error_url, embed_result['batch_url'])
        self.update_sot('embed_backward', embed_backward_result, embed_backward_result['block_number'])

    def get_latest_model_params(self):
        response = requests.get(f"{self.sot_url}/latest_model_params")
        return response.json()

    def handle_embed_forward(self, model_params):
        batch_url = self.get_batch_url()
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

    def handle_layer_backward(self, layer_idx, error_url, model_params):
        task_type = f'backward_layer_{layer_idx}'
        task_params = {'layer_idx': layer_idx, 'error_url': error_url, 'model_params': model_params}
        task_id, block_number = self.submit_task(task_type, task_params)
        result = self.wait_for_result(task_type, task_id)
        result['block_number'] = block_number
        self.update_sot_with_sparse(task_type, result, block_number)
        return result

    def handle_final_logits_backward(self, error_url, inputs_url, model_params):
        task_params = {'error_url': error_url, 'inputs_url': inputs_url, 'model_params': model_params}
        task_id, block_number = self.submit_task('final_logits_backward', task_params)
        result = self.wait_for_result('final_logits_backward', task_id)
        result['block_number'] = block_number
        self.update_sot_with_sparse('final_logits_backward', result, block_number)
        return result

    def handle_embed_backward(self, error_url, batch_url):
        task_params = {'error_url': error_url, 'batch_url': batch_url}
        task_id, block_number = self.submit_task('embed_backward', task_params)
        result = self.wait_for_result('embed_backward', task_id)
        result['block_number'] = block_number
        self.update_sot_with_sparse('embed_backward', result, block_number)
        return result

    def wait_for_result(self, task_type, task_id):
        while True:
            result = self.get_task_result(task_type, task_id)
            if result:
                return result
            time.sleep(5)

    def update_sot(self, task_type, result, block_number):
        response = requests.post(f"{self.sot_url}/update_state", json={'task_type': task_type, 'result_url': result['result_url'], 'block_number': block_number})
        if response.status_code != 200:
            logging.error(f"Failed to update SOT for {task_type}: {response.text}")
        else:
            logging.info(f"Updated SOT for {task_type} with result: {result}")

    def update_sot_with_sparse(self, task_type, result, block_number):
        self.update_sot(task_type, result, block_number)
        self.update_adam_state(task_type, result['adam_m_url'], result['adam_v_url'], block_number)

    def update_adam_state(self, task_type, adam_m_url, adam_v_url, block_number):
        response = requests.post(f"{self.sot_url}/update_adam", json={'task_type': task_type, 'adam_m': adam_m_url, 'adam_v': adam_v_url, 'block_number': block_number})
        if response.status_code != 200:
            logging.error(f"Failed to update Adam state for {task_type}: {response.text}")
        else:
            logging.info(f"Updated Adam state for {task_type}")

    def get_batch_url(self):
        response = requests.get(f"{self.sot_url}/get_batch")
        return response.json()['batch_url']

    def get_targets_url(self):
        response = requests.get(f"{self.sot_url}/get_targets")
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
