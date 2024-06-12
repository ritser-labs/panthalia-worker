import json
import logging
import time
import requests
import os
from web3 import Web3
from web3.middleware import geth_poa_middleware
from web3.exceptions import ContractCustomError, TransactionNotFound
from common import load_contracts

logging.basicConfig(level=logging.INFO)

class Master:
    def __init__(self, rpc_url, private_key, sot_url, subnet_addresses):
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
        self.account = self.web3.eth.account.from_key(private_key)
        self.sot_url = sot_url
        self.subnet_addresses = subnet_addresses
        self.abis, self.contracts, self.error_selectors = load_contracts(self.web3, subnet_addresses)

        if 'SubnetManager' not in self.contracts:
            raise ValueError("SubnetManager contract not found. Please check the subnet_addresses configuration.")

    def approve_token(self, token_address, spender_address, amount):
        token_contract = self.web3.eth.contract(address=token_address, abi=self.abis['SubnetManager'])
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

            # Validate the presence of submitTaskRequest in ABI
            if not hasattr(self.contracts[task_type].functions, 'submitTaskRequest'):
                raise ValueError(f"'submitTaskRequest' method not found in contract for task type {task_type}")

            # Use a placeholder task ID for fee calculation if necessary
            placeholder_task_id = 0
            fee = self.contracts[task_type].functions.calculateFee(placeholder_task_id).call()

            # Perform token approval using the 'SubnetManager' contract instance
            subnet_manager_contract = self.contracts[task_type]
            token_address = subnet_manager_contract.functions.token().call()
            spender_address = self.contracts[task_type].address
            self.approve_token(token_address, spender_address, fee)

            # Prepare the transaction
            nonce = self.web3.eth.get_transaction_count(self.account.address)
            gas_price = self.web3.eth.gas_price
            transaction = self.contracts[task_type].functions.submitTaskRequest(encoded_params).build_transaction({
                'chainId': self.web3.eth.chain_id,
                'gas': 10000000,
                'gasPrice': gas_price,
                'nonce': nonce
            })

            # Sign the transaction
            signed_txn = self.web3.eth.account.sign_transaction(transaction, private_key=self.account._private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            logging.info(f"Transaction receipt: {receipt}")

            if receipt['status'] == 0:
                try:
                    # Try to retrieve the revert reason
                    tx = self.web3.eth.get_transaction(receipt['transactionHash'])
                    error_message = self.web3.eth.call({
                        'to': tx['to'],
                        'data': tx['input']
                    }, receipt['blockNumber'])
                    logging.error(f"Error message: {error_message}")
                except (TransactionNotFound, ValueError) as e:
                    logging.error(f"Error retrieving transaction details: {e}")
                raise ValueError(f"Transaction failed: {receipt}")

            # Process receipt to get task ID
            logs = self.contracts[task_type].events.TaskRequestSubmitted().process_receipt(receipt)
            logging.info(f"Event logs: {logs}")

            if not logs:
                raise ValueError("No TaskRequestSubmitted event found in the receipt")

            task_id = logs[0]['args']['taskId']
            logging.info(f"Task submitted successfully. Task ID: {task_id}")
            return task_id
        except ContractCustomError as e:
            logging.error(f"Custom error encountered: {e.data}")
            logging.error(f"Raw error data: {e.data}")
            try:
                error_bytes = bytes.fromhex(e.data[2:])
                logging.error(f"Error bytes: {error_bytes}")
                logging.error(f"Error bytes as integers: {list(error_bytes)}")
                decoded_message = self.decode_custom_error(error_bytes)
                logging.error(f"Decoded error message: {decoded_message}")
            except Exception as decode_err:
                logging.error(f"Failed to decode error data: {e.data}. Error: {decode_err}")
            raise
        except Exception as e:
            logging.error(f"Error submitting task: {e}")
            raise

    def decode_custom_error(self, error_bytes):
        try:
            # Extract selector
            selector = '0x' + error_bytes[:4].hex().lower()  # Ensure consistent case for matching and add '0x' prefix
            data = error_bytes[4:]

            logging.info(f"Selector: {selector}, Data: {data.hex()}")
            logging.info(f"Error Selectors: {self.error_selectors.keys()}")

            if selector in self.error_selectors:
                error_info = self.error_selectors[selector]
                error_name = error_info['name']
                inputs = error_info['inputs']
                decoded_params = self.web3.codec.decode([input['type'] for input in inputs], data)
                param_str = ', '.join(f"{input['name']}: {value}" for input, value in zip(inputs, decoded_params))
                return f"Error {error_name}: {param_str}"

            return f"Unknown error with selector {selector} and data {data.hex()}"
        except Exception as e:
            logging.error(f"Error decoding message chunk: {e}")
            raise

    def get_task_result(self, task_type, task_id):
        task = self.contracts[task_type].functions.getTask(task_id).call()
        if task[0] == 4:  # Assuming TaskStatus.SolutionSubmitted
            return json.loads(task[6].decode('utf-8'))  # Assuming task.postedSolution
        return None

    def main(self):
        model_params = self.get_latest_model_params()

        # Forward pass for embedding
        embed_result = self.handle_embed_forward(model_params)
        layer_inputs_url = embed_result['result_url']
        self.update_sot('embed', embed_result)

        # Forward pass for each transformer layer
        for layer_idx in range(model_params['n_layers']):
            layer_inputs_url = self.handle_layer_forward(layer_idx, layer_inputs_url, model_params)
            self.update_sot(f'forward_layer_{layer_idx}', layer_inputs_url)

        # Final logits forward pass
        final_logits_result = self.handle_final_logits_forward(layer_inputs_url)
        self.update_sot('final_logits', final_logits_result)

        # Loss computation
        loss_result = self.handle_loss_computation(final_logits_result['result_url'])
        self.update_sot('loss', loss_result)

        # Backward pass for each transformer layer
        error_url = loss_result['result_url']
        for layer_idx in reversed(range(model_params['n_layers'])):
            error_url = self.handle_layer_backward(layer_idx, error_url, model_params)
            self.update_sot(f'backward_layer_{layer_idx}', error_url)

        # Backward pass for final logits
        final_logits_backward_result = self.handle_final_logits_backward(error_url, final_logits_result['result_url'], model_params)
        self.update_sot('final_logits_backward', final_logits_backward_result)

        # Backward pass for embedding
        embed_backward_result = self.handle_embed_backward(error_url, embed_result['batch_url'])
        self.update_sot('embed_backward', embed_backward_result)

    def get_latest_model_params(self):
        response = requests.get(f"{self.sot_url}/latest_model_params")
        return response.json()

    def handle_embed_forward(self, model_params):
        batch_url = self.get_batch_url()
        task_params = {'batch_url': batch_url, 'model_params': model_params}
        task_id = self.submit_task('embed', task_params)
        result = self.wait_for_result('embed', task_id)
        result['batch_url'] = batch_url
        return result

    def handle_layer_forward(self, layer_idx, inputs_url, model_params):
        task_type = f'forward_layer_{layer_idx}'
        task_params = {'layer_idx': layer_idx, 'inputs_url': inputs_url, 'model_params': model_params}
        task_id = self.submit_task(task_type, task_params)
        result = self.wait_for_result(task_type, task_id)
        return result

    def handle_final_logits_forward(self, inputs_url):
        task_params = {'inputs_url': inputs_url}
        task_id = self.submit_task('final_logits', task_params)
        result = self.wait_for_result('final_logits', task_id)
        return result

    def handle_loss_computation(self, logits_url):
        targets_url = self.get_targets_url()
        task_params = {'logits_url': logits_url, 'targets_url': targets_url}
        task_id = self.submit_task('loss', task_params)
        result = self.wait_for_result('loss', task_id)
        return result

    def handle_layer_backward(self, layer_idx, error_url, model_params):
        task_type = f'backward_layer_{layer_idx}'
        task_params = {'layer_idx': layer_idx, 'error_url': error_url, 'model_params': model_params}
        task_id = self.submit_task(task_type, task_params)
        result = self.wait_for_result(task_type, task_id)
        self.update_adam_state(task_type, result['adam_m_url'], result['adam_v_url'])
        return result['error_url']

    def handle_final_logits_backward(self, error_url, inputs_url, model_params):
        task_params = {'error_url': error_url, 'inputs_url': inputs_url, 'model_params': model_params}
        task_id = self.submit_task('final_logits_backward', task_params)
        result = self.wait_for_result('final_logits_backward', task_id)
        self.update_adam_state('final_logits_backward', result['adam_m_url'], result['adam_v_url'])
        return result

    def handle_embed_backward(self, error_url, batch_url):
        task_params = {'error_url': error_url, 'batch_url': batch_url}
        task_id = self.submit_task('embed_backward', task_params)
        result = self.wait_for_result('embed_backward', task_id)
        self.update_adam_state('embed_backward', result['adam_m_url'], result['adam_v_url'])
        return result

    def wait_for_result(self, task_type, task_id):
        while True:
            result = self.get_task_result(task_type, task_id)
            if result:
                return result
            time.sleep(1)

    def update_sot(self, task_type, result):
        response = requests.post(f"{self.sot_url}/update", json={'task_type': task_type, 'result': result})
        if response.status_code != 200:
            logging.error(f"Failed to update SOT for {task_type}: {response.text}")

    def update_adam_state(self, task_type, adam_m_url, adam_v_url):
        response = requests.post(f"{self.sot_url}/update_adam", json={'task_type': task_type, 'adam_m': adam_m_url, 'adam_v': adam_v_url})
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
    rpc_url = "http://localhost:8545"
    private_key = os.environ['PRIVATE_KEY']
    sot_url = os.environ['SOT_URL']
    with open(os.environ['SUBNET_ADDRESSES_JSON'], 'r') as file:
        subnet_addresses = json.load(file)

    master = Master(rpc_url, private_key, sot_url, subnet_addresses)
    master.main()
