import json
import logging
import time
import requests
import os
from web3 import Web3
from web3.middleware import geth_poa_middleware
from web3.exceptions import ContractCustomError
from eth_utils import function_signature_to_4byte_selector
import struct

logging.basicConfig(level=logging.INFO)

class Master:
    def __init__(self, rpc_url, private_key, sot_url, subnet_addresses):
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
        self.account = self.web3.eth.account.from_key(private_key)
        self.sot_url = sot_url
        self.subnet_addresses = subnet_addresses
        self.load_contracts()

    def load_contracts(self):
        self.contracts = {}
        abi_dir = 'abis'
        self.abis = {}

        for root, _, files in os.walk(abi_dir):
            for file in files:
                if file.endswith('.json'):
                    with open(os.path.join(root, file), 'r') as abi_file:
                        contract_name = os.path.splitext(file)[0]
                        self.abis[contract_name] = json.load(abi_file).get('abi', [])
        
        for task, address in self.subnet_addresses.items():
            contract_name = 'SubnetManager'  # Assuming all tasks use SubnetManager ABI for simplicity
            if contract_name in self.abis:
                self.contracts[task] = self.web3.eth.contract(address=address, abi=self.abis[contract_name])
                logging.info(f"Loaded contract for {task} with address {address}")
            else:
                logging.error(f"ABI for {contract_name} not found")

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

            tx = self.contracts[task_type].functions.submitTaskRequest(encoded_params).transact({
                'from': self.account.address,
                'gas': 3000000  # Increase the gas limit
            })
            receipt = self.web3.eth.wait_for_transaction_receipt(tx)
            task_id = self.contracts[task_type].events.TaskRequestSubmitted().process_receipt(receipt)[0]['args']['taskId']
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
            # Attempt to decode using common error patterns
            function_selector = error_bytes[:4]
            error_data = error_bytes[4:]

            # Example of decoding a bytes32 and uint256 from the error data
            if len(error_data) == 64:
                decoded_data = struct.unpack('32s32s', error_data)
                decoded_message = f"Selector: {function_selector.hex()}, Decoded Data: {decoded_data}"
            else:
                decoded_message = f"Selector: {function_selector.hex()}, Raw Data: {error_data.hex()}"
            
            return decoded_message
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
        return self.wait_for_result('final_logits', task_id)

    def handle_loss_computation(self, logits_url):
        targets_url = self.get_targets_url()
        task_params = {'logits_url': logits_url, 'targets_url': targets_url}
        task_id = self.submit_task('loss', task_params)
        return self.wait_for_result('loss', task_id)

    def handle_layer_backward(self, layer_idx, error_url, model_params):
        task_type = f'backward_layer_{layer_idx}'
        task_params = {'layer_idx': layer_idx, 'error_url': error_url, 'model_params': model_params}
        task_id = self.submit_task(task_type, task_params)
        result = self.wait_for_result(task_type, task_id)
        self.update_adam_state(task_type, result['adam_m_url'], result['adam_v_url'])
        return result

    def handle_final_logits_backward(self, error_url, logits_url, model_params):
        task_params = {'error_url': error_url, 'logits_url': logits_url, 'model_params': model_params}
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
