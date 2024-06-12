import json
import logging
import time
import requests
from web3 import Web3
from web3.middleware import geth_poa_middleware

logging.basicConfig(level=logging.INFO)

class Master:
    def __init__(self, rpc_url, private_key, sot_url, subnet_addresses):
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
        self.account = self.web3.eth.account.from_key(private_key)
        self.sot_url = sot_url
        self.subnet_addresses = subnet_addresses
        self.contracts = {task: self.web3.eth.contract(address=address, abi=json.loads('YourContractABI')) for task, address in subnet_addresses.items()}

    def submit_task(self, task_type, params):
        tx = self.contracts[task_type].functions.submitTaskRequest(json.dumps(params).encode('utf-8')).transact({'from': self.account.address})
        receipt = self.web3.eth.wait_for_transaction_receipt(tx)
        task_id = self.contracts[task_type].events.TaskSubmitted().processReceipt(receipt)[0]['args']['taskId']
        return task_id

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

        # Backward pass for embedding
        self.handle_embed_backward(error_url, embed_result['batch_url'])
        self.update_sot('embed_backward', {'error_url': error_url})

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
        task_params = {'layer_idx': layer_idx, 'inputs_url': inputs_url, 'model_params': model_params}
        task_id = self.submit_task('forward', task_params)
        result = self.wait_for_result('forward', task_id)
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
        task_params = {'layer_idx': layer_idx, 'error_url': error_url, 'model_params': model_params}
        task_id = self.submit_task('backward', task_params)
        result = self.wait_for_result('backward', task_id)
        return result

    def handle_embed_backward(self, error_url, batch_url):
        task_params = {'error_url': error_url, 'batch_url': batch_url}
        task_id = self.submit_task('embed_backward', task_params)
        self.wait_for_result('embed_backward', task_id)

    def wait_for_result(self, task_type, task_id):
        while True:
            result = self.get_task_result(task_type, task_id)
            if result:
                return result
            time.sleep(5)

    def update_sot(self, task_type, result):
        response = requests.post(f"{self.sot_url}/update_state", json={'task_type': task_type, 'result': result})
        if response.status_code != 200:
            logging.error(f"Failed to update SOT for {task_type}: {response.text}")
        else:
            logging.info(f"Updated SOT for {task_type}")

    def get_batch_url(self):
        response = requests.get(f"{self.sot_url}/get_batch")
        return response.json()['batch_url']

    def get_targets_url(self):
        response = requests.get(f"{self.sot_url}/get_targets")
        return response.json()['targets_url']

if __name__ == "__main__":
    rpc_url = "http://localhost:8545"
    private_key = "YourPrivateKey"
    sot_url = "YourSourceOfTruthURL"
    subnet_addresses = {
        'embed': 'EmbedSubnetAddress',
        'forward': 'ForwardSubnetAddress',
        'final_logits': 'FinalLogitsSubnetAddress',
        'loss': 'LossSubnetAddress',
        'backward': 'BackwardSubnetAddress',
        'embed_backward': 'EmbedBackwardSubnetAddress'
    }
    
    master = Master(rpc_url, private_key, sot_url, subnet_addresses)
    master.main()
