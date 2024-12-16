# spl/master/main_logic.py
import asyncio
import json
import time
import datetime
import math
import logging
import aiohttp
import os
from eth_account.messages import encode_defunct
from eth_account import Account
from .config import args
from ..db.db_adapter_client import DBAdapterClient
from ..plugins.manager import get_plugin
from ..models import TaskStatus, ServiceType, OrderType
from ..common import (
    MAX_SUBMIT_TASK_RETRY_DURATION, TENSOR_NAME,
)
logging.getLogger(__name__)

class Master:
    def __init__(
        self,
        sot_url,
        job_id,
        subnet_id,
        db_adapter,
        max_iterations,
        detailed_logs
    ):
        logging.info("Initializing Master")
        self.sot_url = sot_url
        self.iteration = 0
        self.losses = []
        self.loss_queue = asyncio.Queue()
        self.current_wallet_index = 0
        self.done = False
        self.max_iterations = max_iterations
        self.job_id = job_id
        self.subnet_id = subnet_id
        self.db_adapter = db_adapter
        if detailed_logs:
            logging.getLogger().setLevel(logging.DEBUG)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "..", "data", "state")
        data_dir = os.path.abspath(data_dir)  # ensure abs path
        os.makedirs(data_dir, exist_ok=True)

        self.tasks = []

    async def initialize(self):
        self.plugin = await get_plugin(
            (await self.db_adapter.get_job(self.job_id)).plugin_id,
            self.db_adapter
        )
        self.max_concurrent_iterations = await self.plugin.get("max_concurrent_iterations")
        logging.info('Initialized plugin')

    async def get_bid_price(self):
        return 1

    async def submit_task(self, params, iteration_number):
        max_duration = datetime.timedelta(
            seconds=MAX_SUBMIT_TASK_RETRY_DURATION
        )
        start_time = datetime.datetime.now()

        attempt = 0
        retry_delay = 1
        while True:
            attempt += 1
            current_time = datetime.datetime.now()
            elapsed_time = current_time - start_time

            if elapsed_time >= max_duration:
                logging.error(
                    f"Failed to select solver within {max_duration}"
                )
                raise RuntimeError(
                    f"Failed to select solver within {max_duration}"
                )

            try:
                task_id = await self.db_adapter.create_bids_and_tasks(
                    self.job_id,
                    1,
                    await self.get_bid_price(),
                    params,
                    None
                )
                return task_id
            except Exception as e:
                logging.error(
                    f"Error submitting task on attempt {attempt}: {e}"
                )
                logging.error(f"Retrying in {retry_delay} seconds...")
                retry_delay = min(2 * retry_delay, 60)
                await asyncio.sleep(retry_delay)

    async def main_iteration(self, iteration_number):
        if self.iteration > self.max_iterations:
            self.done = True
            return
        self.iteration += 1
        logging.info(f"Starting iteration {iteration_number}")

        learning_params = await self.plugin.get_master_learning_hyperparameters()
        logging.info(
            f"Learning parameters for iteration {iteration_number}: {learning_params}"
        )

        logging.info(f"Iteration {iteration_number}: Starting training task")
        input_url = await self.get_input_url()  # We'll create a similar function or rename get_batch_and_targets_url to get_input_url

        task_params = json.dumps({
            "input_url": input_url,
            **learning_params,
        })

        task_id = (await self.submit_task(
            task_params, iteration_number
        ))[0]['task_id']

        result = await self.wait_for_result(task_id)
        logging.info(f"Iteration {iteration_number}: Received result: {result}")
        loss_value = result["loss"]
        await self.loss_queue.put(loss_value)
        await self.update_latest_loss(loss_value, result["version_number"])
        await self.update_sot(
            learning_params,
            TENSOR_NAME,
            result,
            batch_url,
            targets_url,
        )

        await self.db_adapter.update_job_iteration(self.job_id, self.iteration)
        
        logging.info(f"Iteration {iteration_number}: Completed training task")

        task = asyncio.create_task(
            self.main_iteration(self.iteration)
        )
        self.tasks.append(task)

    async def run_main(self):
        logging.info("Starting main process")
        await self.initialize()

        for _ in range(self.max_concurrent_iterations):
            task = asyncio.create_task(
                self.main_iteration(self.iteration)
            )
            self.tasks.append(task)

        while not self.done:
            if self.tasks:
                done, pending = await asyncio.wait(
                    self.tasks, return_when=asyncio.FIRST_COMPLETED
                )
                self.tasks = [task for task in self.tasks if not task.done()]
            else:
                await asyncio.sleep(1)

    def sign_message(self, message):
        message = encode_defunct(text=message)
        account = Account.from_key(args.private_key)
        logging.info(f"signing with address {account.address}")
        signed_message = account.sign_message(message)
        return signed_message.signature.hex()

    def generate_message(self, endpoint):
        import uuid
        nonce = str(uuid.uuid4())
        timestamp = int(time.time())
        message = {
            "endpoint": endpoint,
            "nonce": nonce,
            "timestamp": timestamp,
        }
        return message

    async def wait_for_result(self, task_id):
        while True:
            task = await self.db_adapter.get_task(task_id)
            if task.status == TaskStatus.ResolvedCorrect.name:
                result = json.loads(task.result)
                return result
            await asyncio.sleep(0.5)

    async def update_sot(
        self, learning_params, tensor_name, result, batch_url, targets_url
    ):
        params = {
            "result_url": result["grads_url"],
            "tensor_name": tensor_name,
            "version_number": result["version_number"],
            "batch_url": batch_url,
            "targets_url": targets_url,
            **learning_params,
        }

        message = json.dumps(
            self.generate_message("update_state"), sort_keys=True
        )
        signature = self.sign_message(message)
        headers = {"Authorization": f"{message}:{signature}"}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.sot_url}/update_state",
                json=params,
                headers=headers,
            ) as response:
                if response.status != 200:
                    logging.error(
                        f"Failed to update SOT for {tensor_name}: {await response.text()}"
                    )
                else:
                    logging.info(
                        f"Updated SOT for {tensor_name} with result: {result}"
                    )

    async def update_latest_loss(self, loss_value, version_number):
        payload = {"loss": loss_value, "version_number": version_number}

        url = os.path.join(self.sot_url, "update_loss")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    logging.error(
                        f"Failed to update loss value: {await response.text()}"
                    )
                else:
                    logging.info(
                        f"Updated latest loss value to {loss_value}"
                    )

    async def get_input_url(self):
        """
        Retrieve a single input_url from the SOT (Source of Truth) service.
        This replaces the old logic of retrieving batch_url and targets_url separately.

        The SOT /get_batch endpoint is expected to return a JSON of the form:
        {
            "input_url": "/data/state/temp/input_XXXX.pt"
        }

        We prepend self.sot_url to form a full URL for the input file.
        """
        url = os.path.join(self.sot_url, "get_batch")

        retry_delay = 1
        max_retries = 400
        retries = 0
        while retries < max_retries:
            try:
                logging.info(f"Retrieving input_url from {url}")
                message = json.dumps(
                    self.generate_message("get_batch"), sort_keys=True
                )
                signature = self.sign_message(message)
                headers = {"Authorization": f"{message}:{signature}"}

                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers) as response:
                        if response.status == 200:
                            response_json = await response.json()
                            if "input_url" in response_json:
                                input_url = self.sot_url + response_json["input_url"]
                                logging.info(f"Retrieved input_url: {input_url}")
                                return input_url
                            else:
                                logging.error("Response JSON does not contain 'input_url'.")
                        else:
                            logging.error(
                                f"Request failed with status code {response.status}"
                            )
            except Exception as e:
                logging.error(
                    f"Request failed: {e}. Retrying in {retry_delay} seconds..."
                )

            retries += 1
            await asyncio.sleep(retry_delay)

        raise Exception(
            f"Failed to retrieve input_url after {max_retries} attempts."
        )
