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
        """
        A "stateful" Master that can persist its iteration progress to the DB. If it crashes
        mid-iteration, on next startup we look up the last stage in the DB and resume from there.
        """
        logging.info("Initializing Master")
        self.sot_url = sot_url
        self.job_id = job_id
        self.subnet_id = subnet_id
        self.db_adapter = db_adapter
        self.max_iterations = max_iterations
        self.done = False
        if detailed_logs:
            logging.getLogger().setLevel(logging.DEBUG)

        # We'll track the iteration in code, but also store in DB
        self.iteration = None
        # We'll keep a local in-memory reference to losses but also store them in DB's job.state_json
        self.losses = []
        # We'll store asyncio.Tasks for concurrency
        self.tasks = []
        # Just for readability
        self.state_key = "master_iteration_state"

        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "..", "data", "state")
        data_dir = os.path.abspath(data_dir)
        os.makedirs(data_dir, exist_ok=True)

    async def initialize(self):
        """
        Load or create iteration tracking from the DB's job entry.
        This means:
          - job.iteration (the numeric iteration)
          - state_json["master_losses"] (the list of historical losses)
          - any "unfinished" iteration in state_json["master_iteration_state"]
        """
        self.plugin = await get_plugin(
            (await self.db_adapter.get_job(self.job_id)).plugin_id,
            self.db_adapter
        )
        job = await self.db_adapter.get_job(self.job_id)
        if not job:
            raise ValueError(f"No job found with ID {self.job_id}")
        self.iteration = job.iteration

        # Also read additional data from job.state_json
        state_data = await self.db_adapter.get_state_for_job(self.job_id)
        self.losses = state_data.get("master_losses", [])

        # Check if there is an unfinished iteration stage. For example:
        #   "master_iteration_state": {
        #       10: { "stage": "pending_wait_for_result", ...other info... },
        #   }
        iteration_state_obj = state_data.get(self.state_key, {})
        last_unfinished_iteration = None
        for it_str, stage_info in iteration_state_obj.items():
            # Only consider int iteration that haven't marked themselves "done"
            try:
                it_num = int(it_str)
                if not stage_info.get("done", False):
                    last_unfinished_iteration = (it_num, stage_info)
            except:
                pass

        if last_unfinished_iteration is not None:
            it_num, stage_info = last_unfinished_iteration
            if it_num > self.iteration:
                # If the job iteration is behind, let's set iteration to it_num
                self.iteration = it_num
            logging.info(f"Resuming iteration {it_num} from stage {stage_info['stage']}")

        # We also pull max_concurrent_iterations from the plugin
        self.max_concurrent_iterations = await self.plugin.get("max_concurrent_iterations")
        logging.info(f"Master init complete. iteration={self.iteration}, losses={self.losses}")

    async def run_main(self):
        """
        The outer "infinite" loop that checks if we are done, and spawns concurrency for each iteration.
        """
        logging.info("Starting main process")
        await self.initialize()

        # If we've exceeded max_iterations, mark ourselves done
        if self.iteration >= self.max_iterations:
            self.done = True
            return

        # Start up to max_concurrent_iterations tasks
        for _ in range(self.max_concurrent_iterations):
            task = asyncio.create_task(self.main_iteration(self.iteration))
            self.tasks.append(task)

        while not self.done:
            if self.tasks:
                done, pending = await asyncio.wait(
                    self.tasks, return_when=asyncio.FIRST_COMPLETED
                )
                self.tasks = [t for t in self.tasks if not t.done()]

                # Possibly check if we are done
                if self.iteration >= self.max_iterations:
                    self.done = True
            else:
                await asyncio.sleep(1)

    async def main_iteration(self, iteration_number):
        """
        One iteration of the training process. We'll break it up into "stages".
        If the master crashes, we can see which stage we were in and resume.
        """
        if iteration_number > self.max_iterations:
            self.done = True
            return

        # We'll load the iteration state from the DB if it exists
        iteration_state = await self._get_iteration_state(iteration_number)
        stage = iteration_state.get("stage", "pending_get_input")

        # If we have not begun or are partially done, run through the steps
        while stage != "done":
            if stage == "pending_get_input":
                # Next stage is pending_submit_task
                input_url = iteration_state.get("input_url", None)
                if not input_url:
                    logging.info(f"Iteration {iteration_number}: retrieving new input_url.")
                    input_url = await self._get_input_url_with_persist(iteration_number)
                iteration_state["input_url"] = input_url
                iteration_state["stage"] = "pending_submit_task"
                await self._save_iteration_state(iteration_number, iteration_state)
                stage = "pending_submit_task"

            elif stage == "pending_submit_task":
                # Submit the task to create the bid & get a task_id
                task_id_info = iteration_state.get("task_id_info", None)
                if not task_id_info:
                    logging.info(f"Iteration {iteration_number}: submitting new training task.")
                    learning_params = await self.plugin.get_master_learning_hyperparameters()
                    iteration_state["learning_params"] = learning_params

                    # Build task_params to store in DB
                    input_url = iteration_state["input_url"]
                    params_json = json.dumps({
                        "input_url": input_url,
                        **learning_params,
                    })
                    task_id_info = await self._submit_task_with_persist(params_json, iteration_number)
                iteration_state["task_id_info"] = task_id_info
                iteration_state["stage"] = "pending_wait_for_result"
                await self._save_iteration_state(iteration_number, iteration_state)
                stage = "pending_wait_for_result"

            elif stage == "pending_wait_for_result":
                # We'll wait for the DB to mark the task as completed
                # Then gather the result (loss, version_number, etc.)
                logging.info(f"Iteration {iteration_number}: waiting for result.")
                task_id = iteration_state["task_id_info"][0]['task_id']
                result = iteration_state.get("result", None)
                if not result:
                    # Wait for the result from the DB, do the "sanity check" if needed
                    result = await self.wait_for_result(task_id)
                iteration_state["result"] = result

                # We log the loss in memory & DB
                loss_value = result["loss"]
                self.losses.append(loss_value)
                await self.update_latest_loss(loss_value, result["version_number"])
                await self.db_adapter.update_job_iteration(self.job_id, iteration_number + 1)

                # store updated losses in job.state_json
                state_data = await self.db_adapter.get_state_for_job(self.job_id)
                state_data["master_losses"] = self.losses
                await self.db_adapter.update_state_for_job(self.job_id, state_data)

                # Possibly update the SOT with new gradient info
                learning_params = iteration_state.get("learning_params", {})
                input_url = iteration_state["input_url"]
                await self.update_sot(learning_params, TENSOR_NAME, result, input_url)

                # Mark iteration done
                iteration_state["done"] = True
                iteration_state["stage"] = "done"
                await self._save_iteration_state(iteration_number, iteration_state)
                stage = "done"

        # Move to the next iteration
        self.iteration += 1
        # Optionally schedule the next iteration
        if self.iteration < self.max_iterations:
            task = asyncio.create_task(self.main_iteration(self.iteration))
            self.tasks.append(task)
        else:
            self.done = True

    async def _get_input_url_with_persist(self, iteration_number):
        """
        Utility to retrieve input URL from the SOT, with retry. Also sets partial state in DB.
        """
        input_url = None
        url = os.path.join(self.sot_url, "get_batch")
        retry_delay = 1
        max_retries = 400
        retries = 0
        while retries < max_retries:
            try:
                logging.info(f"Iteration {iteration_number}: retrieving input_url from {url}")
                message = self.generate_message("get_batch")
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
                            logging.error(f"Request failed with status code {response.status}")
            except Exception as e:
                logging.error(f"Request failed: {e}. Retrying in {retry_delay} seconds...")
            retries += 1
            await asyncio.sleep(retry_delay)
        raise Exception(f"Failed to retrieve input_url after {max_retries} attempts.")

    async def _submit_task_with_persist(self, params, iteration_number):
        """
        Utility to create a DB task & place a bid for it. We store partial results in DB
        so that if the master crashes, we can resume from the same `task_id`.
        """
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
                bid_info = await self.db_adapter.create_bids_and_tasks(
                    self.job_id, 1, await self.get_bid_price(), params, None
                )
                return bid_info
            except Exception as e:
                logging.error(f"Error creating task/bid on attempt {attempt}: {e}")
                logging.error(f"Retrying in {retry_delay} seconds...")
                retry_delay = min(2 * retry_delay, 60)
                await asyncio.sleep(retry_delay)

    async def wait_for_result(self, task_id):
        """
        Wait for a DB task to become ResolvedCorrect or ResolvedIncorrect.
        If it hits SanityCheckPending, we run the model_adapter's sanity check logic
        and finalize it accordingly.
        """
        logging.info(f"Waiting for result of task {task_id}")
        while True:
            task = await self.db_adapter.get_task(task_id)
            if not task:
                await asyncio.sleep(0.5)
                continue

            if task.status == TaskStatus.SanityCheckPending.name:
                if task.result is not None:
                    is_valid = await self.plugin.call_submodule(
                        "model_adapter", "run_sanity_check", task.result
                    )
                    await self.finalize_sanity_check(task_id, is_valid)

            if task.status in [TaskStatus.ResolvedCorrect.name, TaskStatus.ResolvedIncorrect.name]:
                return task.result

            await asyncio.sleep(0.5)

    async def finalize_sanity_check(self, task_id: int, is_valid: bool):
        """
        Calls the DB endpoint to finalize the sanity check (transition from SanityCheckPending
        to ResolvedCorrect or ResolvedIncorrect).
        """
        url = f"{self.db_adapter.base_url}/finalize_sanity_check"
        payload = {"task_id": task_id, "is_valid": is_valid}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    logging.error(f"Failed to finalize sanity check for task {task_id}: {await response.text()}")
                else:
                    logging.info(f"Finalized sanity check for task {task_id} with is_valid={is_valid}")

    async def update_sot(self, learning_params, tensor_name, result, input_url):
        """
        After finishing a task (one iteration), we push updated grads to SOT via /update_state.
        """
        params = {
            "result_url": result["grads_url"],
            "tensor_name": tensor_name,
            "version_number": result["version_number"],
            "input_url": input_url,
            **learning_params,
        }
        message = self.generate_message("update_state")
        signature = self.sign_message(message)
        headers = {"Authorization": f"{message}:{signature}"}
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.sot_url}/update_state", json=params, headers=headers) as response:
                if response.status != 200:
                    logging.error(f"Failed to update SOT for {tensor_name}: {await response.text()}")
                else:
                    logging.info(f"Updated SOT for {tensor_name} with result: {result}")

    async def update_latest_loss(self, loss_value, version_number):
        """
        Save the latest loss to the SOT so that other watchers can see the training progress.
        """
        payload = {"loss": loss_value, "version_number": version_number}
        url = os.path.join(self.sot_url, "update_loss")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    logging.error(f"Failed to update loss value: {await response.text()}")
                else:
                    logging.info(f"Updated latest loss value to {loss_value}")

    async def get_bid_price(self):
        """
        Return the price you want to place a bid for the next training task. 
        This is a placeholder: you might want a dynamic strategy.
        """
        return 1

    def generate_message(self, endpoint):
        import uuid
        nonce = str(uuid.uuid4())
        timestamp = int(time.time())
        message = {
            "endpoint": endpoint,
            "nonce": nonce,
            "timestamp": timestamp,
        }
        return json.dumps(message, sort_keys=True)

    def sign_message(self, message):
        message = encode_defunct(text=message)
        account = Account.from_key(args.private_key)
        logging.debug(f"Signing with address {account.address}")
        signed_message = account.sign_message(message)
        return signed_message.signature.hex()

    async def _get_iteration_state(self, iteration_number):
        """
        Retrieve the iteration sub-dict from DB's job.state_json["master_iteration_state"][iteration_number].
        If it doesn't exist, we create an empty object for it.
        """
        state_data = await self.db_adapter.get_state_for_job(self.job_id)
        iteration_state_obj = state_data.get(self.state_key, {})
        if str(iteration_number) not in iteration_state_obj:
            iteration_state_obj[str(iteration_number)] = {"stage": "pending_get_input"}
        return iteration_state_obj[str(iteration_number)]

    async def _save_iteration_state(self, iteration_number, iteration_state):
        """
        Save the iteration sub-dict into DB's job.state_json["master_iteration_state"][iteration_number].
        """
        state_data = await self.db_adapter.get_state_for_job(self.job_id)
        iteration_state_obj = state_data.get(self.state_key, {})
        iteration_state_obj[str(iteration_number)] = iteration_state
        state_data[self.state_key] = iteration_state_obj
        await self.db_adapter.update_state_for_job(self.job_id, state_data)
