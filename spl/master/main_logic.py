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

        In this version, we *remove* each iteration’s entry from state_json["master_iteration_state"]
        once it completes, so we never keep finished states around.
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

        # We'll track iteration in code, but also store in DB
        self.iteration = None
        # We'll store concurrency tasks
        self.tasks = []
        # We'll keep losses in memory but also store them in DB's job.state_json
        self.losses = []

        # We'll store iteration sub-states in DB under:
        #    job.state_json["master_iteration_state"][str(iter_number)]
        self.state_key = "master_iteration_state"

        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "..", "data", "state")
        data_dir = os.path.abspath(data_dir)
        os.makedirs(data_dir, exist_ok=True)

    async def initialize(self):
        """
        Load iteration tracking from DB:
          - job.iteration (int)
          - master_losses
          - any incomplete iteration in state_json["master_iteration_state"]
        """
        self.plugin = await get_plugin(
            (await self.db_adapter.get_job(self.job_id)).plugin_id,
            self.db_adapter
        )
        job = await self.db_adapter.get_job(self.job_id)
        if not job:
            raise ValueError(f"No job found with ID {self.job_id}")

        self.iteration = job.iteration

        # Load known data from job.state_json
        state_data = await self.db_adapter.get_state_for_job(self.job_id)
        self.losses = state_data.get("master_losses", [])

        # See if there's any iteration "in progress"
        iteration_state_obj = state_data.get(self.state_key, {})
        last_unfinished_iteration = None
        for it_str, stage_info in iteration_state_obj.items():
            try:
                it_num = int(it_str)
                # We consider it "unfinished" if it doesn't have stage == "done"
                if stage_info.get("stage") != "done":
                    last_unfinished_iteration = (it_num, stage_info)
            except:
                pass

        if last_unfinished_iteration:
            it_num, stage_info = last_unfinished_iteration
            if it_num > self.iteration:
                self.iteration = it_num
            logging.info(f"Resuming iteration {it_num} from stage {stage_info.get('stage')}")

        # Also read concurrency from the plugin
        self.max_concurrent_iterations = await self.plugin.get("max_concurrent_iterations")
        logging.info(f"Master init complete. iteration={self.iteration}, losses={self.losses}")

    async def run_main(self):
        """
        Main loop for spawning training iteration tasks until done.
        """
        logging.info("Starting main process")
        await self.initialize()

        if self.iteration >= self.max_iterations:
            self.done = True
            return

        # Launch concurrency
        for _ in range(self.max_concurrent_iterations):
            task = asyncio.create_task(self.main_iteration(self.iteration))
            self.tasks.append(task)

        while not self.done:
            if self.tasks:
                done, pending = await asyncio.wait(
                    self.tasks, return_when=asyncio.FIRST_COMPLETED
                )
                self.tasks = [t for t in self.tasks if not t.done()]
                if self.iteration >= self.max_iterations:
                    self.done = True
            else:
                await asyncio.sleep(1)

    async def main_iteration(self, iteration_number):
        """
        A single iteration of the training process, with multiple stages:
          - pending_get_input
          - pending_submit_task
          - pending_wait_for_result
          - done
        """
        if iteration_number > self.max_iterations:
            self.done = True
            return

        iteration_state = await self._get_iteration_state(iteration_number)
        stage = iteration_state.get("stage", "pending_get_input")

        while stage != "done":
            if stage == "pending_get_input":
                input_url = iteration_state.get("input_url")
                if not input_url:
                    logging.info(f"Iteration {iteration_number}: retrieving input URL from SOT.")
                    input_url = await self._get_input_url_with_persist(iteration_number)
                iteration_state["input_url"] = input_url
                iteration_state["stage"] = "pending_submit_task"
                await self._save_iteration_state(iteration_number, iteration_state)
                stage = "pending_submit_task"

            elif stage == "pending_submit_task":
                task_id_info = iteration_state.get("task_id_info")
                if not task_id_info:
                    logging.info(f"Iteration {iteration_number}: creating a new DB task + bid.")
                    learning_params = await self.plugin.get_master_learning_hyperparameters()
                    iteration_state["learning_params"] = learning_params

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
                logging.info(f"Iteration {iteration_number}: waiting for result.")
                task_id = iteration_state["task_id_info"][0]["task_id"]
                result = iteration_state.get("result")
                if not result:
                    result = await self.wait_for_result(task_id)
                iteration_state["result"] = result

                # Extract the loss & record it
                loss_val = result["loss"]
                self.losses.append(loss_val)
                await self.update_latest_loss(loss_val, result["version_number"])
                # Bump the job iteration
                await self.db_adapter.update_job_iteration(self.job_id, iteration_number + 1)

                # Update state_json with new master_losses
                state_data = await self.db_adapter.get_state_for_job(self.job_id)
                state_data["master_losses"] = self.losses
                await self.db_adapter.update_state_for_job(self.job_id, state_data)

                # Possibly update the SOT with the new gradient data
                learning_params = iteration_state.get("learning_params", {})
                input_url = iteration_state["input_url"]
                await self.update_sot(learning_params, TENSOR_NAME, result, input_url)

                # Mark iteration done
                iteration_state["stage"] = "done"
                await self._save_iteration_state(iteration_number, iteration_state)

                # Now remove the iteration entirely from the DB so we keep no finished states
                await self._remove_iteration_entry(iteration_number)

                stage = "done"

        # Move on
        self.iteration += 1
        if self.iteration < self.max_iterations:
            task = asyncio.create_task(self.main_iteration(self.iteration))
            self.tasks.append(task)
        else:
            self.done = True

    async def _get_input_url_with_persist(self, iteration_number):
        """
        Retrieve input_url from the SOT's /get_batch endpoint, with retries.
        """
        url = os.path.join(self.sot_url, "get_batch")
        retry_delay = 1
        max_retries = 400
        retries = 0
        while retries < max_retries:
            try:
                logging.info(f"Iteration {iteration_number}: calling {url}")
                message = self.generate_message("get_batch")
                signature = self.sign_message(message)
                headers = {"Authorization": f"{message}:{signature}"}
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            if "input_url" in data:
                                return self.sot_url + data["input_url"]
                            else:
                                logging.error("No input_url in JSON response.")
                        else:
                            logging.error(f"Got HTTP {response.status} from get_batch.")
            except Exception as e:
                logging.error(f"Failed to retrieve input_url: {e}")
            retries += 1
            await asyncio.sleep(retry_delay)
        raise RuntimeError("Cannot get input_url after repeated retries.")

    async def _submit_task_with_persist(self, params_str, iteration_number):
        """
        Create a DB task + Bid. If we crash, we can pick up from the same task_id next time.
        """
        max_duration = datetime.timedelta(seconds=MAX_SUBMIT_TASK_RETRY_DURATION)
        start_time = datetime.datetime.now()
        attempt = 0
        retry_delay = 1
        while True:
            attempt += 1
            elapsed_time = datetime.datetime.now() - start_time
            if elapsed_time >= max_duration:
                raise RuntimeError(f"No solver assigned within {max_duration}!")

            try:
                price = await self.get_bid_price()
                bid_info = await self.db_adapter.create_bids_and_tasks(
                    self.job_id, 1, price, params_str, None
                )
                return bid_info
            except Exception as e:
                logging.error(f"Failure creating Bids/Tasks (attempt #{attempt}): {e}")
                logging.info(f"Retrying in {retry_delay}s...")
                retry_delay = min(2 * retry_delay, 60)
                await asyncio.sleep(retry_delay)

    async def wait_for_result(self, task_id):
        """
        Keep polling the DB for the final result. If we see SanityCheckPending, we run
        a local model-based sanity check and finalize it with /finalize_sanity_check.
        """
        while True:
            task = await self.db_adapter.get_task(task_id)
            if not task:
                await asyncio.sleep(0.5)
                continue

            if task.status == TaskStatus.SanityCheckPending.name:
                if task.result:
                    is_valid = await self.plugin.call_submodule(
                        "model_adapter", "run_sanity_check", task.result
                    )
                    await self.finalize_sanity_check(task_id, is_valid)

            if task.status in [TaskStatus.ResolvedCorrect.name, TaskStatus.ResolvedIncorrect.name]:
                return task.result

            await asyncio.sleep(0.5)

    async def finalize_sanity_check(self, task_id: int, is_valid: bool):
        """
        POST to the DB's /finalize_sanity_check to finalize a pending solution.
        """
        url = f"{self.db_adapter.base_url}/finalize_sanity_check"
        payload = {"task_id": task_id, "is_valid": is_valid}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    text = await response.text()
                    logging.error(f"finalize_sanity_check error: {text}")

    async def update_sot(self, learning_params, tensor_name, result, input_url):
        """
        Upload the new gradient updates to SOT via /update_state.
        """
        params = {
            "result_url": result["grads_url"],
            "tensor_name": tensor_name,
            "version_number": result["version_number"],
            "input_url": input_url,
            **learning_params,
        }
        msg = self.generate_message("update_state")
        sig = self.sign_message(msg)
        headers = {"Authorization": f"{msg}:{sig}"}

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.sot_url}/update_state", json=params, headers=headers) as resp:
                if resp.status != 200:
                    logging.error(f"Failed to update SOT for iteration results: {await resp.text()}")
                else:
                    logging.info("SOT updated with new grads.")

    async def update_latest_loss(self, loss_val, version_num):
        """
        Save latest loss in SOT for monitoring convenience.
        """
        data = {"loss": loss_val, "version_number": version_num}
        url = os.path.join(self.sot_url, "update_loss")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as resp:
                if resp.status != 200:
                    logging.error(f"update_loss error: {await resp.text()}")

    async def _remove_iteration_entry(self, iteration_num: int):
        """
        Remove the iteration entry from state_json["master_iteration_state"] entirely once it's done.
        """
        state_data = await self.db_adapter.get_state_for_job(self.job_id)
        iteration_state_obj = state_data.get(self.state_key, {})

        key_str = str(iteration_num)
        if key_str in iteration_state_obj:
            del iteration_state_obj[key_str]
            state_data[self.state_key] = iteration_state_obj
            await self.db_adapter.update_state_for_job(self.job_id, state_data)
            logging.info(f"Iteration {iteration_num} state removed from DB; not stored once done.")

    async def get_bid_price(self):
        """
        Return the price for your training tasks. 
        """
        return 1

    def generate_message(self, endpoint):
        import uuid
        nonce = str(uuid.uuid4())
        timestamp = int(time.time())
        return json.dumps({
            "endpoint": endpoint,
            "nonce": nonce,
            "timestamp": timestamp
        }, sort_keys=True)

    def sign_message(self, message):
        """
        Sign a message with our private key for authentication with SOT.
        """
        from eth_account.messages import encode_defunct
        msg_defunct = encode_defunct(text=message)
        account = Account.from_key(args.private_key)
        signed = account.sign_message(msg_defunct)
        return signed.signature.hex()

    async def _get_iteration_state(self, iteration_number):
        """
        Retrieve or initialize the sub-dict in job.state_json["master_iteration_state"][str(iteration_number)].
        """
        state_data = await self.db_adapter.get_state_for_job(self.job_id)
        iteration_state_obj = state_data.get(self.state_key, {})
        if str(iteration_number) not in iteration_state_obj:
            iteration_state_obj[str(iteration_number)] = {"stage": "pending_get_input"}
            state_data[self.state_key] = iteration_state_obj
            await self.db_adapter.update_state_for_job(self.job_id, state_data)
        return iteration_state_obj[str(iteration_number)]

    async def _save_iteration_state(self, iteration_number, iteration_state):
        """
        Save the iteration sub-dict for iteration_number back into the DB.
        """
        state_data = await self.db_adapter.get_state_for_job(self.job_id)
        iteration_state_obj = state_data.get(self.state_key, {})
        iteration_state_obj[str(iteration_number)] = iteration_state
        state_data[self.state_key] = iteration_state_obj
        await self.db_adapter.update_state_for_job(self.job_id, state_data)
