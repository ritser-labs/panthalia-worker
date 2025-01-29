# file: spl/master/main_logic.py

import asyncio
import json
import time
import datetime
import logging
import aiohttp
import os
from eth_account import Account
from eth_account.messages import encode_defunct

from .config import args
from ..db.db_adapter_client import DBAdapterClient
from ..plugins.manager import get_plugin
from ..models import TaskStatus
from ..common import (
    MAX_SUBMIT_TASK_RETRY_DURATION,
    TENSOR_NAME,
)

# Import iteration helpers from iteration_logic.py
from .iteration_logic import (
    wait_for_result,
    get_input_url_with_persist,
    submit_task_with_persist,
    get_iteration_state,
    save_iteration_state,
    remove_iteration_entry,
)

logger = logging.getLogger(__name__)


class Master:
    def __init__(
        self,
        sot_url: str,
        job_id: int,
        subnet_id: int,
        db_adapter: DBAdapterClient,
        max_iterations: float,
        detailed_logs: bool,
        max_concurrent_iterations: int = 4
    ):
        """
        :param sot_url:  The SOT base URL (e.g. http://localhost:5001)
        :param job_id:   The Job ID in the DB
        :param subnet_id:The Subnet ID
        :param db_adapter: DBAdapterClient instance for DB queries
        :param max_iterations: Max iteration count
        :param detailed_logs:  If True, set logger to DEBUG
        :param max_concurrent_iterations: concurrency limit
        """
        self.sot_url = sot_url
        self.job_id = job_id
        self.subnet_id = subnet_id
        self.db_adapter = db_adapter
        self.max_iterations = max_iterations
        self.detailed_logs = detailed_logs
        self.max_concurrent_iterations = max_concurrent_iterations

        self.logger = logging.getLogger(__name__)
        self.done = False
        self.iteration = None
        self.losses = []
        self.tasks = []  # each is an asyncio.Task for an iteration

        # The key in master_state_json where iteration sub-states live
        self.state_key = "master_iteration_state"

        if detailed_logs:
            logging.getLogger().setLevel(logging.DEBUG)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "..", "data", "state")
        data_dir = os.path.abspath(data_dir)
        os.makedirs(data_dir, exist_ok=True)

    async def initialize(self):
        """
        Fetch the job & plugin from DB, load iteration & existing loss array from master_state_json.
        """
        job_obj = await self.db_adapter.get_job(self.job_id)
        if not job_obj:
            raise ValueError(f"No job found with ID={self.job_id}")

        self.plugin = await get_plugin(job_obj.plugin_id, self.db_adapter)
        self.iteration = job_obj.iteration

        # Load stored master_state_json from DB
        state_data = await self.db_adapter.get_master_state_for_job(self.job_id)
        self.losses = state_data.get("master_losses", [])

        # Possibly resume an unfinished iteration
        iteration_state_obj = state_data.get(self.state_key, {})
        last_unfinished = None
        for it_str, stage_info in iteration_state_obj.items():
            try:
                it_num = int(it_str)
                if stage_info.get("stage") != "done":
                    last_unfinished = (it_num, stage_info)
            except:
                pass

        if last_unfinished:
            it_num, stage_info = last_unfinished
            if it_num > self.iteration:
                self.iteration = it_num
            logger.info(f"[Master.initialize] Resuming iteration={it_num}, stage={stage_info.get('stage')}")

        logger.info(f"[Master.initialize] iteration={self.iteration}, known_losses={self.losses}")

    async def run_main(self):
        """
        Loop to spawn iteration tasks if job is active. If job is inactive or we 
        hit max_iterations, we eventually end. Also waits for all iteration tasks.
        """
        while not self.done:
            job_obj = await self.db_adapter.get_job(self.job_id)
            if not job_obj:
                logger.info(f"[Master.run_main] job {self.job_id} missing => stop.")
                self.done = True
                break

            can_spawn = (job_obj.active and (len(self.tasks) < self.max_concurrent_iterations))
            if can_spawn:
                # Spawn a new iteration
                new_task = asyncio.create_task(self.main_iteration(self.iteration))
                self.tasks.append(new_task)
                self.iteration += 1
            else:
                # if we can't spawn, rest briefly
                await asyncio.sleep(1)

            # See if any iteration tasks are done
            if self.tasks:
                done_set, pending_set = await asyncio.wait(
                    self.tasks,
                    timeout=0.5,
                    return_when=asyncio.FIRST_COMPLETED
                )
                for t in done_set:
                    self.tasks.remove(t)
            else:
                # If no tasks remain, check for inactivity or max iteration
                if (not job_obj.active) or (self.iteration >= self.max_iterations):
                    self.done = True

        # Wait for leftover tasks
        if self.tasks:
            logger.info(f"[Master.run_main] finishing leftover {len(self.tasks)} iteration tasks.")
            await asyncio.wait(self.tasks)

        logger.info(f"[Master.run_main] Master done for job {self.job_id}.")

    async def main_iteration(self, iteration_number: int):
        """
        One iteration’s state machine: pending_get_input -> pending_submit_task -> pending_wait_for_result -> done
        We store the sub-state in job.master_state_json["master_iteration_state"][str(iteration_number)].
        """
        if iteration_number > self.max_iterations:
            self.done = True
            return

        iteration_state = await get_iteration_state(
            db_adapter=self.db_adapter,
            job_id=self.job_id,
            state_key=self.state_key,
            iteration_number=iteration_number
        )
        stage = iteration_state.get("stage", "pending_get_input")

        while stage != "done":
            if stage == "pending_get_input":
                input_url = iteration_state.get("input_url")
                if not input_url:
                    self.logger.info(f"Iteration {iteration_number} => retrieving input from SOT.")
                    input_url = await get_input_url_with_persist(
                        self.sot_url, self.db_adapter, iteration_number
                    )
                iteration_state["input_url"] = input_url
                iteration_state["stage"] = "pending_submit_task"
                await save_iteration_state(
                    self.db_adapter, self.job_id, self.state_key,
                    iteration_number, iteration_state
                )
                stage = "pending_submit_task"

            elif stage == "pending_submit_task":
                task_id_info = iteration_state.get("task_id_info")
                if not task_id_info:
                    learning_params = await self.plugin.get_master_learning_hyperparameters()
                    iteration_state["learning_params"] = learning_params

                    input_url = iteration_state["input_url"]
                    params_json = json.dumps({
                        "input_url": input_url,
                        **learning_params,
                    })
                    # Create a DB task+bid
                    task_id_info = await submit_task_with_persist(
                        self.db_adapter, self.job_id, iteration_number, params_json
                    )
                    if not task_id_info or len(task_id_info) < 1:
                        logger.error(
                            f"[main_iteration] create_bids_and_tasks returned empty => job inactive."
                        )
                        await self.db_adapter.update_job_active(self.job_id, False)
                        iteration_state["stage"] = "done"
                        await save_iteration_state(
                            self.db_adapter, self.job_id, self.state_key,
                            iteration_number, iteration_state
                        )
                        break

                iteration_state["task_id_info"] = task_id_info
                iteration_state["stage"] = "pending_wait_for_result"
                await save_iteration_state(
                    self.db_adapter, self.job_id, self.state_key,
                    iteration_number, iteration_state
                )
                stage = "pending_wait_for_result"

            elif stage == "pending_wait_for_result":
                logger.info(f"Iteration {iteration_number} => waiting for result.")
                task_id_info = iteration_state.get("task_id_info", [])
                if not task_id_info or len(task_id_info) < 1:
                    logger.error("[main_iteration] No task_id_info => can't wait. Marking done.")
                    iteration_state["stage"] = "done"
                    await save_iteration_state(
                        self.db_adapter, self.job_id, self.state_key,
                        iteration_number, iteration_state
                    )
                    break

                actual_task_id = task_id_info[0].get("task_id")
                if not actual_task_id:
                    logger.error("[main_iteration] task_id_info missing 'task_id'.")
                    iteration_state["stage"] = "done"
                    await save_iteration_state(
                        self.db_adapter, self.job_id, self.state_key,
                        iteration_number, iteration_state
                    )
                    break

                result = iteration_state.get("result")
                if not result:
                    # This call can re-create tasks if forcibly deleted
                    result = await wait_for_result(
                        self.db_adapter, self.plugin, self.sot_url,
                        self.job_id, actual_task_id
                    )
                iteration_state["result"] = result or {}

                # If no result => bail out
                if not result:
                    logger.warning(f"Iteration {iteration_number}: no result => bail.")
                    iteration_state["stage"] = "done"
                    await save_iteration_state(
                        self.db_adapter, self.job_id, self.state_key,
                        iteration_number, iteration_state
                    )
                    break

                # Record the loss
                loss_val = result.get("loss")
                version_num = result.get("version_number")
                if loss_val is not None:
                    self.losses.append(loss_val)
                    await self.update_latest_loss(loss_val, version_num)

                # Bump iteration in DB
                await self.db_adapter.update_job_iteration(self.job_id, iteration_number + 1)

                # Save new master_losses
                state_data = await self.db_adapter.get_master_state_for_job(self.job_id)
                state_data["master_losses"] = self.losses
                await self.db_adapter.update_master_state_for_job(self.job_id, state_data)

                # Possibly update SOT with new grads
                learning_params = iteration_state.get("learning_params", {})
                input_url = iteration_state["input_url"]
                if loss_val is not None and version_num is not None:
                    await self.update_sot(learning_params, TENSOR_NAME, result, input_url)

                # Done with iteration
                iteration_state["stage"] = "done"
                await save_iteration_state(
                    self.db_adapter, self.job_id, self.state_key,
                    iteration_number, iteration_state
                )
                await remove_iteration_entry(
                    self.db_adapter, self.job_id, self.state_key,
                    iteration_number
                )
                stage = "done"

    async def finalize_sanity_check(self, task_id: int, is_valid: bool):
        """
        If we see a task is 'SanityCheckPending', we finalize it as correct/incorrect.
        """
        url = f"{self.db_adapter.base_url}/finalize_sanity_check"
        payload = {"task_id": task_id, "is_valid": is_valid}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"[finalize_sanity_check] error => {text}")

    async def update_sot(self, learning_params, tensor_name, result, input_url):
        """
        Upload new gradients to the SOT's /update_state
        """
        if "grads_url" not in result or "version_number" not in result:
            logger.error("[update_sot] result missing grads_url or version_number.")
            return

        payload = {
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
            update_url = f"{self.sot_url}/update_state"
            async with session.post(update_url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    logger.error(f"[update_sot] error => {await resp.text()}")
                else:
                    logger.info("[update_sot] SOT updated with new grads.")

    async def update_latest_loss(self, loss_val, version_num):
        """
        POST to SOT /update_loss so we can track the latest loss in the SOT as well
        """
        data = {"loss": loss_val, "version_number": version_num}
        url = os.path.join(self.sot_url, "update_loss")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as resp:
                if resp.status != 200:
                    logger.error(f"[update_latest_loss] error => {await resp.text()}")

    async def get_bid_price(self) -> int:
        """
        Retrieve or compute the next bid price. Default: just use subnet_db.target_price
        """
        subnet_db = await self.db_adapter.get_subnet(self.subnet_id)
        return subnet_db.target_price

    def generate_message(self, endpoint: str) -> str:
        import uuid
        nonce = str(uuid.uuid4())
        ts = int(time.time())
        return json.dumps({
            "endpoint": endpoint,
            "nonce": nonce,
            "timestamp": ts
        }, sort_keys=True)

    def sign_message(self, msg: str) -> str:
        """
        Sign the JSON message with the Master’s private key (args.private_key).
        """
        msg_defunct = encode_defunct(text=msg)
        account = Account.from_key(args.private_key)
        signed = account.sign_message(msg_defunct)
        return signed.signature.hex()
