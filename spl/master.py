import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


import asyncio
import json
import time
import datetime
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import aiohttp
import queue
import requests
from web3 import AsyncWeb3
from web3.middleware import async_geth_poa_middleware
from web3.exceptions import ContractCustomError, TransactionNotFound
from .common import (
    load_contracts,
    TaskStatus,
    PoolState,
    Task,
    async_transact_with_contract_function,
    wait_for_state_change,
    approve_token_once,
    MAX_SUBMIT_TASK_RETRY_DURATION,
    MAX_SELECT_SOLVER_TIME,
    TENSOR_NAME,
    generate_wallets,
    fund_wallets,
    SOT_PRIVATE_PORT,
    wait_for_health,
    load_abi,
    TENSOR_NAME
)
from io import BytesIO
import os
from eth_account.messages import encode_defunct
from eth_account import Account
import uuid
from .db.db_adapter_client import DBAdapterClient
from .plugin_manager import get_plugin
from typing import List, Dict
import subprocess
from .models import TaskStatus, ServiceType
from .deploy.local_test_run import LOG_DIR, SOT_LOG_FILE, package_root_dir
from .deploy.test_run import (
    GPU_TYPE,
    DOCKER_IMAGE,
    get_log_file,
    BASE_TEMPLATE_ID,
    DOCKER_CMD
)
from .deploy.cloud_adapters.runpod import (
    launch_instance_and_record_logs,
    get_public_ip_and_port
)
import traceback

class Master:
    def __init__(
        self,
        rpc_url,
        wallets,
        sot_url,
        subnet_addresses,
        job_id,  # Add job_id to interact with the database
        db_url,
        db_adapter,
        max_concurrent_iterations=2,
        max_iterations=float('inf'),
        detailed_logs=False,
    ):
        logger.info("Initializing Master")
        self.web3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
        self.web3.middleware_onion.inject(async_geth_poa_middleware, layer=0)
        self.sot_url = sot_url
        self.subnet_addresses = subnet_addresses
        self.max_concurrent_iterations = max_concurrent_iterations
        self.iteration = 0  # Track the number of iterations
        self.losses = []  # Initialize loss list
        self.loss_queue = queue.Queue()
        self.wallets = wallets
        self.current_wallet_index = 0
        self.done = False
        self.max_iterations = max_iterations
        self.job_id = job_id  # Track the job ID
        self.db_adapter = db_adapter
        if detailed_logs:
            logging.getLogger().setLevel(logging.DEBUG)

        # Setup paths for loss data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data", "state")
        os.makedirs(data_dir, exist_ok=True)
        self.plot_file = os.path.join(data_dir, "plot.json")

        # Load existing loss values if plot.json exists
        if os.path.exists(self.plot_file):
            try:
                with open(self.plot_file, "r") as f:
                    self.losses = json.load(f)
                logger.info(
                    f"Loaded {len(self.losses)} existing loss values from {self.plot_file}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to load existing loss values from {self.plot_file}: {e}"
                )
                self.losses = []
        else:
            self.losses = []
            logger.info(f"No existing plot.json found. Starting fresh.")

        # Set up the plot
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(
            range(len(self.losses)), self.losses, label="Loss"
        )
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Loss over Iterations")
        self.ax.legend()
        self.ax.grid(True)

        # Run the main process
        self.tasks = []  # Track running tasks

    def get_next_wallet(self):
        wallet = self.wallets[self.current_wallet_index]
        self.current_wallet_index = (
            self.current_wallet_index + 1
        ) % len(self.wallets)
        return wallet

    async def initialize(self):
        self.abis, self.contracts, self.error_selectors = load_contracts(
            self.web3, self.subnet_addresses
        )
        if not self.contracts:
            raise ValueError(
                "SubnetManager contracts not found. Please check the subnet_addresses configuration."
            )

        self.pool_address = None
        for contract in self.contracts.values():
            if hasattr(contract.functions, "pool"):
                self.pool_address = await contract.functions.pool().call()
                break
        if not self.pool_address:
            raise ValueError(
                "Pool contract address not found in any of the SubnetManager contracts."
            )

        self.pool = self.web3.eth.contract(
            address=self.pool_address, abi=self.abis["Pool"]
        )
        self.plugin = await get_plugin(
            (await self.db_adapter.get_job(self.job_id)).plugin_id,
            self.db_adapter
        )
        logger.info('Initialized contracts and plugin')

    async def approve_tokens_at_start(self):
        tasks = []
        for contract in self.contracts.values():
            token_address = await contract.functions.token().call()
            token_contract = self.web3.eth.contract(
                address=token_address, abi=self.abis["ERC20"]
            )
            for wallet in self.wallets:
                task = approve_token_once(
                    self.web3,
                    token_contract,
                    wallet["private_key"],
                    contract.address,
                    2**256 - 1,
                )
                tasks.append(task)
            await asyncio.gather(*tasks)
            tasks = []

    def update_plot(self, frame=None):
        while not self.loss_queue.empty():
            loss = self.loss_queue.get()
            self.losses.append(loss)
        self.line.set_xdata(range(len(self.losses)))
        self.line.set_ydata(self.losses)
        self.ax.relim()
        self.ax.autoscale_view()

        # Save the updated plot
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "loss_plot.png")
        self.fig.savefig(file_path)  # Save the figure after each update
        logger.info(f"Saved updated plot to {file_path}")

        # Save the updated loss values to plot.json
        try:
            with open(self.plot_file, "w") as f:
                json.dump(self.losses, f)
            logger.debug(f"Saved {len(self.losses)} loss values to {self.plot_file}")
        except Exception as e:
            logger.error(f"Failed to save loss values to {self.plot_file}: {e}")

    async def submit_task(self, task_type, params, iteration_number):
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
                logger.error(
                    f"Failed to select solver within {max_duration}"
                )
                raise RuntimeError(
                    f"Failed to select solver within {max_duration}"
                )

            try:
                if task_type not in self.contracts:
                    raise ValueError(
                        f"No contract loaded for task type {task_type}"
                    )

                logger.info(
                    f"Submitting task of type {task_type} with params: {params}"
                )
                encoded_params = json.dumps(params).encode("utf-8")

                await wait_for_state_change(
                    self.web3,
                    self.pool,
                    PoolState.Unlocked.value,
                    self.get_next_wallet()["private_key"],
                )
                receipt = await async_transact_with_contract_function(
                    self.web3,
                    self.contracts[task_type],
                    "submitTaskRequest",
                    self.get_next_wallet()["private_key"],
                    encoded_params,
                    attempts=1,
                )
                logger.info(
                    f"submitTaskRequest transaction receipt: {receipt}"
                )

                logs = self.contracts[task_type].events.TaskRequestSubmitted().process_receipt(
                    receipt
                )
                if not logs:
                    raise ValueError(
                        "No TaskRequestSubmitted event found in the receipt"
                    )

                task_id = logs[0]["args"]["taskId"]
                logger.info(
                    f"Iteration {iteration_number} - Task submitted successfully. Task ID: {task_id}"
                )

                # Create a new task in the DB
                await self.db_adapter.create_task(
                    self.job_id,
                    task_id,
                    iteration_number,
                    TaskStatus.SelectingSolver.name
                )

                return task_id
            except Exception as e:
                logger.error(
                    f"Error submitting task on attempt {attempt}: {e}"
                )
                logger.error(f"Retrying in {retry_delay} seconds...")
                retry_delay = min(2 * retry_delay, 60)
                await asyncio.sleep(retry_delay)

    async def select_solver(self, task_type, task_id, iteration_number):
        max_duration = datetime.timedelta(seconds=MAX_SELECT_SOLVER_TIME)
        start_time = datetime.datetime.now()

        attempt = 0
        while True:
            attempt += 1
            current_time = datetime.datetime.now()
            elapsed_time = current_time - start_time

            if elapsed_time >= max_duration:
                logger.error(
                    f"Failed to select solver within {max_duration}"
                )
                raise RuntimeError(
                    f"Failed to select solver within {max_duration}"
                )

            try:
                logger.info(f"Selecting solver for task ID: {task_id}")

                start_transact_time = time.time()
                receipt = await async_transact_with_contract_function(
                    self.web3,
                    self.contracts[task_type],
                    "selectSolver",
                    self.get_next_wallet()["private_key"],
                    task_id,
                    attempts=1,
                )
                end_transact_time = time.time()

                logger.info(
                    f"Transaction duration: {end_transact_time - start_transact_time} seconds"
                )
                logger.info(
                    f"Iteration {iteration_number} - selectSolver transaction receipt: {receipt}"
                )
                return True
            except Exception as e:
                logger.error(
                    f"Error selecting solver on attempt {attempt}: {e}"
                )
                logger.error(f"Retrying...")
                await asyncio.sleep(0.5)

    async def get_task_result(self, task_type, task_id, iteration_number):
        result = None
        try:
            task_tuple = await self.contracts[task_type].functions.getTask(
                task_id
            ).call()
            task = Task(*task_tuple)
            logger.info(
                f"Iteration {iteration_number} - {task_type} Task status: {task.status}"
            )
            logger.info(
                f"Expected status: {TaskStatus.SolutionSubmitted.value}"
            )
            if task.status == TaskStatus.SolutionSubmitted.value:
                await self.db_adapter.update_time_solved(
                    task_id,
                    self.job_id,
                    task.timeStatusChanged
                )
            if task.status == TaskStatus.SolverSelected.value:
                await self.db_adapter.update_time_solver_selected(
                    task_id,
                    self.job_id,
                    task.timeStatusChanged
                )
            if (
                task.status == TaskStatus.SolutionSubmitted.value
                or task.status == TaskStatus.ResolvedCorrect.value
                or task.status == TaskStatus.ResolvedIncorrect.value
            ):
                try:
                    result = json.loads(task.postedSolution.decode("utf-8"))
                except Exception as e:
                    result = "DECODING_ERROR"
                    logger.error(
                        f"Error decoding solution for task ID {task_id}: {e}"
                    )
            
            await self.db_adapter.update_task_status(
                task_id,
                self.job_id,
                TaskStatus(task.status).name,
                result,
                task.solver
            )
            return result
        except Exception as e:
            logger.error(
                f"Error getting task result for {task_type} with task ID {task_id}: {e}"
            )
            return None

    async def main_iteration(self, iteration_number):
        if self.iteration > self.max_iterations:
            self.done = True
            return
        self.iteration += 1
        logger.info(f"Starting iteration {iteration_number}")

        learning_params = self.plugin.get_master_learning_hyperparameters(iteration_number)
        logger.info(
            f"Learning parameters for iteration {iteration_number}: {learning_params}"
        )
        batch_url, targets_url = await self.get_batch_and_targets_url()

        logger.info(f"Iteration {iteration_number}: Starting training task")
        task_params = {
            "batch_url": batch_url,
            "targets_url": targets_url,
            **learning_params,
        }
        task_id = await self.submit_task(
            TENSOR_NAME, task_params, iteration_number
        )

        await wait_for_state_change(
            self.web3,
            self.pool,
            PoolState.SelectionsFinalizing.value,
            self.get_next_wallet()["private_key"],
        )
        await wait_for_state_change(
            self.web3,
            self.pool,
            PoolState.Unlocked.value,
            self.get_next_wallet()["private_key"],
        )

        await self.select_solver(TENSOR_NAME, task_id, iteration_number)
        result = await self.wait_for_result(
            TENSOR_NAME, task_id, iteration_number
        )
        loss_value = result["loss"]
        self.loss_queue.put(loss_value)
        await self.update_latest_loss(loss_value, result["version_number"])
        await self.update_sot(
            learning_params,
            TENSOR_NAME,
            result,
            batch_url,
            targets_url,
        )

        # Update the iteration count in the database
        await self.db_adapter.update_job_iteration(self.job_id, self.iteration)

        # Schedule the next iteration
        task = asyncio.create_task(
            self.main_iteration(self.iteration)
        )
        self.tasks.append(task)

        self.update_plot()

    async def run_main(self):
        logger.info("Starting main process")
        await self.initialize()
        await self.approve_tokens_at_start()

        # Start the initial set of iterations
        for _ in range(self.max_concurrent_iterations):
            task = asyncio.create_task(
                self.main_iteration(self.iteration)
            )
            self.tasks.append(task)

        # Dynamically await new tasks as they are added
        while not self.done:
            if self.tasks:
                done, pending = await asyncio.wait(
                    self.tasks, return_when=asyncio.FIRST_COMPLETED
                )
                self.tasks = [task for task in self.tasks if not task.done()]
            else:
                await asyncio.sleep(1)  # Prevent tight loop

    def sign_message(self, message):
        message = encode_defunct(text=message)
        account = self.web3.eth.account.from_key(args.private_key)
        logger.info(f"signing with address {account.address}")
        signed_message = account.sign_message(message)
        return signed_message.signature.hex()

    def generate_message(self, endpoint):
        nonce = str(uuid.uuid4())
        timestamp = int(time.time())
        message = {
            "endpoint": endpoint,
            "nonce": nonce,
            "timestamp": timestamp,
        }
        return message

    async def wait_for_result(self, task_type, task_id, iteration_number):
        while True:
            result = await self.get_task_result(
                task_type, task_id, iteration_number
            )
            if result is not None:
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
                    logger.error(
                        f"Failed to update SOT for {tensor_name}: {await response.text()}"
                    )
                else:
                    logger.info(
                        f"Updated SOT for {tensor_name} with result: {result}"
                    )

    async def update_latest_loss(self, loss_value, version_number):
        """Send the latest loss value to the SOT server."""
        payload = {"loss": loss_value, "version_number": version_number}

        url = os.path.join(self.sot_url, "update_loss")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    logger.error(
                        f"Failed to update loss value: {await response.text()}"
                    )
                else:
                    logger.info(
                        f"Updated latest loss value to {loss_value}"
                    )

    async def get_batch_and_targets_url(self):
        url = os.path.join(self.sot_url, "get_batch")

        retry_delay = 1
        max_retries = 400
        retries = 0
        while retries < max_retries:
            try:
                logger.info(f"Retrieving batch and targets URL from {url}")
                message = json.dumps(
                    self.generate_message("get_batch"), sort_keys=True
                )
                signature = self.sign_message(message)
                headers = {"Authorization": f"{message}:{signature}"}
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, headers=headers
                    ) as response:
                        if response.status == 200:
                            response_json = await response.json()
                            return (
                                self.sot_url + response_json["batch_url"],
                                self.sot_url + response_json["targets_url"],
                            )
                        else:
                            logger.error(
                                f"Request failed with status code {response.status}"
                            )
            except Exception as e:
                logger.error(
                    f"Request failed: {e}. Retrying in {retry_delay} seconds..."
                )

            retries += 1
            await asyncio.sleep(retry_delay)

        raise Exception(
            f"Failed to retrieve batch and targets URL after {self.max_retries} attempts."
        )


def load_wallets(wallets_string):
    with open(wallets_string, "r") as file:
        return json.load(file)

async def launch_sot(db_adapter, job, deploy_type, db_url):
    logging.info(f"launch_sot")
    sot_wallet = generate_wallets(1)[0]
    sot_id = await db_adapter.create_sot(job.id, None)
    
    if deploy_type == 'local':
        sot_url = f"http://localhost:{SOT_PRIVATE_PORT}"
        sot_log = open(SOT_LOG_FILE, 'w')
        sot_process = subprocess.Popen(
            [
                'python', '-m', 'spl.sot',
                '--sot_id', str(sot_id),
                '--db_url', db_url,
                '--private_key', sot_wallet['private_key'],
            ],
            stdout=sot_log, stderr=sot_log, cwd=package_root_dir
        )
        instance_private_key = None
        instance_pod_id = None
        instance_pid = sot_process.pid
    elif deploy_type == 'cloud':
        sot_instance, sot_helpers = await launch_instance_and_record_logs(
            name="sot",
            gpu_type=GPU_TYPE,
            container_disk_in_gb=40,
            image=DOCKER_IMAGE,
            gpu_count=1,
            ports=f'{SOT_PRIVATE_PORT}/tcp',
            log_file=get_log_file("sot"),
            template_id=BASE_TEMPLATE_ID,
            cmd=DOCKER_CMD,
            env={
                'GITHUB_TOKEN': os.environ.get('GITHUB_TOKEN', ''),
                'SERVICE_TYPE': 'sot',
                'SOT_PRIVATE_PORT': str(SOT_PRIVATE_PORT),
                'SOT_ID': sot_id,
                'DB_URL': db_url,
                'PRIVATE_KEY': sot_wallet['private_key'],
            }
        )
        instance_private_key = sot_helpers['private_key']
        instance_pod_id = sot_instance['id']
        instance_pid = None
        sot_ip, sot_port = await get_public_ip_and_port(sot_instance['id'], private_port=SOT_PRIVATE_PORT)
        sot_url = f"http://{sot_ip}:{sot_port}"

    await db_adapter.create_instance(
        "sot",
        ServiceType.Sot.name,
        job.id,
        instance_private_key,
        instance_pod_id,
        instance_pid
    )
    logging.info(f"SOT service started")
    if not await wait_for_health(sot_url):
        logging.error("Error: SOT service did not become available within the timeout period.")
        sot_process.terminate()
        exit(1)
    await db_adapter.update_sot(sot_id, sot_url)
    sot_db = await db_adapter.get_sot(job.id)
    sot_perm_id = sot_db.perm
    private_key_address = Account.from_key(args.private_key).address
    await db_adapter.create_perm(private_key_address, sot_perm_id)
    await db_adapter.create_perm(sot_wallet['address'], DB_PERM_ID)
    return sot_db, sot_url

async def launch_worker(
    worker_idx,
    db_adapter, job, deploy_type,
    subnet,
    worker_private_key: str,
    db_url: str,
    sot_url: str,
):
    this_worker_wallets = [worker_private_key]
    worker_name = f'worker_{worker_idx}'
    private_keys = '+'.join(this_worker_wallets)
    if deploy_type == 'local':
        command = [
            'python', '-m', 'spl.worker',
            '--subnet_id', str(subnet.id),
            '--private_keys', private_keys,
            '--sot_url', sot_url,
            '--db_url', db_url,
        ]
        if args.torch_compile:
            command.append('--torch_compile')
        
        if args.detailed_logs:
            command.append('--detailed_logs')
        log_file_path = os.path.join(LOG_DIR, f"{worker_name}.log")
        log_file = open(log_file_path, 'w')
        worker_process = subprocess.Popen(command, stdout=log_file, stderr=log_file, cwd=package_root_dir)
        instance_private_key = None
        instance_pod_id = None
        instance_pid = worker_process.pid
    elif deploy_type == 'cloud':
        env = {
            'GITHUB_TOKEN': os.environ.get('GITHUB_TOKEN', ''),
            'SUBNET_ID': subnet.id,
            'PRIVATE_KEYS': private_keys,
            'SOT_URL': sot_url,
            'DB_URL': db_url,
        }
        if args.torch_compile:
            env['TORCH_COMPILE'] = 'true'
        if args.detailed_logs:
            env['DETAILED_LOGS'] = 'true'
        
        worker_instance, worker_helpers = await launch_instance_and_record_logs(
            name=worker_name,
            gpu_type=GPU_TYPE,
            image=DOCKER_IMAGE,
            gpu_count=1,
            ports='',
            log_file=get_log_file(worker_name),
            env=env,
            template_id=BASE_TEMPLATE_ID,
            cmd=DOCKER_CMD
        )
        instance_private_key = worker_helpers['private_key']
        instance_pod_id = worker_instance['id']
        instance_pid = None
        # Create instance entry in the DB
    await db_adapter.create_instance(
        name=worker_name,
        service_type=ServiceType.Worker.name,
        job_id=job.id,
        private_key=instance_private_key,
        pod_id=instance_pod_id,
        process_id=instance_pid
    )
    logging.info(f"Started worker process {worker_idx} for tasks with command: {' '.join(command)}")


async def launch_workers(
    db_adapter, job, deploy_type,
    subnet,
    db_url: str,
    sot_url: str,
):
    web3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(subnet.rpc_url))
    worker_wallets = generate_wallets(args.num_workers)
    token_contract = web3.eth.contract(
        address=subnet.token_address, abi=load_abi('ERC20')
    )
    await fund_wallets(
        web3, args.private_key, worker_wallets,
        token_contract, 1, 10000 * 10**18, subnet.distributor_address
    )
    worker_tasks = []
    for i in range(args.num_workers):
        worker_wallet = worker_wallets[i]
        task = asyncio.create_task(launch_worker(
            i,
            db_adapter, job, deploy_type, subnet,
            worker_wallet['private_key'], db_url,
            sot_url
        ))
        worker_tasks.append(task)
    await asyncio.gather(*worker_tasks)

DB_PERM_ID = 1
# updated function to launch master process as an asyncio task
async def run_master_task(*args):
    try:
        logging.info("run_master_task")
        obj = Master(*args)
        await obj.run_main()
    except Exception as e:
        logging.error(f"Error in run_master_task: {e}")
        traceback_str = traceback.format_exc()
        logging.error(f"Traceback:\n{traceback_str}")

# updated check_for_new_jobs function to handle concurrent jobs correctly
async def check_for_new_jobs(
    private_key: str,
    db_url: str,
    max_concurrent_iterations: int,
    detailed_logs: bool,
    num_workers: int,
    deploy_type: str,
    num_master_wallets: int,
):
    jobs_processing = {}
    db_adapter = DBAdapterClient(db_url, private_key)
    
    logger.info(f"Checking for new jobs")
    while True:
        new_jobs = await db_adapter.get_jobs_without_instances()
        for job in new_jobs:
            if job.id in jobs_processing:
                continue

            logger.info(f"Starting new job: {job.id}")
            subnet = await db_adapter.get_subnet(job.subnet_id)
            subnet_addresses = {}
            subnet_addresses[TENSOR_NAME] = subnet.address
            # SOT
            logging.info(f"Starting SOT service")
            sot_db, sot_url = await launch_sot(
                db_adapter, job, deploy_type, db_url)
            
            # Workers
            logging.info(f"Starting worker processes")
            await launch_workers(
                db_adapter, job, deploy_type, subnet,
                db_url, sot_url
            )

            # Master
            logging.info(f"Starting master process")
            master_wallets = generate_wallets(num_master_wallets)
            web3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(subnet.rpc_url))
            token_contract = web3.eth.contract(
                address=subnet.token_address, abi=load_abi('ERC20')
            )
            logging.info(f"Funding master wallets")
            await fund_wallets(
                web3,
                private_key,
                master_wallets,
                token_contract,
                1,
                10000 * 10**18,
                subnet.distributor_address
            )
            master_args = [
                subnet.rpc_url,
                master_wallets,
                sot_url,
                subnet_addresses,
                job.id,
                db_url,
                db_adapter,
                max_concurrent_iterations,
                detailed_logs,
            ]

            # run master in a non-blocking way
            logger.info(f'Starting master process for job {job.id}')
            task = asyncio.create_task(run_master_task(*master_args))
            jobs_processing[job.id] = task

        # clean up finished jobs
        completed_jobs = [job_id for job_id, task in jobs_processing.items() if task.done()]
        for job_id in completed_jobs:
            jobs_processing.pop(job_id)

        await asyncio.sleep(1)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Master process for task submission"
    )
    parser.add_argument(
        "--private_key",
        type=str,
        required=True,
        help="The wallet private key",
    )
    parser.add_argument(
        "--db_url",
        type=str,
        required=True,
        help="URL for the database",
    )
    parser.add_argument(
        "--max_concurrent_iterations",
        type=int,
        default=4,
        help="Maximum number of concurrent iterations",
    )
    parser.add_argument(
        "--detailed_logs",
        action="store_true",
        help="Enable detailed logs",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of workers to start for each job",
    )
    parser.add_argument(
        "--num_master_wallets",
        type=int,
        help="Number of wallets to generate for the master",
        default=70,
    )
    parser.add_argument(
        "--deploy_type",
        type=str,
        required=True,
        help="Type of deployment (disabled, local, cloud)",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Enable torch.compile and model warmup",
    )

    args = parser.parse_args()

    logger.info(f'Starting master process')

    asyncio.run(check_for_new_jobs(
        args.private_key,
        args.db_url,
        args.max_concurrent_iterations,
        args.detailed_logs,
        args.num_workers,
        args.deploy_type,
        args.num_master_wallets,
    ))
