# spl/adapters/model_adapter.py
from abc import ABC, abstractmethod
import logging
import torch
import time
from ..device import device
from ..common import get_future_version_number, download_file
import torch.nn.functional as F
import torch.distributed as dist
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
import os
import aiohttp
import asyncio

class ModelAdapter(ABC):
    def __init__(self, model_config):
        self.model_config = model_config

    @abstractmethod
    async def train_task(
        self, TENSOR_NAME, sot_url, tensor_version_interval, expected_worker_time,
        task_params, predownloaded_data
    ):
        pass

    @abstractmethod
    def get_dummy_input(self):
        pass

    @abstractmethod
    def model_to_tensor(self, model: torch.nn.Module) -> torch.Tensor:
        pass

    @abstractmethod
    def tensor_to_model(self, tensor: torch.Tensor, existing_model=None) -> torch.nn.Module:
        pass

    @abstractmethod
    def init_tensor(self, zero_init: bool = False) -> torch.Tensor:
        pass

    @abstractmethod
    def compile_model(self, model: torch.nn.Module) -> torch.nn.Module:
        pass

    def initialize_environment(self, *args, **kwargs):
        pass


class StandardModelAdapter(ModelAdapter):
    @abstractmethod
    def loss_fn(self, logits, targets) -> torch.Tensor:
        pass

    @abstractmethod
    def preprocess_for_loss(self, logits, targets):
        pass

    @abstractmethod
    def postprocess_model_after_batch(self, model):
        pass

    def get_forward_args(self):
        pass

    def get_forward_kwargs(self):
        pass

    def forward(self, model, inputs):
        forward_args = self.get_forward_args()
        forward_kwargs = self.get_forward_kwargs()
        if forward_args is None:
            forward_args = []
        if forward_kwargs is None:
            forward_kwargs = {}
        return model(inputs, *forward_args, **forward_kwargs)

    def forward_and_loss(self, model, inputs, targets):
        output = self.forward(model, inputs)
        reshaped_logits, reshaped_targets = self.preprocess_for_loss(output, targets)
        loss = self.loss_fn(reshaped_logits, reshaped_targets)
        return loss

    async def train_task(
        self,
        TENSOR_NAME,
        sot_url,
        tensor_version_interval,
        expected_worker_time,
        task_params,
        combined_tensor
    ):
        """
        Train the model using a combined input tensor containing both batch and targets.
        The `combined_tensor` is passed in directly, and should already be loaded as a torch.Tensor.
        The first half of combined_tensor is the batch, and the second half is the targets.
        """

        logging.info("Starting train_task with a pre-downloaded combined input tensor")

        # this is so janky bruh
        sot_url = sot_url.replace('localhost', 'host.docker.internal')

        # Extract hyperparameters from task_params
        steps = task_params['steps']
        max_lr = task_params['max_lr']
        min_lr = task_params['min_lr']
        T_0 = task_params['T_0']
        weight_decay = task_params['weight_decay']

        # Split the combined tensor into batch and targets
        half = combined_tensor.size(0) // 2
        batch = combined_tensor[:half]
        targets = combined_tensor[half:]

        # Initialize model from the current tensor state
        model, version_number = await self.initialize_tensor(
            TENSOR_NAME, sot_url, tensor_version_interval, expected_worker_time
        )

        start_time = time.time()
        inputs = batch.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        batch_size = inputs.shape[0] // steps
        initial_params = [param.clone().detach() for param in model.parameters()]

        # Set up optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_0, eta_min=min_lr)

        first_loss = None
        for i in range(steps):
            # Slice the appropriate portion of the batch
            batch_inputs = inputs[i * batch_size:(i + 1) * batch_size].detach()
            batch_targets = targets[i * batch_size:(i + 1) * batch_size].detach()

            optimizer.zero_grad()
            loss = self.forward_and_loss(model, batch_inputs, batch_targets)
            loss_value = loss.item()
            if first_loss is None:
                first_loss = loss_value

            loss.backward()
            optimizer.step()
            scheduler.step()

            self.postprocess_model_after_batch(model)

        # Compute updates as difference between final and initial parameters
        with torch.no_grad():
            param_diff = [init_param - param for param, init_param in zip(model.parameters(), initial_params)]
            updates = torch.cat([diff.view(-1) for diff in param_diff])

        logging.info(f"Model task completed. Loss: {first_loss:.4f}. Total time: {time.time() - start_time:.2f} s")
        return version_number, updates, first_loss


    async def initialize_tensor(
        self, tensor_name, sot_url, tensor_version_interval,
        expected_worker_time, retries=3, backoff=1, chunk_timeout=5
    ):
        logging.debug(f"Starting initialization for tensor: {tensor_name}")

        init_start_time = time.time()
        valid_version = False
        max_iterations = 10
        iterations = 0
        timeout = aiohttp.ClientTimeout(total=200)

        while not valid_version and iterations < max_iterations:
            iterations += 1
            logging.debug(f"Initialization loop iteration {iterations}")
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{sot_url}/current_timestamp",
                        params={'tensor_name': tensor_name}
                    ) as response:
                        response.raise_for_status()
                        version_number = (await response.json())['version_number']
                        logging.debug(f"Received version_number: {version_number}")
            except aiohttp.ClientError as e:
                logging.error(f"Error fetching current_timestamp: {e}")
                await asyncio.sleep(backoff * iterations)
                continue

            def time_until_next_version(tensor_version_interval):
                return get_future_version_number(tensor_version_interval) - time.time()

            time_until_next = time_until_next_version(tensor_version_interval)
            logging.debug(f"Time until next version: {time_until_next} seconds")

            if time_until_next < expected_worker_time:
                logging.debug(f'Not enough time until next version. Waiting {time_until_next} s.')
                await asyncio.sleep(time_until_next)
            else:
                valid_version = True

        if not valid_version:
            raise RuntimeError("initialize_tensor: failed to get a valid version")

        for attempt in range(1, retries + 1):
            try:
                url = f"{sot_url}/latest_state"
                logging.debug(f"Requesting tensor {tensor_name} from URL: {url}")
                fetch_start_time = time.time()

                tensor = await download_file(url, tensor_name=tensor_name, download_type='tensor')
                download_end_time = time.time()
                logging.debug(f"Downloaded in {download_end_time - fetch_start_time:.2f}s")

                model = self.tensor_to_model(tensor.detach(), None)
                model.train()

                return model, version_number

            except asyncio.TimeoutError:
                logging.error(f"Attempt {attempt}: Chunk download timed out.")
            except aiohttp.ClientError as e:
                logging.error(f"Attempt {attempt}: Failed to fetch tensor {tensor_name}: {e}")
            if attempt < retries:
                await asyncio.sleep(backoff * attempt)

        raise RuntimeError(f"initialize_tensor: Failed to initialize tensor {tensor_name} after {retries} attempts")

    def compile_model(self, model: torch.nn.Module) -> torch.nn.Module:
        model = torch.compile(model)
        _ = self.forward(model, self.get_dummy_input())
        return model

    @abstractmethod
    def get_dummy_input(self):
        pass
    
    def model_to_tensor(self, model: torch.nn.Module) -> torch.Tensor:
        return torch.cat(tuple(p.view(-1) for p in model.parameters())).to(device)



    def tensor_to_model(self, tensor: torch.Tensor, existing_model=None) -> torch.nn.Module:
        if existing_model is not None:
            model = existing_model
        else:
            model = self.model_config.create_model().to(device)
        pointer = 0
        total_params = sum(p.numel() for p in model.parameters())

        if tensor.numel() != total_params:
            raise ValueError(f"Total number of parameters {total_params} does not match the size of the tensor {tensor.numel()}")

        for param in model.parameters():
            num_param = param.numel()
            logging.debug(f"Pointer: {pointer}, Num param: {num_param}, Tensor size: {tensor.numel()}")

            if pointer + num_param > tensor.numel():
                raise ValueError(f"Pointer {pointer} with num_param {num_param} exceeds tensor size {tensor.numel()}")

            param.data = tensor[pointer:pointer + num_param].view(param.size()).to(device)
            pointer += num_param

        return model

    def init_tensor(self, zero_init: bool = False) -> torch.Tensor:
        module = self.model_config.create_model().to(device)
        
        if not zero_init:
            # Initialize module parameters with Kaiming (He) initialization for all parameters
            for param in module.parameters():
                if param.requires_grad:
                    if param.ndimension() >= 2:  # Ensure the parameter tensor has at least 2 dimensions
                        torch.nn.init.kaiming_uniform_(param, a=0)  # He initialization (uniform)
                    else:
                        param.data.uniform_(-0.01, 0.01)  # Small random uniform initialization for scalars or 1D tensors
            tensors = [param.data for param in module.parameters()]
            tensor = torch.cat([tensor.view(-1) for tensor in tensors])
        else:  # Zero initialization for Adam tensors
            tensors = [param.data for param in module.parameters()]
            tensor = torch.cat([torch.zeros_like(tensor).view(-1) for tensor in tensors])
        return tensor
    
    def initialize_environment(self, *args, **kwargs):
        torch.set_float32_matmul_precision('high')

class FairscaleModelAdapter(ModelAdapter):
    def initialize_environment(self, backend='nccl'):
        logging.info("Initializing distributed environment")
        self.initialize_distributed_environment(backend)
        initialize_model_parallel(model_parallel_size_=1)
        super().initialize_environment()
        logging.info("Environment and global variables initialized")
    
    def initialize_distributed_environment(self, backend, master_addr='localhost', master_port=None):
        if master_port is None:
            master_port = str(12356 + os.getpid() % 10000)
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)

class TransformerModelAdapter(StandardModelAdapter):
    def loss_fn(self, logits, targets) -> torch.Tensor:
        return F.cross_entropy(logits, targets, ignore_index=self.model_config.tokenizer.pad_id)

    def preprocess_for_loss(self, logits, targets):
        reshaped_logits = logits.view(-1, logits.size(-1))
        reshaped_targets = targets.view(-1)
        return reshaped_logits, reshaped_targets
    
    def postprocess_model_after_batch(self, model):
        if hasattr(model, 'layers'):
            for layer in model.layers:
                if hasattr(layer, 'attention'):
                    layer.attention.cache_k = layer.attention.cache_k.detach()
                    layer.attention.cache_v = layer.attention.cache_v.detach()
    
    def get_dummy_input(self):
        batch_size = 1
        value = torch.randint(
            1,
            self.model_config.get_vocab_size(),
            (batch_size, self.model_config.get_max_seq_len())
        ).to(device)
        print(f'Dummy input: {value}')
        return value

    def get_top_token(self, logits):
        return torch.argmax(logits, dim=-1).cpu().numpy().tolist()[0]
    
    def generate(self, model, input, max_new_tokens=None):
        pass


    # Other model_to_tensor, tensor_to_model, init_tensor methods as previously implemented.
    # ... (Assume these are defined as in your original code.)

class AdderModelAdapter(StandardModelAdapter):
    def loss_fn(self, logits, targets) -> torch.Tensor:
        return F.mse_loss(logits, targets)

    def preprocess_for_loss(self, logits, targets):
        if logits.shape != targets.shape:
            raise ValueError(f"Logits shape {logits.shape} does not match targets shape {targets.shape}")
        return logits, targets

    def postprocess_model_after_batch(self, model):
        pass

    def get_dummy_input(self):
        input_tensor = torch.tensor([[2.0, 3.0]]).to(device)
        return input_tensor

    # model_to_tensor, tensor_to_model, init_tensor would be implemented here as needed.

class LlamaModelAdapter(TransformerModelAdapter, FairscaleModelAdapter):
    def get_forward_kwargs(self):
        return {'start_pos': 0}

class NanoGPTModelAdapter(TransformerModelAdapter):
    def forward_and_loss(self, model, inputs, targets):
        logging.debug(f'Inputs: {inputs}')
        logging.debug(f'Targets: {targets}')
        logging.debug(f'Shape of inputs: {inputs.shape}')
        logging.debug(f'Shape of targets: {targets.shape}')
        return model.forward(inputs, targets)[1]
    
    def forward(self, model, inputs):
        return model.forward(inputs)[0]

    def generate(self, model, input, max_new_tokens=None, temperature=0.8, top_k=200):
        if max_new_tokens is None:
            max_new_tokens = self.model_config.get_max_seq_len()
        return model.generate(input, max_new_tokens, temperature=temperature, top_k=top_k)