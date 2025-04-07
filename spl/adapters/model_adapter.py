# file: spl/adapters/model_adapter.py
from abc import ABC, abstractmethod
import logging
import torch
import time
from ..device import device, safetensors_device
from ..common import get_future_version_number, download_file
import torch.nn.functional as F
import torch.distributed as dist
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
import os
import aiohttp
import asyncio
from safetensors.torch import load_file as safetensors_load_file

from ..util import demo
import tempfile
import json

class ModelAdapter(ABC):
    def __init__(self, model_config):
        self.model_config = model_config
        self.enforce_reproducibility()
        
    def enforce_reproducibility(self):
        SEED_VALUE = 0
        torch.manual_seed(SEED_VALUE)
        torch.cuda.manual_seed(SEED_VALUE)
        torch.cuda.manual_seed_all(SEED_VALUE)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
    
    @abstractmethod
    async def load_input_tensor(self, input_tensor_dict):
        pass

    @abstractmethod
    async def execute_step(
        self,
        task_params: dict,
        step_idx: int
    ):
        pass

    @abstractmethod
    async def apply_diff(self, diff_tensor: torch.Tensor):
        pass

    async def initialize_tensor(self, tensor_name, sot_url, task_params):
        version_number = 0
        force_ver = task_params.get("force_version_number", None)
        param_tensor = None

        if force_ver is not None:
            endpoint = f"{sot_url}/latest_state?version_number={force_ver}"
        else:
            endpoint = f"{sot_url}/latest_state"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint) as resp:
                    if resp.status == 200:
                        ver_str = resp.headers.get("X-Version-Number", "0")
                        version_number = int(ver_str)

                        file_bytes = await resp.read()
                        with tempfile.NamedTemporaryFile(delete=False) as tmp_f:
                            tmp_path = tmp_f.name
                            tmp_f.write(file_bytes)

                        loaded_tensor = safetensors_load_file(tmp_path, device=safetensors_device)["tensor"].to(device)
                        os.remove(tmp_path)

                        param_tensor = loaded_tensor.to(device)
                        logging.info(
                            f"[initialize_tensor] Fetched param from SOT, version={version_number}, "
                            f"shape={param_tensor.shape}, forced={force_ver}"
                        )
                    else:
                        logging.info(
                            f"[initialize_tensor] SOT returned status={resp.status}. "
                            "Falling back to local init_tensor()."
                        )
        except Exception as e:
            logging.error(
                f"[initialize_tensor] Error requesting /latest_state from SOT: {e}. "
                "Falling back to local init_tensor().",
                exc_info=True
            )

        if param_tensor is None:
            param_tensor = self.init_tensor(zero_init=False)
            version_number = 0
            logging.info(
                f"[initialize_tensor] Created new param tensor locally, shape={param_tensor.shape}"
            )

        self._cached_param = param_tensor
        return version_number


class StandardModelAdapter(ModelAdapter):
    def __init__(self, model_config):
        super().__init__(model_config)
        self._cached_param = None
        self.inputs = None
        self.targets = None
        self.prev_error = None

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
        return []

    def get_forward_kwargs(self):
        return {}

    def forward(self, model, inputs):
        return model(inputs, *self.get_forward_args(), **self.get_forward_kwargs())

    def forward_and_loss(self, model, inputs, targets):
        output = self.forward(model, inputs)
        reshaped_logits, reshaped_targets = self.preprocess_for_loss(output, targets)
        loss = self.loss_fn(reshaped_logits, reshaped_targets)
        return loss

    async def load_input_tensor(self, input_tensor_dict):
        self.inputs = None
        self.targets = None
        self.prev_error = None
        
        self.inputs = input_tensor_dict['inputs'].to(device, non_blocking=True)
        self.targets = input_tensor_dict['targets'].to(device, non_blocking=True)

    async def execute_step(
        self,
        task_params: dict,
        step_idx: int
    ):
        """
        Execute forward/backward on a subset of (inputs, targets), optionally
        splitting that subset into multiple micro-batches for gradient accumulation.
        """
        self.enforce_reproducibility()
        if self._cached_param is None:
            raise ValueError("No parameter tensor set. Did you call initialize_tensor first?")

        # 1) Read the big parameters:
        steps = task_params.get('steps', 1)                     # total large steps
        accum_steps = task_params.get('accumulations_per_step', 1)

        # 2) Rebuild the model from the flat param vector
        model = self.tensor_to_model(self._cached_param)

        # 3) Slice out the portion of the batch for this step_idx
        total_examples = self.inputs.size(0)
        step_batch_size = total_examples // steps if steps > 0 else total_examples

        # If for any reason the step_batch_size is 0, fallback to using the entire set
        if step_batch_size <= 0:
            step_batch_size = total_examples

        start_idx = step_idx * step_batch_size
        end_idx   = min(start_idx + step_batch_size, total_examples)

        x_step = self.inputs[start_idx:end_idx].detach()
        y_step = self.targets[start_idx:end_idx].detach()

        # 4) Partition the step batch into micro-batches
        micro_batch_size = x_step.size(0) // accum_steps
        # We'll handle any remainder by giving the last micro-step the leftover
        remainder = x_step.size(0) % accum_steps

        # 5) Prepare an accumulator for flattening the sum of gradients
        accum_grads = torch.zeros_like(self._cached_param, dtype=torch.float32, device=self._cached_param.device)
        total_loss = 0.0

        # 6) Loop over each micro-step
        pointer_start = 0
        for micro_step in range(accum_steps):
            # This micro-step's slice
            batch_size_this_step = micro_batch_size
            if micro_step < remainder:
                # Spread out any leftover among the first 'remainder' micro-steps
                batch_size_this_step += 1

            pointer_end = pointer_start + batch_size_this_step
            x_micro = x_step[pointer_start:pointer_end]
            y_micro = y_step[pointer_start:pointer_end]
            pointer_start = pointer_end

            # Clear gradients
            model.zero_grad(set_to_none=True)

            # Forward + backward
            loss = self.forward_and_loss(model, x_micro, y_micro)

            # Keep track of raw (unscaled) loss for logging
            loss_value = loss.item()
            total_loss += loss_value

            # Scale the loss so that the final sums of grads end up averaged
            loss = loss / accum_steps
            loss.backward()

            # Accumulate flattened gradients
            pointer_param = 0
            for p in model.parameters():
                numel = p.numel()
                if p.grad is not None:
                    accum_grads[pointer_param:pointer_param + numel] += p.grad.detach().view(-1)
                pointer_param += numel

            # Explicitly break references
            del loss

        # Notice we do NOT do accum_grads /= accum_steps again. 
        # The gradient is already averaged because we did (loss / accum_steps).

        # If you want the *mean* loss across micro-steps, we do:
        total_loss /= accum_steps

        # 7) Optionally encode the gradient into a compressed form (like chunked DCT)
        chunk_shape = self.plugin.chunk_shape
        k_for_encoding = self.plugin.k
        (
            freq_idxs,
            freq_vals_int8,
            freq_scales,
            freq_zero_points,
            new_error,
            pad_count
        ) = demo.chunked_dct_encode_int8(
            accum_grads,
            chunk_shape=chunk_shape,
            k=k_for_encoding,
            prev_error=self.prev_error,
            norm='ortho'
        )
        self.prev_error = new_error

        encoded_dict = {
            'freq_idxs': freq_idxs,
            'freq_vals_int8': freq_vals_int8,
            'freq_scales': freq_scales,
            'freq_zero_points': freq_zero_points,
            'chunk_shape': chunk_shape,
            'orig_shape': torch.tensor(accum_grads.shape, dtype=torch.int64),
            'pad_count': pad_count
        }

        return encoded_dict, total_loss

    def compile_model(self, model: torch.nn.Module) -> torch.nn.Module:
        model = torch.compile(model)
        _ = self.forward(model, self.get_dummy_input())
        return model

    def model_to_tensor(self, model: torch.nn.Module) -> torch.Tensor:
        return torch.cat([p.view(-1) for p in model.parameters()]).to(device)

    def tensor_to_model(self, tensor: torch.Tensor, existing_model=None) -> torch.nn.Module:
        # ADDED: Check that the tensor is not None.
        if tensor is None:
            raise ValueError("Parameter tensor is not initialized. Did you call initialize_tensor()?")
        if existing_model is not None:
            model = existing_model
        else:
            model = self.model_config.create_model().to(device)

        pointer = 0
        for p in model.parameters():
            num_p = p.numel()
            p.data = tensor[pointer:pointer+num_p].view_as(p).to(device)
            pointer += num_p
        return model

    def init_tensor(self, zero_init: bool = False) -> torch.Tensor:
        module = self.model_config.create_model().to(device)
        if not zero_init:
            for p in module.parameters():
                if p.requires_grad:
                    if p.ndimension() >= 2:
                        torch.nn.init.kaiming_uniform_(p, a=0)
                    else:
                        p.data.uniform_(-0.01, 0.01)
        else:
            for p in module.parameters():
                p.data.zero_()

        all_p = [p.data for p in module.parameters()]
        return torch.cat([x.view(-1) for x in all_p]).to(device)

    async def decode_and_apply_diff(self, diff_data: dict):
        diff_tensor = await demo.decode_diff(diff_data)
        await self.apply_diff(diff_tensor)

    async def apply_diff(self, diff_tensor: torch.Tensor):
        if self._cached_param is None:
            raise ValueError("No param is set. Did you call initialize_tensor first?")
        diff_tensor = diff_tensor.to(self._cached_param.device)
        self._cached_param = self._cached_param + diff_tensor

    async def train_loop(self, task_params: dict, num_steps: int):
        lr = task_params.get('learning_rate', 1e-3)
        accum_steps = task_params.get('accumulations_per_step', 1)

        model = self.tensor_to_model(self._cached_param)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            loss = self.forward_and_loss(model, self.inputs, self.targets)
            loss_value = loss.item()
            
            loss = loss / accum_steps
            
            loss.backward()
            
            optimizer.step()
            
            self._cached_param = self.model_to_tensor(model)
            
            print(f"Training step {step+1}/{num_steps}: Loss = {loss_value:.4f}")
            
            await asyncio.sleep(0)
        
        return self._cached_param


class FairscaleModelAdapter(ModelAdapter):
    def initialize_environment(self, backend='nccl'):
        logging.info("Initializing distributed environment")
        self.initialize_distributed_environment(backend)
        initialize_model_parallel(model_parallel_size_=1)
        super().initialize_environment()
        logging.info("Environment and global variables initialized")
    
    def initialize_distributed_environment(self, backend, master_addr='localhost', master_port=None):
        if master_port is None:
            import os
            master_port = str(12356 + os.getpid() % 10000)
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)


class TransformerModelAdapter(StandardModelAdapter, ABC):
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


class AdderModelAdapter(StandardModelAdapter):
    def loss_fn(self, logits, targets) -> torch.Tensor:
        return F.mse_loss(logits, targets)

    def preprocess_for_loss(self, logits, targets):
        if logits.shape != targets.shape:
            raise ValueError(f"Logits shape {logits.shape} != targets shape {targets.shape}")
        return logits, targets

    def postprocess_model_after_batch(self, model):
        pass

    def get_dummy_input(self):
        input_tensor = torch.tensor([[2.0, 3.0]]).to(device)
        return input_tensor


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
