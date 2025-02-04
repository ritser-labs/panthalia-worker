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
from ..util.docker import janky_url_replace

# We'll import the chunked-DCT from demo.py:
from ..util import demo
import tempfile

class ModelAdapter(ABC):
    def __init__(self, model_config):
        self.model_config = model_config

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
    async def load_input_tensor(self, input_tensor):
        pass

    @abstractmethod
    async def execute_step(
        self,
        current_param_tensor: torch.Tensor,
        task_params: dict,
        step_idx: int
    ):
        """
        Do forward+backward on the mini-batch for 'step_idx',
        gather the gradient, chunked-DCT encode it, and return
        (encoded_grad_dict, loss_value).
        """
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

    async def load_input_tensor(self, input_tensor):
        """
        Expects input_tensor:  [ batch + batch ] => we split half for input, half for target
        """
        if input_tensor is None or not isinstance(input_tensor, torch.Tensor) or input_tensor.numel() == 0:
            raise ValueError("Invalid or empty input_tensor in load_input_tensor.")

        half = input_tensor.size(0) // 2
        if half == 0 or (half * 2 != input_tensor.size(0)):
            raise ValueError("Invalid shape for splitting into (inputs, targets).")

        self.inputs = input_tensor[:half].to(device, non_blocking=True)
        self.targets = input_tensor[half:].to(device, non_blocking=True)

    async def execute_step(
        self,
        current_param_tensor: torch.Tensor,
        task_params: dict,
        step_idx: int
    ):
        """
        We do forward/backward => chunked-DCT encode the gradient => return it.
        We'll rely on the worker code to apply any param diffs from the SOT.
        """
        steps = task_params['steps']
        model = self.tensor_to_model(current_param_tensor)

        batch_size = self.inputs.size(0) // steps
        if batch_size <= 0:
            raise ValueError(f"execute_step: Invalid batch_size, total={self.inputs.size(0)}, steps={steps}")

        start_idx = step_idx * batch_size
        end_idx = (step_idx + 1) * batch_size
        x_step = self.inputs[start_idx:end_idx].detach()
        y_step = self.targets[start_idx:end_idx].detach()

        # zero grads
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # forward/backward
        loss = self.forward_and_loss(model, x_step, y_step)
        loss_value = loss.item()
        loss.backward()

        # gather parameter grads into a single flat vector
        grads_list = []
        for p in model.parameters():
            if p.grad is not None:
                grads_list.append(p.grad.view(-1))
            else:
                grads_list.append(torch.zeros_like(p).view(-1))
        full_grads = torch.cat(grads_list, dim=0).detach()

        # chunked-DCT encode
        chunk_shape = task_params['chunk_shape']
        k_for_encoding = task_params['k']
        (
            freq_idxs,
            freq_vals_int8,
            freq_scales,
            freq_zero_points,
            new_error,
            pad_count
        ) = demo.chunked_dct_encode_int8(
            full_grads,
            chunk_shape=chunk_shape,
            k=k_for_encoding,
            prev_error=None,
            norm='ortho'
        )

        # build final encoded payload
        encoded_dict = {
            'freq_idxs': freq_idxs.cpu(),
            'freq_vals_int8': freq_vals_int8.cpu(),
            'freq_scales': freq_scales.cpu(),
            'freq_zero_points': freq_zero_points.cpu(),
            'chunk_shape': chunk_shape,
            'orig_shape': full_grads.shape,
            'pad_count': pad_count  # <-- NEW
        }

        return encoded_dict, loss_value

    def run_sanity_check(self, task_result: dict) -> bool:
        if not isinstance(task_result, dict):
            return False
        return True

    def compile_model(self, model: torch.nn.Module) -> torch.nn.Module:
        model = torch.compile(model)
        _ = self.forward(model, self.get_dummy_input())
        return model

    def model_to_tensor(self, model: torch.nn.Module) -> torch.Tensor:
        return torch.cat([p.view(-1) for p in model.parameters()]).to(device)

    def tensor_to_model(self, tensor: torch.Tensor, existing_model=None) -> torch.nn.Module:
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
            # some random init
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

    def initialize_environment(self, *args, **kwargs):
        torch.set_float32_matmul_precision('high')
    
    async def initialize_tensor(self, tensor_name, sot_url, task_params):
        """
        Attempt to download the latest parameter from the SOT by calling /latest_state.
        If no state exists yet (404 or other error), fallback to a fresh init_tensor().
        Returns (param_tensor, version_number).
        """
        version_number = 0
        param_tensor = None

        # Construct endpoint
        # You could optionally append ?version_number=<something>, but the default
        # is to serve the SOT’s *current* version if none is provided.
        endpoint = f"{sot_url}/latest_state"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint) as resp:
                    if resp.status == 200:
                        # The SOT includes X-Version-Number in the response header
                        ver_str = resp.headers.get("X-Version-Number", "0")
                        version_number = int(ver_str)

                        # Download the param file in memory or via a temp file
                        file_bytes = await resp.read()
                        with tempfile.NamedTemporaryFile(delete=False) as tmp_f:
                            tmp_path = tmp_f.name
                            tmp_f.write(file_bytes)

                        loaded_tensor = torch.load(tmp_path, map_location=device)
                        os.remove(tmp_path)

                        param_tensor = loaded_tensor.to(device)
                        logging.info(
                            f"[initialize_tensor] Fetched param from SOT, version={version_number}, "
                            f"shape={param_tensor.shape}"
                        )
                    else:
                        # 404 or some other code => fallback
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

        # If no param_tensor was loaded, revert to local initialization
        if param_tensor is None:
            param_tensor = self.init_tensor(zero_init=False)
            version_number = 0
            logging.info(
                f"[initialize_tensor] Created new param tensor locally, shape={param_tensor.shape}"
            )

        return param_tensor, version_number



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
