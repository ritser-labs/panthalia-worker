from abc import ABC, abstractmethod
import logging
import torch
import time
from ..device import device
import torch.nn.functional as F
from .model_config import BaseModelConfig, TransformerModelConfig
import torch.distributed as dist
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
import os

class ModelAdapter(ABC):
    def __init__(self, model_config: BaseModelConfig):
        self.model_config = model_config
    
    @abstractmethod
    def train_task(self, model, inputs, targets, accumulation_steps):
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
        #logging.info(f'Output: {output}')
        reshaped_logits, reshaped_targets = self.preprocess_for_loss(output, targets)
        loss = self.loss_fn(reshaped_logits, reshaped_targets)
        return loss

    def train_task(self, model, inputs, targets, accumulation_steps):
        logging.info("Starting train_task")

        start_time = time.time()
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logging.debug(f"Moved inputs and targets to device. Time taken: {time.time() - start_time:.2f} seconds")

        microbatch_size = inputs.shape[0] // accumulation_steps

        # Preallocate gradient accumulation tensors and zero them
        grads_accumulated = [torch.zeros_like(param, device=device) for param in model.parameters()]

        total_loss = 0.0

        logging.info(f"Accumulation steps: {accumulation_steps}, Microbatch size: {microbatch_size}")
        

        for i in range(accumulation_steps):
            batch_start_time = time.time()
            try:
                microbatch_inputs = inputs[i * microbatch_size:(i + 1) * microbatch_size].detach()
                microbatch_targets = targets[i * microbatch_size:(i + 1) * microbatch_size].detach()
                
                #logging.info(f'Microbatch Inputs: {microbatch_inputs}')
                #logging.info(f'Microbatch Targets: {microbatch_targets}')

                # Forward pass
                loss = self.forward_and_loss(model, microbatch_inputs, microbatch_targets)
                #logging.info(f'Loss: {loss}')
                total_loss += loss.item()

                logging.debug(f"Microbatch {i + 1}/{accumulation_steps}: Forward pass completed. Time taken: {time.time() - batch_start_time:.2f} seconds")
                backprop_start_time = time.time()
                # Backward pass and accumulate gradients
                loss.backward()
                list_of_params = [param for param in model.parameters()]
                #logging.info(f'Params: {list_of_params}')
                list_of_grads = [param.grad for param in model.parameters()]
                #logging.info(f'Grads: {list_of_grads}')

                with torch.no_grad():
                    for j, param in enumerate(model.parameters()):
                        if param.grad is not None:
                            grads_accumulated[j] += param.grad
                # Clear gradients for next accumulation step
                model.zero_grad()
                
                self.postprocess_model_after_batch(model)

                # Delete intermediate variables
                del loss, microbatch_inputs, microbatch_targets
                torch.cuda.empty_cache()

                logging.debug(f"Microbatch {i + 1}/{accumulation_steps}: Backward pass completed. Time taken: {time.time() - backprop_start_time:.2f} seconds")

            except Exception as e:
                logging.error(f"Error processing microbatch {i + 1}/{accumulation_steps}: {e}", exc_info=True)
                raise  # Re-raise the exception after logging

        # Normalize accumulated gradients
        with torch.no_grad():
            for grad in grads_accumulated:
                grad.div_(accumulation_steps)
        updates = torch.cat([grad.view(-1) for grad in grads_accumulated])
        loss = total_loss / accumulation_steps

        logging.info(f"Model task completed. Total loss: {loss:.4f}. Total time taken: {time.time() - start_time:.2f} seconds")
        logging.info(f'Updates: {updates}')
        return updates, loss
    
    def compile_model(self, model: torch.nn.Module) -> torch.nn.Module:
        model = torch.compile(model)
        _ = self.forward(model, self.get_dummy_input())
        return model

    @abstractmethod
    def get_dummy_input(self):
        pass
    
    def model_to_tensor(self, model: torch.nn.Module) -> torch.Tensor:
        params = list(model.parameters())
        return torch.cat([p.view(-1) for p in params])

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

class FairscaleModelAdapter(ModelAdapter):
    def initialize_environment(self, backend='nccl'):
        logging.info("Initializing distributed environment")
        self.initialize_distributed_environment(backend)
        initialize_model_parallel(model_parallel_size_=1)

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
            1,  # Lower bound for random integers (inclusive)
            self.model_config.get_vocab_size(),  # Upper bound for random integers (exclusive)
            (batch_size, self.model_config.get_max_seq_len())  # Shape: (batch_size, max_seq_len)
        ).to(device)
        print(f'Dummy input: {value}')
        return value
    
    def get_top_token(self, logits):
        return torch.argmax(logits, dim=-1).cpu().numpy().tolist()[0]

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
        input_tensor = torch.tensor([[2.0, 3.0]]).to(device)  # Batch size of 1, two numbers per input
        return input_tensor

class LlamaModelAdapter(TransformerModelAdapter, FairscaleModelAdapter):
    def get_forward_kwargs(self):
        return {'start_pos': 0}

class NanoGPTModelAdapter(TransformerModelAdapter):
    def forward_and_loss(self, model, inputs, targets):
        return model.forward(inputs, targets)[1]
    
    def forward(self, model, inputs):
        return model.forward(inputs)[0]
