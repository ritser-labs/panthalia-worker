"""
rl3.py

An RL-like plugin that:
  1) Uses a small policy network for (state_dim -> action_dim).
  2) Generates synthetic data if no input is provided, ensuring data is available for download.
  3) Minimizes overhead for demonstration.
"""

import math
import random
import logging
import torch
import torch.nn as nn
import torch.optim as optim

# Import the chunked DCT encoding utility
from .util import demo

# Import the device from the .device module
from .device import device

from .adapters.model_adapter import ModelAdapter
from .adapters.default_sot_adapter import DefaultSOTAdapter
from .adapters.plugins import StandardPlugin

################################################################
# 1) Basic RL Model
################################################################
class RLPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim)
        )

    def forward(self, x):
        # Returns raw logits
        return self.fc(x)

################################################################
# 2) RLModelConfig
################################################################
class RLModelConfig:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def create_model(self):
        return RLPolicy(self.state_dim, self.action_dim)

################################################################
# 3) RLModelAdapter with chunked-DCT encoding
################################################################
class RLModelAdapter(ModelAdapter):
    """
    The aggregator expects partial updates in chunked DCT int8 form.
    So we do:
      - forward/backward => get gradients,
      - chunked_dct_encode_int8(grad_vector),
      - return freq_idxs, freq_vals_int8, etc.
    """

    def __init__(self, model_config):
        super().__init__(model_config)
        self.model_config = model_config

        # Initialize flattened parameters on the specified device.
        self._cached_param = self.init_tensor(zero_init=False)
        self.policy = self.model_config.create_model().to(device)

        # Store progressive quantization error.
        self._quant_error = None

        # RL data buffers.
        self.states = None
        self.actions = None
        self.rewards = None

        # For chunked DCT.
        self.chunk_shape = torch.tensor([32, 32], dtype=torch.int64, device=device)  # or pick a shape
        self.k = torch.tensor(2, dtype=torch.int64, device=device)                   # or pick a frequency limit

    def get_dummy_input(self):
        return torch.randn(1, self.model_config.state_dim, device=device)

    def model_to_tensor(self, model: nn.Module) -> torch.Tensor:
        pieces = []
        for p in model.parameters():
            pieces.append(p.view(-1))
        return torch.cat(pieces)

    def tensor_to_model(self, tensor: torch.Tensor, existing_model=None) -> nn.Module:
        if existing_model is None:
            model = self.model_config.create_model().to(device)
        else:
            model = existing_model
        ptr = 0
        for p in model.parameters():
            numel = p.numel()
            p.data.copy_(tensor[ptr : ptr + numel].view_as(p))
            ptr += numel
        return model

    def init_tensor(self, zero_init=False) -> torch.Tensor:
        net = self.model_config.create_model().to(device)
        if zero_init:
            for p in net.parameters():
                p.data.zero_()
        else:
            for p in net.parameters():
                if p.dim() >= 2:
                    nn.init.kaiming_uniform_(p, a=0)
                else:
                    nn.init.zeros_(p)
        return self.model_to_tensor(net)

    async def load_input_tensor(self, input_tensor_dict):
        """
        Expects a list of (stateList, actionInt, rewardFloat).
        If empty or invalid, synthetic data is generated so that data is available for download.
        """
        if not isinstance(input_tensor_dict, list) or len(input_tensor_dict) == 0:
            logging.warning("[RLAdapter] load_input_tensor => empty input provided. Generating synthetic data.")
            bsz = 16
            self.states = torch.randn(bsz, self.model_config.state_dim, device=device)
            self.actions = torch.randint(0, self.model_config.action_dim, (bsz,), device=device)
            self.rewards = torch.randn(bsz, device=device)
            return

        valid_states = []
        valid_actions = []
        valid_rewards = []

        for sample in input_tensor_dict:
            if not isinstance(sample, (list, tuple)) or len(sample) < 3:
                continue
            s, a, r = sample
            try:
                st_val = [float(x) for x in s]
                a_val = int(a)
                r_val = float(r)
            except:
                continue
            valid_states.append(torch.tensor(st_val, dtype=torch.float32, device=device))
            valid_actions.append(torch.tensor(a_val, dtype=torch.long, device=device))
            valid_rewards.append(torch.tensor(r_val, dtype=torch.float32, device=device))

        if len(valid_states) == 0:
            logging.info("[RLAdapter] No valid samples found in input. Generating synthetic data.")
            bsz = 16
            self.states = torch.randn(bsz, self.model_config.state_dim, device=device)
            self.actions = torch.randint(0, self.model_config.action_dim, (bsz,), device=device)
            self.rewards = torch.randn(bsz, device=device)
            return

        self.states = torch.stack(valid_states)
        self.actions = torch.stack(valid_actions)
        self.rewards = torch.stack(valid_rewards)

    async def execute_step(self, task_params: dict, step_idx: int):
        """
        Produce real chunked DCT int8 partial updates:
          1) Rebuild model from self._cached_param.
          2) Forward/backward pass to compute gradients.
          3) Encode gradients via chunked DCT.
          4) Return the encoded gradient and loss.
        """
        # At this point self.states, self.actions, and self.rewards are guaranteed to be available.
        model = self.tensor_to_model(self._cached_param, existing_model=self.policy)

        lr = task_params.get("learning_rate", 1e-3)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.zero_grad()

        # Forward pass to compute log probabilities and compute RL loss.
        logits = model(self.states)
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen_log_probs = log_probs.gather(1, self.actions.unsqueeze(1)).squeeze()
        loss = - (chosen_log_probs * self.rewards).mean()

        loss.backward()
        optimizer.step()

        # Update the cached parameters.
        new_params = self.model_to_tensor(model)
        # The applied gradient is (new_params - old_params).
        grad_vector = (new_params - self._cached_param)
        self._cached_param = new_params

        # Chunked DCT encode the gradient vector.
        freq_idxs, freq_vals_i8, freq_scales, freq_zp, new_error, pad_count = demo.chunked_dct_encode_int8(
            grad_vector,
            chunk_shape=self.chunk_shape,
            k=self.k,
            prev_error=self._quant_error,  # optional progressive error
            norm='ortho'
        )
        self._quant_error = new_error

        encoded_grad = {
            "freq_idxs":        freq_idxs,
            "freq_vals_int8":   freq_vals_i8,
            "freq_scales":      freq_scales,
            "freq_zero_points": freq_zp,
            "chunk_shape":      self.chunk_shape,
            "orig_shape":       torch.tensor(grad_vector.shape, dtype=torch.int64, device=device),
            "pad_count":        torch.tensor(pad_count, dtype=torch.int64, device=device),
        }
        return (encoded_grad, float(loss.item()))

    async def apply_diff(self, diff_tensor: torch.Tensor):
        # Simply add the provided difference to the cached parameters.
        self._cached_param += diff_tensor

    async def decode_and_apply_diff(self, diff_data: dict):
        """
        Decode the provided diff data (chunked DCT int8 format) and apply it to the cached parameters.
        """
        diff_tensor = await demo.decode_diff(diff_data)
        await self.apply_diff(diff_tensor)

    def compile_model(self, model: nn.Module) -> nn.Module:
        return model

################################################################
# 4) RLDataLoader for indefinite RL data (Optional)
################################################################
class RLDataLoader:
    """
    If you want your aggregator to generate data, it can use this class.
    Otherwise, the synthetic fallback in load_input_tensor now guarantees data is available.
    """
    def __init__(self, batch_size, state_dim, action_dim, num_batches=None):
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_batches = num_batches
        self.counter = 0

    async def initialize_dataset(self):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.num_batches is not None and self.counter >= self.num_batches:
            raise StopAsyncIteration
        self.counter += 1

        # Produce random states, actions, rewards on the specified device.
        states = torch.randn(self.batch_size, self.state_dim, device=device)
        actions = torch.randint(0, self.action_dim, (self.batch_size,), device=device)
        rewards = torch.randn(self.batch_size, device=device)

        batch = []
        for i in range(self.batch_size):
            s = states[i].tolist()
            a = int(actions[i].item())
            r = float(rewards[i].item())
            batch.append((s, a, r))
        return batch

################################################################
# 6) Instantiate & export
################################################################
state_dim = 4
action_dim = 2
batch_size = 8

# If you want infinite data, set num_batches=None.
dataset = RLDataLoader(batch_size, state_dim, action_dim, num_batches=None)
model_config = RLModelConfig(state_dim, action_dim)
model_adapter = RLModelAdapter(model_config)

sot_adapter = DefaultSOTAdapter(
    model_adapter=model_adapter,
    dataset=dataset,
    state_dir="/tmp/rl_state",
    tensor_version_interval=10
)

# FIX: Pass tensor values for chunk_shape and k to StandardPlugin so that the hyperparameters include these keys.
exported_plugin = StandardPlugin(
    model_adapter,
    model_config,
    sot_adapter,
    dataset,
    num_steps=1,
    examples_per_accumulation=1,
    accumulations_per_step=1,
    outer_max_lr=1e-3,
    outer_min_lr=1e-3,
    outer_weight_decay=0.0,
    tensor_version_interval=10,
    max_concurrent_iterations=1,
    chunk_shape=torch.tensor((64, 64), dtype=torch.int64),
    k=torch.tensor(2, dtype=torch.int64)
)

# Ensure the SOT adapter can get hyperparameters from the plugin.
sot_adapter.hyperparams_getter = exported_plugin.get_sot_learning_hyperparameters
