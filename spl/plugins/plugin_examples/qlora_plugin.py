"""
qlora_plugin.py

A fully functional plugin that fine-tunes a HuggingFace LLM in 4-bit precision with QLoRA
on the HuggingFace IMDB dataset, returning partial updates for the aggregator. This code:

- Loads a base model (in 4-bit) and applies LoRA adapters (trainable parameters).
- Slices the dataset for each aggregator step, performing multiple micro-steps (accumulations).
- Produces chunked-DCT partial gradients, exactly as in rl3.py / simple_plugin.py.
- Streams data from the IMDB dataset for indefinite training (or until exhausted).
- Implements the standard aggregator interface used by the aggregator code.

Place this file at `spl/plugins/plugin_examples/qlora_plugin.py`.
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import asyncio

# SPL framework imports from your existing codebase:
from .adapters.model_adapter import ModelAdapter
from .adapters.default_sot_adapter import DefaultSOTAdapter
from .adapters.plugins import StandardPlugin
from .tokenizer import CharacterLevelTokenizer  # Example import; you can replace if you prefer

# Chunked-DCT partial update utilities
from .util import demo
from .device import device

# HuggingFace + PEFT (QLoRA) dependencies: make sure you have installed:
#   pip install transformers peft accelerate bitsandbytes datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# HF dataset for demonstration
from datasets import load_dataset


################################################################
# 0) Logging Setup
################################################################
logging.basicConfig(level=logging.INFO)


################################################################
# 1) QLoraModelConfig
################################################################
class QLoraModelConfig:
    """
    Holds relevant hyperparameters for QLoRA-based models, including:
    - base_model_name: the checkpoint to load in 4-bit
    - LoRA config: lora_r, lora_alpha, lora_dropout, etc.
    - target_modules: which model modules get LoRA injected
    - hf_dtype: model dtype used for loading (e.g. torch.float16 or torch.float32)
    """

    def __init__(
        self,
        base_model_name: str = "meta-llama/Llama-2-7b-hf",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules=("q_proj", "k_proj", "v_proj"),
        hf_dtype=torch.float16,
    ):
        self.base_model_name = base_model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.hf_dtype = hf_dtype


################################################################
# 2) QLoraModelAdapter
################################################################
class QLoraModelAdapter(ModelAdapter):
    """
    A ModelAdapter that:
      - Loads a 4-bit quantized HuggingFace model and applies QLoRA LoRA adapters
      - Maintains a flattened LoRA weight vector (`_cached_param`) for aggregator updates
      - Slices the loaded input batch according to aggregator step_idx
      - Accumulates multiple micro-steps (accumulations_per_step) if requested
      - Encodes partial updates with chunked-DCT

    The aggregator calls e.g. execute_step(task_params, step_idx) repeatedly, so we
    handle each portion of data for that step, returning partial updates.
    """

    def __init__(self, model_config: QLoraModelConfig):
        super().__init__(model_config)
        self.model_config = model_config
        self._peft_model = None
        self._cached_param = None  # Flattened LoRA adapter param vector
        self._quant_error = None   # Progressive quant. error for chunked-DCT
        self.inputs = None
        self.labels = None

    def initialize_environment(self, *args, **kwargs):
        """
        Called by plugin_server at start. We'll load the 4-bit model & attach LoRA here.
        """
        logging.info("[QLoraModelAdapter] Initializing QLoRA environment...")

        if self._peft_model is None:
            # 1) Load model in 4-bit
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_config.base_model_name,
                load_in_4bit=True,
                device_map={"": 0},  # load on GPU if available
                torch_dtype=self.model_config.hf_dtype
            )
            # 2) Prepare for 4-bit training
            base_model = prepare_model_for_kbit_training(base_model)
            # 3) Create LoRA config
            lora_config = LoraConfig(
                r=self.model_config.lora_r,
                lora_alpha=self.model_config.lora_alpha,
                target_modules=self.model_config.target_modules,
                lora_dropout=self.model_config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            # 4) Wrap the base model with LoRA
            self._peft_model = get_peft_model(base_model, lora_config)

        logging.info("[QLoraModelAdapter] QLoRA model loaded successfully.")

    def get_dummy_input(self):
        """
        Return a small dummy input to test shape or warm up. Not used heavily in aggregator, but required by interface.
        """
        return torch.tensor([[42, 43, 44, 45]], dtype=torch.long, device=device)

    def compile_model(self, model: nn.Module) -> nn.Module:
        """
        Optionally compile the model with torch.compile for speed. Here we just return it.
        """
        return model

    def model_to_tensor(self, model: nn.Module) -> torch.Tensor:
        """
        Flatten the trainable LoRA parameters (only those with requires_grad=True).
        """
        trainable_params = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                trainable_params.append(p.view(-1))
        if not trainable_params:
            raise ValueError("No trainable parameters found in QLoRA model.")
        return torch.cat(trainable_params).to(device)

    def tensor_to_model(self, tensor: torch.Tensor, existing_model=None) -> nn.Module:
        """
        Write a flattened param vector back into the model's trainable LoRA parameters.
        Keeps the same order as model_to_tensor.
        """
        if existing_model is None:
            existing_model = self._peft_model
        pointer = 0
        for n, p in existing_model.named_parameters():
            if p.requires_grad:
                numel = p.numel()
                chunk = tensor[pointer : pointer + numel]
                pointer += numel
                p.data.copy_(chunk.view_as(p))
        return existing_model

    def init_tensor(self, zero_init: bool = False) -> torch.Tensor:
        """
        Initialize LoRA weights. If zero_init=True, we zero them. Otherwise we keep the existing init.
        """
        if self._peft_model is None:
            self.initialize_environment()

        for n, p in self._peft_model.named_parameters():
            if p.requires_grad:
                if zero_init:
                    p.data.zero_()

        param_tensor = self.model_to_tensor(self._peft_model)
        return param_tensor

    async def load_input_tensor(self, input_tensor_dict):
        """
        Aggregator calls this to supply the next chunk of data (from the dataset).
        We store them in self.inputs / self.labels to be used in `execute_step`.
        """
        # We expect: {"input_ids": [B, T], "labels": [B, T]}
        in_ids = input_tensor_dict.get('input_ids')
        lbls = input_tensor_dict.get('labels')
        if in_ids is None or lbls is None:
            logging.warning("[QLoraModelAdapter] Received invalid input (missing input_ids or labels).")
            return

        self.inputs = in_ids.to(device)
        self.labels = lbls.to(device)
        logging.info(
            f"[QLoraModelAdapter] Loaded new input batch => inputs shape {self.inputs.shape}, "
            f"labels shape {self.labels.shape}"
        )

    async def execute_step(self, task_params: dict, step_idx: int):
        """
        1) Rebuild model from self._cached_param
        2) Slice the loaded batch for this step if needed
        3) Accumulate multiple micro-steps if 'accumulations_per_step' > 1
        4) Compute final gradient, chunked-DCT encode, return partial update + loss
        """

        if self._cached_param is None:
            raise ValueError("No param is set. Did you call init_tensor first?")

        # The aggregator typically calls execute_step multiple times, each with a new step_idx
        # or calls load_input_tensor in between. We'll do partial slicing below.

        if self.inputs is None or self.labels is None:
            logging.warning("[QLoraModelAdapter] No loaded batch => returning no partial update.")
            return (None, 0.0)

        steps = task_params.get('steps', 1)
        accum_steps = task_params.get('accumulations_per_step', 1)

        # figure out sub-batch for this step:
        total_size = self.inputs.size(0)
        batch_size = total_size // steps if steps > 0 else total_size
        start_idx = step_idx * batch_size
        end_idx = min(start_idx + batch_size, total_size)
        if start_idx >= total_size:
            # If aggregator overshoots, no data left
            logging.info(f"[QLoraModelAdapter] step_idx={step_idx} => no data left.")
            return (None, 0.0)

        x_step = self.inputs[start_idx:end_idx].detach()
        y_step = self.labels[start_idx:end_idx].detach()

        # Rebuild model
        model = self.tensor_to_model(self._cached_param, existing_model=self._peft_model)
        model.train()

        # We'll accumulate gradients over 'accum_steps' micro-batches
        # If the sub-batch is smaller than accum_steps, we do single pass. Otherwise we subdivide further.
        # For simplicity, we chunk x_step into accum_steps micro-batches if possible.
        batch_len = x_step.size(0)
        micro_batch_size = max(1, batch_len // accum_steps)

        accum_grads = torch.zeros_like(self._cached_param, dtype=torch.float32, device=device)
        total_loss = 0.0

        for micro_idx in range(accum_steps):
            mb_start = micro_idx * micro_batch_size
            mb_end = min(mb_start + micro_batch_size, batch_len)
            if mb_start >= batch_len:
                break

            mb_x = x_step[mb_start:mb_end]
            mb_y = y_step[mb_start:mb_end]
            # skip micro-batches that have zero length
            if mb_x.size(0) == 0:
                continue

            # Forward/backward
            model.zero_grad()
            outputs = model(mb_x, labels=mb_y)
            loss = outputs.loss
            # We'll accumulate the final gradient in accum_grads
            # If we want an average, we do loss / accum_steps
            loss = loss / accum_steps
            loss.backward()

            grads_list = []
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    grads_list.append(p.grad.view(-1))
                elif p.requires_grad:
                    # If p.grad is None for some reason, treat it as zero
                    grads_list.append(torch.zeros_like(p).view(-1))

            partial_grad = torch.cat(grads_list, dim=0).detach()
            accum_grads += partial_grad

            total_loss += float(loss.item())  # We add the average piece for each micro-step

        # Now accum_grads is the total gradient for this aggregator step
        # We apply it to the cached LoRA params => new_params
        old_params = self._cached_param
        new_params = old_params + accum_grads
        self._cached_param = new_params

        # chunked-DCT encode the gradient (accum_grads)
        chunk_shape = task_params.get("chunk_shape", torch.tensor([64, 64], device=device))
        k_val = task_params.get("k", torch.tensor(2, device=device))

        (
            freq_idxs,
            freq_vals_i8,
            freq_scales,
            freq_zp,
            new_error,
            pad_count
        ) = demo.chunked_dct_encode_int8(
            accum_grads,
            chunk_shape=chunk_shape,
            k=k_val,
            prev_error=self._quant_error,
            norm='ortho'
        )
        self._quant_error = new_error

        encoded_grad = {
            "freq_idxs":        freq_idxs,
            "freq_vals_int8":   freq_vals_i8,
            "freq_scales":      freq_scales,
            "freq_zero_points": freq_zp,
            "chunk_shape":      chunk_shape,
            "orig_shape":       torch.tensor(accum_grads.shape, dtype=torch.int64, device=device),
            "pad_count":        torch.tensor(pad_count, dtype=torch.int64, device=device),
        }

        # Return partial update and total_loss
        return (encoded_grad, total_loss)

    async def apply_diff(self, diff_tensor: torch.Tensor):
        """
        Aggregator calls this to apply a diff to the local LoRA param vector.
        """
        if self._cached_param is None:
            raise ValueError("No param is set. Did you call init_tensor first?")
        diff_tensor = diff_tensor.to(self._cached_param.device)
        self._cached_param += diff_tensor

    async def decode_and_apply_diff(self, diff_data: dict):
        """
        Decode chunked-DCT data, then call apply_diff.
        """
        diff_tensor = await demo.decode_diff(diff_data)
        await self.apply_diff(diff_tensor)


################################################################
# 3) Streaming IMDB Data Loader
################################################################
class IMDBDataLoader:
    """
    Minimal example data loader that streams the HuggingFace IMDB dataset. It tokenizes
    each sample for a causal-LM style (labels = input_ids). If aggregator calls repeatedly,
    we produce indefinite batches until the dataset is exhausted.

    We store them in a queue-like approach, or aggregator can call get_batch() => load_input_tensor.
    """

    def __init__(self, tokenizer_name="gpt2", max_length=128, batch_size=4):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if not self.tokenizer.pad_token_id:
            # For GPT2-like tokenizers that do not have a pad_token, reuse EOS
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.batch_size = batch_size
        self._initialized = False
        self._dataset_iter = None

    async def initialize_dataset(self):
        logging.info("[IMDBDataLoader] Initializing streaming dataset from HuggingFace: imdb.")
        ds = load_dataset("imdb", split="train", streaming=True)
        ds = ds.shuffle(seed=0)
        self._dataset_iter = iter(ds)
        self._initialized = True

    def tokenize_fn(self, text):
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # For causal LM, labels = input_ids
        input_ids = enc["input_ids"][0]
        return input_ids, input_ids

    async def __anext__(self):
        """
        Produces a dict: {"input_ids": Tensor, "labels": Tensor} of shape [batch_size, seq_len].
        Raises StopAsyncIteration if exhausted.
        """
        if not self._initialized:
            raise StopAsyncIteration("IMDBDataLoader not initialized. Call initialize_dataset() first.")

        input_list = []
        label_list = []
        for _ in range(self.batch_size):
            try:
                sample = next(self._dataset_iter)
            except StopIteration:
                raise StopAsyncIteration("IMDB dataset exhausted.")
            text = sample["text"]
            in_ids, lbls = self.tokenize_fn(text)
            input_list.append(in_ids)
            label_list.append(lbls)

        batch_in = torch.stack(input_list, dim=0)
        batch_lbl = torch.stack(label_list, dim=0)
        return {"input_ids": batch_in, "labels": batch_lbl}

    def __aiter__(self):
        return self


################################################################
# 4) Create the DefaultSOTAdapter
################################################################
sot_adapter = DefaultSOTAdapter(
    model_adapter=None,       # We'll assign the QLoraModelAdapter after creation
    dataset=None,             # We'll assign the IMDB loader after creation
    state_dir="/app/data/qlora_state",
    tensor_version_interval=60
)


################################################################
# 5) Instantiate + Export the Plugin
################################################################
# 5.1) QLoRA config
qlora_config = QLoraModelConfig(
    base_model_name="meta-llama/Llama-2-7b-hf",  # or another 4-bit-friendly model
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=("q_proj","k_proj","v_proj"),
    hf_dtype=torch.float16,
)

# 5.2) Model adapter
qlora_adapter = QLoraModelAdapter(qlora_config)

# 5.3) IMDB dataset loader
imdb_loader = IMDBDataLoader(tokenizer_name="gpt2", max_length=128, batch_size=4)

# 5.4) Link them to the SOT adapter
sot_adapter.model_adapter = qlora_adapter
sot_adapter.dataset = imdb_loader

# 5.5) Create the StandardPlugin
exported_plugin = StandardPlugin(
    model_adapter=qlora_adapter,
    model_config=qlora_config,
    sot_adapter=sot_adapter,
    dataset=imdb_loader,
    num_steps=4,                  # aggregator might run 4 steps per chunk
    examples_per_accumulation=4,  # number of examples we feed in each aggregator step
    accumulations_per_step=1,     # how many micro-steps to accumulate
    tokenizer=None,               # if needed, pass a tokenizer object
    outer_max_lr=1e-4,
    outer_min_lr=1e-5,
    outer_weight_decay=0.01,
    tensor_version_interval=60,
    max_concurrent_iterations=2,
    chunk_shape=torch.tensor((64, 64), dtype=torch.int64),
    k=torch.tensor(2, dtype=torch.int64),
)

# Make sure SOT adapter can fetch the hyperparams from the plugin
sot_adapter.hyperparams_getter = exported_plugin.get_sot_learning_hyperparameters
