from .adapters.dataloader import FineWebDataLoader
from .adapters.model_config import NanoGPTConfig
from .adapters.models.nanogpt import GPT, GPTConfig
from .adapters.model_adapter import NanoGPTModelAdapter
from .adapters.plugins import StandardPlugin
from .tokenizer import GPT2Tokenizer
import math
import os

tokenizer = GPT2Tokenizer()

model_params = GPTConfig(
    block_size=1024,
    vocab_size=tokenizer.get_vocab_size(),
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    bias=False,
    pad_token_id=tokenizer.pad_id
)

model_config = NanoGPTConfig(tokenizer, model_params)

model_adapter = NanoGPTModelAdapter(model_config)

NUM_STEPS = 1500

EXAMPLES_PER_STEP = 14

dataset = FineWebDataLoader(
    model_config,
    buffer_size=16,
    batch_size=NUM_STEPS * EXAMPLES_PER_STEP,
    max_seq_len=model_params.block_size
)

exported_plugin = StandardPlugin(
    model_adapter,
    model_config,
    dataset,
    tokenizer,
    num_steps=NUM_STEPS,
    examples_per_step=EXAMPLES_PER_STEP,
    outer_max_lr=1,
    outer_min_lr=1,
    outer_weight_decay=0.0,
    tensor_version_interval=770,
    expected_worker_time=760,
    max_concurrent_iterations=14,
    inner_max_lr=6e-4,
    inner_min_lr=6e-4,
    inner_T_0=200,
    inner_weight_decay=0.1,
    preload_batch_count=4
)