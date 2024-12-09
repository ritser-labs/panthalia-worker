from .adapters.dataloader import ShakespeareDataLoader
from .adapters.model_config import NanoGPTConfig
from .adapters.models.nanogpt import GPT, GPTConfig
from .adapters.model_adapter import NanoGPTModelAdapter
from .adapters.plugins import StandardPlugin
from .tokenizer import CharacterLevelTokenizer
import math
import os

tokenizer = CharacterLevelTokenizer()

model_params = GPTConfig(
    block_size=256,
    vocab_size=tokenizer.get_vocab_size(),
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.2,
    bias=False,
    pad_token_id=tokenizer.pad_id
)

model_config = NanoGPTConfig(tokenizer, model_params)

model_adapter = NanoGPTModelAdapter(model_config)

NUM_STEPS = 285

EXAMPLES_PER_STEP = 64

dataset = ShakespeareDataLoader(
    model_config,
    buffer_size=16,
    max_seq_len=model_params.block_size,
    batch_size=NUM_STEPS * EXAMPLES_PER_STEP
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
    tensor_version_interval=36,
    expected_worker_time=31,
    max_concurrent_iterations=4,
    inner_max_lr=0.001,
    inner_min_lr=0.0001,
    inner_T_0=200,
    preload_batch_count=4
)