# spl/plugins/plugin_examples/plugin.py

from .adapters.dataloader import ShakespeareDataLoader
from .adapters.model_config import NanoGPTConfig
from .adapters.models.nanogpt import GPT, GPTConfig
from .adapters.model_adapter import NanoGPTModelAdapter
from .adapters.default_sot_adapter import DefaultSOTAdapter
from .adapters.plugins import StandardPlugin
from .tokenizer import CharacterLevelTokenizer

import torch

################################################################
# 1) Set up model, dataset, etc.
################################################################
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

ACCUMULATIONS_PER_STEP = 1
NUM_STEPS = 2 # small number for testing
EXAMPLES_PER_ACCUMULATION = 64

dataset = ShakespeareDataLoader(
    model_config,
    buffer_size=16,
    max_seq_len=model_params.block_size,
    batch_size=NUM_STEPS * EXAMPLES_PER_ACCUMULATION * ACCUMULATIONS_PER_STEP,
)

################################################################
# 2) Create the "DefaultSOTAdapter"
################################################################
TENSOR_VERSION_INTERVAL = 10
sot_adapter = DefaultSOTAdapter(
    model_adapter=model_adapter,
    dataset=dataset,
    state_dir="/app/data/state",
    tensor_version_interval=TENSOR_VERSION_INTERVAL
)

################################################################
# 3) Create the plugin with the SOT adapter
################################################################
exported_plugin = StandardPlugin(
    model_adapter,         # model adapter
    model_config,          # model config
    sot_adapter,           # <--- we pass the "sot_adapter" here
    dataset,               # dataset
    tokenizer,
    num_steps=NUM_STEPS,
    examples_per_accumulation=EXAMPLES_PER_ACCUMULATION,
    accumulations_per_step=ACCUMULATIONS_PER_STEP,
    outer_max_lr=1e-4,
    outer_min_lr=1e-4,
    outer_weight_decay=0.01,
    tensor_version_interval=TENSOR_VERSION_INTERVAL,
    expected_worker_time=3,
    max_concurrent_iterations=4,
    chunk_shape=torch.tensor((64, 64)),
    k=torch.tensor((2)),
)

sot_adapter.hyperparams_getter = exported_plugin.get_sot_learning_hyperparameters