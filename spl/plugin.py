
from .adapters.dataloader import *
from .adapters.model_config import *
import os
from .adapters.model_adapter import *
from .adapters.plugins import StandardPlugin
from .tokenizer import *
import math


BUFFER_SIZE = 100000  # Size of the buffer to shuffle data

tokenizer = CharacterLevelTokenizer()

model_params = GPTConfig(
    block_size=256,
    vocab_size=tokenizer.get_vocab_size(),
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.2,
    bias=True,
    pad_token_id=tokenizer.pad_id
)

model_config = NanoGPTConfig(tokenizer, model_params)

dataset = ShakespeareDataLoader(model_config, buffer_size=BUFFER_SIZE, max_seq_len=model_params.block_size)

model_adapter = NanoGPTModelAdapter(model_config)

NUM_MICROBATCHES = 450 # 256,512

EXAMPLES_PER_MICROBATCH = 32 # 32,512

exported_plugin = StandardPlugin(
    model_adapter,
    model_config,
    dataset,
    tokenizer,
    num_microbatches=NUM_MICROBATCHES,
    example_per_microbatch=EXAMPLES_PER_MICROBATCH
)
