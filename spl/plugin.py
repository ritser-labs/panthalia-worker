
from .adapters.dataloader import *
from .adapters.model_config import *
import os
from .adapters.model_adapter import *
from .adapters.nanogpt import GPTConfig
from .tokenizer import Tokenizer


BUFFER_SIZE = 100000  # Size of the buffer to shuffle data

current_dir = os.path.dirname(os.path.abspath(__file__))


# IMPORTANT: if you change tokenizer, dont forget to comment out the code in tokenizer.py
# that ignores non ascii characters
# and change pad_id
tokenizer_path = os.path.join(current_dir, 'tokenizers', 'char.tiktoken')

tokenizer = Tokenizer(tokenizer_path)


model_args = GPTConfig(
    block_size=256,
    vocab_size=tokenizer.get_vocab_size(),
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.2,
    bias=True
)

model_config = NanoGPTConfig(tokenizer, model_args)

dataset = ShakespeareDataLoader(model_config, buffer_size=BUFFER_SIZE, max_seq_len=model_args.block_size)

model_adapter = NanoGPTModelAdapter(model_config)