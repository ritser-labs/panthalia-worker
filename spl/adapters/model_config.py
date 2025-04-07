from abc import ABC
from .models.llama3 import Transformer  # Assuming this import is still needed
from .models.nanogpt import GPT, GPTConfig  # Import the GPT and GPTConfig classes
from .models.adder import Adder

class BaseModelConfig(ABC):
    def create_model(self):
        return self.model_class(*self.args, **self.kwargs)

class TransformerModelConfig(BaseModelConfig):
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_max_seq_len(self):
        return self.max_seq_len

class LlamaModelConfig(TransformerModelConfig):
    def __init__(self, tokenizer, params):
        self.tokenizer = tokenizer
        self.args = [params]
        self.kwargs = {}
        self.model_class = Transformer
        self.vocab_size = params.vocab_size
        self.max_seq_len = params.max_seq_len
    
    def encode(text):
        return tokenizer.encode(
            text,
            bos=False,
            eos=False,
            allowed_special=set(),
            disallowed_special=(),
        )
    
    def decode(tokens):
        return tokenizer.decode(tokens)


class NanoGPTConfig(TransformerModelConfig):
    def __init__(self, tokenizer, params):
        self.tokenizer = tokenizer
        self.args = [params]
        self.kwargs = {}
        self.model_class = GPT
        self.vocab_size = params.vocab_size
        self.max_seq_len = params.block_size
    
    def encode(text):
        return tokenizer.encode(text)
    
    def decode(tokens):
        return tokenizer.decode(tokens)

class AdderModelConfig(BaseModelConfig):
    def __init__(self):
        self.args = []
        self.kwargs = {}
        self.model_class = Adder