# model_config.py

from abc import ABC
from .llama3 import Transformer  # Assuming this import is still needed
from .nanogpt import GPT, GPTConfig  # Import the GPT and GPTConfig classes

class BaseModelConfig(ABC):
    pass

class TransformerModelConfig(BaseModelConfig):
    def __init__(self, tokenizer, model_args):
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.model_class = Transformer

class NanoGPTConfig(BaseModelConfig):
    def __init__(self, tokenizer, model_args):
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.model_class = GPT
