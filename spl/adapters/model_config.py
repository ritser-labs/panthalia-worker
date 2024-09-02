from abc import ABC
from .models.llama3 import Transformer  # Assuming this import is still needed
from .models.nanogpt import GPT, GPTConfig  # Import the GPT and GPTConfig classes
from .models.adder import Adder

class BaseModelConfig(ABC):
    def create_model(self):
        return self.model_class(*self.args, **self.kwargs)

class TransformerModelConfig(BaseModelConfig):
    def __init__(self, tokenizer, params):
        self.tokenizer = tokenizer
        self.args = [params]
        self.kwargs = {}
        self.model_class = Transformer

class NanoGPTConfig(BaseModelConfig):
    def __init__(self, tokenizer, params):
        self.tokenizer = tokenizer
        self.args = [params]
        self.kwargs = {}
        self.model_class = GPT

class AdderModelConfig(BaseModelConfig):
    def __init__(self):
        self.args = []
        self.kwargs = {}
        self.model_class = Adder