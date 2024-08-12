from abc import ABC
from spl.adapters.llama3 import Transformer

class BaseModelConfig(ABC):
    pass

class TransformerModelConfig(BaseModelConfig):
    def __init__(self, tokenizer, model_args):
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.model_class = Transformer