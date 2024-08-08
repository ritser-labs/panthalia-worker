from abc import ABC, abstractmethod
import torch

class ModelAdapter(ABC):
    @abstractmethod
    def get_model() -> torch.nn.Module:
        pass

    @abstractmethod
    def train_loop(model, inputs, targets, accumulation_steps):
        pass


class TransformerModelAdapter(ModelAdapter):
    def get_model() -> torch.nn.Module:
        pass

    def train_loop(model, inputs, targets, accumulation_steps):
        pass