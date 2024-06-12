import torch
import os
from tokenizer import Tokenizer

class ModelArgs:
    def __init__(self, vocab_size, dim, n_layers, n_heads, ffn_dim_multiplier):
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ffn_dim_multiplier = ffn_dim_multiplier

tokenizer = Tokenizer(encoding_name='cl100k_base')

# Define global model_args
model_args = ModelArgs(
    vocab_size=tokenizer.get_vocab_size(),  # Example value, update as needed
    dim=512,
    n_layers=6,
    n_heads=8,
    ffn_dim_multiplier=1
)

def save_to_disk(data, filename):
    torch.save(data, filename)
    print(f"Saved to {filename}")

def load_from_disk(filename):
    if os.path.exists(filename):
        data = torch.load(filename)
        print(f"Loaded from {filename}")
        return data
    else:
        print(f"File {filename} does not exist")
        return None

def save_layer_state_dict(state_dict, filename):
    torch.save(state_dict, filename)
    print(f"Layer state dict saved to {filename}")

def load_layer_state_dict(filename):
    if os.path.exists(filename):
        state_dict = torch.load(filename)
        print(f"Layer state dict loaded from {filename}")
        return state_dict
    else:
        print(f"File {filename} does not exist")
        return None
