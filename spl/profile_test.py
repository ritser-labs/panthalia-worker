import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from collections import defaultdict
from dataclasses import dataclass
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from common import initialize_distributed_environment
import os

# Assuming we have the following classes and functions already defined
from model import VocabParallelEmbedding
from device import device

# Define the model arguments
@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = 30522  # Random vocab size, adjust as needed
    norm_eps: float = 1e-5
    max_seq_len: int = 2048

args = ModelArgs()

# Create random inputs
batch_size = 8
seq_len = 128
input_ids = torch.randint(0, args.vocab_size, (batch_size, seq_len)).to(device)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"

initialize_distributed_environment('nccl')
initialize_model_parallel(model_parallel_size_=1)

# Initialize embedding layer
embedding = VocabParallelEmbedding(args.vocab_size, args.dim).to(device)


# Use torch.compile with torchdynamo for optimization
embedding = torch.compile(embedding)

# Define the forward function for the embedding task
def embed_task(input_ids):
    with torch.no_grad():
        outputs = embedding(input_ids)
    return outputs

# Profile the embedding forward pass
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with record_function("embed_task_forward"):
        outputs = torch._dynamo.explain(embedding)(input_ids)
        
    # Synchronize CUDA operations before printing the profiling results
    torch.cuda.synchronize()

# Print the profiling results
print(prof.key_averages().table(row_limit=10))
print(outputs)