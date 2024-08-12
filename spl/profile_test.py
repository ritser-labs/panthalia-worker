import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from collections import defaultdict
from dataclasses import dataclass
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from common import initialize_distributed_environment, model_args
import os

# Assuming we have the following classes and functions already defined
from spl.adapters.llama3 import Transformer, precompute_freqs_cis, RMSNorm
from device import device

args = model_args

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

# Initialize the TransformerBlock
layer_idx = 0  # Specify which layer of the Transformer you want to test
transformer = Transformer(model_args).to(device)

# Use torch.compile with torchdynamo for optimization
transformer = torch.compile(transformer)

# Define the forward function for the transformer block task
def transformer_task(inputs, start_pos=0):
    with torch.no_grad():
        outputs = transformer(inputs, start_pos=start_pos)
    return outputs

# Profile the transformer block forward pass
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with record_function("transformer_block_forward"):
        outputs = torch._dynamo.explain(transformer_task)(input_ids)
        
    # Synchronize CUDA operations before printing the profiling results
    torch.cuda.synchronize()

# Print the profiling results
print(prof.key_averages().table(row_limit=10))
print(outputs)
