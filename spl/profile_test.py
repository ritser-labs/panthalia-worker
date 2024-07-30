import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from collections import defaultdict
from dataclasses import dataclass
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from common import initialize_distributed_environment, model_args
import os

# Assuming we have the following classes and functions already defined
from model import TransformerBlock, precompute_freqs_cis, RMSNorm
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
transformer_block = TransformerBlock(layer_idx, args).to(device)

# Precompute frequency sinusoidal embeddings and attention mask
freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2, args.rope_theta).to(device)
mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1).to(device)

# Convert input_ids to input embeddings (assuming that input embeddings are already computed)
# For simplicity, using random embeddings; replace with actual input embeddings
input_embeddings = torch.randn(batch_size, seq_len, args.dim).to(device)

# Use torch.compile with torchdynamo for optimization
transformer_block = torch.compile(transformer_block)

# Define the forward function for the transformer block task
def transformer_block_task(inputs, start_pos=0):
    with torch.no_grad():
        outputs = transformer_block(inputs, start_pos, freqs_cis[:seq_len], mask)
    return outputs

# Profile the transformer block forward pass
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with record_function("transformer_block_forward"):
        outputs = torch._dynamo.explain(transformer_block_task)(input_embeddings)
        
    # Synchronize CUDA operations before printing the profiling results
    torch.cuda.synchronize()

# Print the profiling results
print(prof.key_averages().table(row_limit=10))
print(outputs)
