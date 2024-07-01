import json
import logging
import requests
import torch
import torch.nn.functional as F
from collections import defaultdict
from model import TransformerBlock, VocabParallelEmbedding, ColumnParallelLinear, precompute_freqs_cis
from common import model_args, tokenizer, device
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from typing import Optional
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

# Global variables
args = model_args
embedding = None
tensors = defaultdict(lambda: None)
adam_m = defaultdict(lambda: None)
adam_v = defaultdict(lambda: None)
last_gradient_update = defaultdict(lambda: None)
gradient_update_paused = False
model_initialized = False
embedding_initialized = False

freqs_cis = None
mask = None

def initialize_model_and_embedding():
    global model_initialized, embedding_initialized, freqs_cis, mask

    if not model_initialized:
        initialize_model_parallel(model_parallel_size_=1)
        model_initialized = True
    
    if not embedding_initialized:
        vocab_size = tokenizer.get_vocab_size()
        global embedding
        embedding = VocabParallelEmbedding(vocab_size, args.dim).to(device)
        embedding_initialized = True

    freqs_cis = precompute_freqs_cis(
        args.dim // args.n_heads,
        args.max_seq_len * 2,
        args.rope_theta,
    )

    mask = torch.triu(torch.full((args.max_seq_len, args.max_seq_len), float('-inf')), diagonal=1).to(device)

def download_json(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return torch.tensor(data, dtype=torch.long).to(device)
    except requests.RequestException as e:
        logging.error(f"Failed to download JSON from {url}: {e}")
        raise

def download_file(url):
    response = requests.get(url)
    return torch.load(BytesIO(response.content))

def check_for_nans(tensor, name):
    if torch.isnan(tensor).any():
        logging.error(f"NaNs detected in {name}")

def embed_task(batch):
    global embedding
    inputs = embedding(batch)
    check_for_nans(inputs, "embedding outputs")
    tensors['outputs'] = inputs

def forward_task(layer_idx, inputs):
    global freqs_cis, mask, tensors
    if torch.isnan(inputs).any() or torch.isinf(inputs).any():
        raise ValueError(f"NaNs or Infs detected in inputs for layer {layer_idx}")

    layer = tensors[f'layer_{layer_idx}']
    start_pos = 0
    seqlen = inputs.shape[1]
    freqs_cis_slice = freqs_cis[start_pos: start_pos + seqlen]

    outputs = layer(inputs.to(device), start_pos, freqs_cis_slice.to(device), mask.to(device))
    check_for_nans(outputs, f"layer {layer_idx} outputs")
    tensors['outputs'] = outputs

def backward_task(layer_idx, error, inputs, learning_rate, beta1, beta2, epsilon, weight_decay, t):
    global freqs_cis, mask, tensors

    if error is None:
        raise ValueError(f"Error tensor is None")

    layer = tensors[f'layer_{layer_idx}']

    start_pos = 0
    seqlen = inputs.shape[1]
    freqs_cis_slice = freqs_cis[start_pos: start_pos + seqlen]

    outputs = layer(inputs.to(device), start_pos, freqs_cis_slice.to(device), mask.to(device))
    check_for_nans(outputs, f"layer {layer_idx} outputs")

    inputs.requires_grad = True

    outputs.retain_grad()
    outputs.backward(error.to(device), retain_graph=True)

    if inputs.grad is None:
        raise ValueError(f"Gradient for inputs is None after backward pass for layer {layer_idx}")

    check_for_nans(inputs.grad, f"Gradient for inputs in layer {layer_idx}")

    grads = [param.grad for param in layer.parameters() if param.grad is not None]
    logging.debug(f"Gradients for layer {layer_idx}: {grads}")

    for i, grad in enumerate(grads):
        check_for_nans(grad, f"Gradient {i} for layer {layer_idx}")

    tensors['error_output'] = (inputs.grad, grads)
    tensors['grads'] = grads

def loss_task(logits, targets):
    global tensors
    pad_id = tokenizer.pad_id

    batch_size, seq_len, vocab_size = logits.shape
    logits = logits.reshape(batch_size * seq_len, vocab_size)
    targets = targets.reshape(-1)

    loss = F.cross_entropy(logits.to(device), targets.to(device), ignore_index=pad_id)
    tensors['loss'] = loss.item()

    logits.retain_grad()
    loss.backward(retain_graph=True)
    check_for_nans(logits.grad, "logits gradients")
    logging.info(f"Logits gradients for loss: {logits.grad.shape}")

    logits_grad = logits.grad.reshape(batch_size, seq_len, vocab_size)
    tensors['logits_grad'] = logits_grad

def preload_batch():
    url = "https://huggingface.co/datasets/wikipedia/20220301.en/resolve/main/data/train-00000-of-00100.arrow"
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        from datasets import load_from_disk, Dataset
        dataset = Dataset.from_arrow(response.raw)
        return dataset
    except requests.RequestException as e:
        logging.error(f"Failed to preload batch: {e}")
        raise

def main():
    initialize_model_and_embedding()

    # Preload batch
    batch_data = preload_batch()
    batch = []
    max_seq_len = args.max_seq_len
    for example in batch_data:
        tokens = tokenizer.encode(
            example['text'], 
            bos=False, 
            eos=False, 
            allowed_special=set(), 
            disallowed_special=(), 
        )
        if len(tokens) < max_seq_len:
            tokens += [tokenizer.pad_id] * (max_seq_len - len(tokens))
        elif len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
        batch.append(tokens)

    batch = torch.tensor(batch, dtype=torch.long).to(device)

    # Example targets tensor
    targets = torch.tensor(batch, dtype=torch.long).to(device)

    # Embedding Task
    embed_task(batch)
    embedding_output = tensors['outputs']
    logging.info(f"Embedding output shape: {embedding_output.shape}")

    # Forward Task
    layer_idx = 0  # Example layer index
    forward_task(layer_idx, embedding_output)
    forward_output = tensors['outputs']
    logging.info(f"Forward output shape for layer {layer_idx}: {forward_output.shape}")

    # Backward Task
    error_tensor = torch.randn_like(forward_output)  # Example error tensor for backward pass
    backward_task(layer_idx, error_tensor, embedding_output, 0.001, 0.9, 0.999, 1e-8, 0.01, 1)
    logging.info(f"Backward gradients: {tensors['grads']}")

    # Loss Task
    loss_task(forward_output, targets)
    logging.info(f"Loss value: {tensors['loss']}")

if __name__ == "__main__":
    main()
