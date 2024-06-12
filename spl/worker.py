import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import save_to_disk, load_from_disk, load_layer_state_dict, model_args
from tokenizer import Tokenizer
from model import TransformerLayer
import logging

def embed_task(batch_file):
    batch = load_from_disk(batch_file)
    tokenizer = Tokenizer(encoding_name='cl100k_base')
    vocab_size = tokenizer.get_vocab_size()
    
    embedding = nn.Embedding(vocab_size, model_args.dim)
    embedding.load_state_dict(load_layer_state_dict("data/embedding.pt"))
    
    inputs = embedding(batch)
    save_to_disk(inputs, "data/inputs.pt")

def forward_task(layer_idx, inputs_file):
    inputs = load_from_disk(inputs_file)
    state_dict = load_layer_state_dict(f"data/layer_{layer_idx}.pt")
    
    layer = TransformerLayer(model_args.dim, model_args.n_heads, model_args.dim * model_args.ffn_dim_multiplier)
    layer.load_state_dict(state_dict)
    
    outputs = layer(inputs)
    save_to_disk(outputs, f"data/logits_layer_{layer_idx}.pt")

def backward_task(layer_idx, error_file):
    error = load_from_disk(error_file)
    state_dict = load_layer_state_dict(f"data/layer_{layer_idx}.pt")
    
    layer = TransformerLayer(model_args.dim, model_args.n_heads, model_args.dim * model_args.ffn_dim_multiplier)
    layer.load_state_dict(state_dict)
    
    # Forward pass
    inputs = load_from_disk(f"data/inputs_layer_{layer_idx}.pt")
    inputs.requires_grad = True
    outputs = layer(inputs)
    
    # Backward pass
    outputs.backward(error)
    
    grads = [param.grad for param in layer.parameters()]
    save_to_disk((inputs.grad, grads), f"data/error_layer_{layer_idx}.pt")

def final_logits_task(inputs_file):
    inputs = load_from_disk(inputs_file)
    tokenizer = Tokenizer(encoding_name='cl100k_base')
    vocab_size = tokenizer.get_vocab_size()
    state_dict = load_layer_state_dict("data/layer_fc.pt")
    
    if state_dict is None:
        raise ValueError("Failed to load state dict for the fully connected layer")

    fc = nn.Linear(model_args.dim, vocab_size)
    fc.load_state_dict(state_dict)
    
    logits = fc(inputs)
    save_to_disk(logits, "data/logits.pt")

def final_logits_backward_task(error_file):
    error = load_from_disk(error_file)
    tokenizer = Tokenizer(encoding_name='cl100k_base')
    vocab_size = tokenizer.get_vocab_size()
    state_dict = load_layer_state_dict("data/layer_fc.pt")
    
    if state_dict is None:
        raise ValueError("Failed to load state dict for the fully connected layer")

    fc = nn.Linear(model_args.dim, vocab_size)
    fc.load_state_dict(state_dict)
    
    # Forward pass
    inputs = load_from_disk("data/final_inputs.pt")
    inputs.requires_grad = True
    logits = fc(inputs)
    
    # Backward pass
    logits.backward(error)
    
    grads = [param.grad for param in fc.parameters()]
    save_to_disk((inputs.grad, grads), "data/error_layer_fc.pt")

def embed_backward_task(error_file):
    error = load_from_disk(error_file)
    tokenizer = Tokenizer(encoding_name='cl100k_base')
    vocab_size = tokenizer.get_vocab_size()
    
    embedding = nn.Embedding(vocab_size, model_args.dim)
    embedding.load_state_dict(load_layer_state_dict("data/embedding.pt"))
    
    batch = load_from_disk("data/batch.pt")
    embeddings = embedding(batch)
    embeddings.requires_grad = True
    
    # Backward pass
    embeddings.backward(error)
    
    grads = [param.grad for param in embedding.parameters()]
    save_to_disk(grads, "data/embedding_grads.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["embed", "forward", "backward", "final_logits", "final_logits_backward", "embed_backward"])
    parser.add_argument("--layer_idx", type=int, required=False)
    parser.add_argument("--inputs", type=str, required=False)
    parser.add_argument("--error", type=str, required=False)
    parser.add_argument("--batch", type=str, required=False)
    args = parser.parse_args()

    if args.task == "embed":
        embed_task(args.batch)
    elif args.task == "forward":
        forward_task(args.layer_idx, args.inputs)
    elif args.task == "backward":
        backward_task(args.layer_idx, args.error)
    elif args.task == "final_logits":
        final_logits_task(args.inputs)
    elif args.task == "final_logits_backward":
        final_logits_backward_task(args.error)
    elif args.task == "embed_backward":
        embed_backward_task(args.error)
