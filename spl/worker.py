import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import save_to_disk, load_from_disk, load_layer_state_dict, model_args
from tokenizer import Tokenizer
from model import TransformerLayer
import logging

def embed_task(batch_file, embedding_file, inputs_file):
    batch = load_from_disk(batch_file)
    tokenizer = Tokenizer(encoding_name='cl100k_base')
    vocab_size = tokenizer.get_vocab_size()
    
    embedding = nn.Embedding(vocab_size, model_args.dim)
    embedding.load_state_dict(load_layer_state_dict(embedding_file))
    
    inputs = embedding(batch)
    save_to_disk(inputs, inputs_file)

def forward_task(layer_idx, inputs_file, state_dict_file, logits_file):
    inputs = load_from_disk(inputs_file)
    state_dict = load_layer_state_dict(state_dict_file)
    
    layer = TransformerLayer(model_args.dim, model_args.n_heads, model_args.dim * model_args.ffn_dim_multiplier)
    layer.load_state_dict(state_dict)
    
    outputs = layer(inputs)
    save_to_disk(outputs, logits_file)

def backward_task(layer_idx, error_file, inputs_file, state_dict_file, error_output_file):
    error = load_from_disk(error_file)
    state_dict = load_layer_state_dict(state_dict_file)
    
    layer = TransformerLayer(model_args.dim, model_args.n_heads, model_args.dim * model_args.ffn_dim_multiplier)
    layer.load_state_dict(state_dict)
    
    # Forward pass
    inputs = load_from_disk(inputs_file)
    inputs.requires_grad = True
    outputs = layer(inputs)
    
    # Reshape error to match the output shape
    error = error.view(outputs.shape)
    
    # Backward pass
    outputs.backward(error)
    
    grads = [param.grad for param in layer.parameters()]
    save_to_disk((inputs.grad, grads), error_output_file)

def final_logits_task(inputs_file, state_dict_file, logits_file):
    inputs = load_from_disk(inputs_file)
    state_dict = load_layer_state_dict(state_dict_file)
    
    if state_dict is None:
        raise ValueError("Failed to load state dict for the fully connected layer")

    fc = nn.Linear(model_args.dim, state_dict['weight'].shape[0])
    fc.load_state_dict(state_dict)
    
    logits = fc(inputs)
    save_to_disk(logits, logits_file)

def final_logits_backward_task(error_file, inputs_file, state_dict_file, error_output_file):
    error = load_from_disk(error_file)
    state_dict = load_layer_state_dict(state_dict_file)
    
    if state_dict is None:
        raise ValueError("Failed to load state dict for the fully connected layer")

    fc = nn.Linear(model_args.dim, state_dict['weight'].shape[0])
    fc.load_state_dict(state_dict)
    
    # Forward pass
    inputs = load_from_disk(inputs_file)
    inputs.requires_grad = True
    logits = fc(inputs)
    
    # Backward pass
    logits.backward(error)
    
    grads = [param.grad for param in fc.parameters()]
    save_to_disk((inputs.grad, grads), error_output_file)

def embed_backward_task(error_file, batch_file, embedding_file, error_output_file):
    error = load_from_disk(error_file)
    tokenizer = Tokenizer(encoding_name='cl100k_base')
    vocab_size = tokenizer.get_vocab_size()
    
    embedding = nn.Embedding(vocab_size, model_args.dim)
    embedding.load_state_dict(load_layer_state_dict(embedding_file))
    
    batch = load_from_disk(batch_file)
    embeddings = embedding(batch)
    embeddings.retain_grad()  # Ensure gradients are retained
    
    # Reshape error to match the embeddings
    error = error.view(embeddings.shape)  # Ensure the gradient tensor matches the output tensor shape
    
    # Backward pass
    embeddings.backward(error)
    
    grads = [param.grad for param in embedding.parameters()]
    save_to_disk(grads, error_output_file)

def loss_task(logits_file, targets_file, gradient_accumulation_steps, loss_file, logits_grad_file):
    logits = load_from_disk(logits_file)
    targets = load_from_disk(targets_file)
    
    tokenizer = Tokenizer(encoding_name='cl100k_base')
    pad_id = tokenizer.pad_id
    
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=pad_id)
    loss = loss / gradient_accumulation_steps
    save_to_disk(loss.item(), loss_file)
    
    logits.retain_grad()
    loss.backward()
    save_to_disk(logits.grad, logits_grad_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["embed", "forward", "backward", "final_logits", "final_logits_backward", "embed_backward", "loss"])
    parser.add_argument("--layer_idx", type=int, required=False)
    parser.add_argument("--inputs", type=str, required=False)
    parser.add_argument("--error", type=str, required=False)
    parser.add_argument("--batch", type=str, required=False)
    parser.add_argument("--state_dict", type=str, required=False)
    parser.add_argument("--logits", type=str, required=False)
    parser.add_argument("--targets", type=str, required=False)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=False)
    parser.add_argument("--embedding_file", type=str, required=False)
    parser.add_argument("--logits_file", type=str, required=False)
    parser.add_argument("--error_output_file", type=str, required=False)
    parser.add_argument("--loss_file", type=str, required=False)
    parser.add_argument("--logits_grad_file", type=str, required=False)
    args = parser.parse_args()

    if args.task == "embed":
        embed_task(args.batch, args.embedding_file, args.inputs)
    elif args.task == "forward":
        forward_task(args.layer_idx, args.inputs, args.state_dict, args.logits_file)
    elif args.task == "backward":
        backward_task(args.layer_idx, args.error, args.inputs, args.state_dict, args.error_output_file)
    elif args.task == "final_logits":
        final_logits_task(args.inputs, args.state_dict, args.logits_file)
    elif args.task == "final_logits_backward":
        final_logits_backward_task(args.error, args.inputs, args.state_dict, args.error_output_file)
    elif args.task == "embed_backward":
        embed_backward_task(args.error, args.batch, args.embedding_file, args.error_output_file)
    elif args.task == "loss":
        loss_task(args.logits, args.targets, args.gradient_accumulation_steps, args.loss_file, args.logits_grad_file)
