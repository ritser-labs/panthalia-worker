import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import save_to_disk, load_from_disk, load_layer_state_dict, save_layer_state_dict, model_args
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

def loss_task(logits_file, targets_file, loss_file, logits_grad_file):
    logits = load_from_disk(logits_file)
    targets = load_from_disk(targets_file)
    
    tokenizer = Tokenizer(encoding_name='cl100k_base')
    pad_id = tokenizer.pad_id
    
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=pad_id)
    save_to_disk(loss.item(), loss_file)
    
    logits.retain_grad()
    loss.backward()
    save_to_disk(logits.grad, logits_grad_file)

def apply_adamw(layer_idx, grads, learning_rate, beta1, beta2, epsilon, weight_decay, t):
    if layer_idx == -1:
        state_dict_file = "data/layer_fc.pt"
        m_file = "data/adam_m_fc.pt"
        v_file = "data/adam_v_fc.pt"
    elif layer_idx == -2:
        state_dict_file = "data/embedding.pt"
        m_file = "data/adam_m_embedding.pt"
        v_file = "data/adam_v_embedding.pt"
    else:
        state_dict_file = f"data/layer_{layer_idx}.pt"
        m_file = f"data/adam_m_{layer_idx}.pt"
        v_file = f"data/adam_v_{layer_idx}.pt"

    state_dict = load_layer_state_dict(state_dict_file)
    if state_dict is None:
        raise ValueError(f"Failed to load state dict for layer {layer_idx}")

    m = load_from_disk(m_file)
    v = load_from_disk(v_file)
    
    if m is None:
        m = [torch.zeros_like(param) for param in state_dict.values()]
    if v is None:
        v = [torch.zeros_like(param) for param in state_dict.values()]

    for i, param in enumerate(state_dict.values()):
        if param.requires_grad:
            # AdamW weight decay
            param.data -= learning_rate * weight_decay * param.data

            # AdamW updates
            m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
            v[i] = beta2 * v[i] + (1 - beta2) * (grads[i] ** 2)

            m_hat = m[i] / (1 - beta1 ** t)
            v_hat = v[i] / (1 - beta2 ** t)

            param.data -= learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)

    save_layer_state_dict(state_dict, state_dict_file)
    save_to_disk(m, m_file)
    save_to_disk(v, v_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["embed", "forward", "backward", "final_logits", "final_logits_backward", "embed_backward", "loss", "apply_adamw"])
    parser.add_argument("--layer_idx", type=int, required=False)
    parser.add_argument("--inputs", type=str, required=False)
    parser.add_argument("--error", type=str, required=False)
    parser.add_argument("--batch", type=str, required=False)
    parser.add_argument("--state_dict", type=str, required=False)
    parser.add_argument("--logits", type=str, required=False)
    parser.add_argument("--targets", type=str, required=False)
    parser.add_argument("--embedding_file", type=str, required=False)
    parser.add_argument("--logits_file", type=str, required=False)
    parser.add_argument("--error_output_file", type=str, required=False)
    parser.add_argument("--loss_file", type=str, required=False)
    parser.add_argument("--logits_grad_file", type=str, required=False)
    parser.add_argument("--grads", type=str, required=False)
    parser.add_argument("--learning_rate", type=float, required=False)
    parser.add_argument("--beta1", type=float, required=False)
    parser.add_argument("--beta2", type=float, required=False)
    parser.add_argument("--epsilon", type=float, required=False)
    parser.add_argument("--weight_decay", type=float, required=False)
    parser.add_argument("--t", type=int, required=False)
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
        loss_task(args.logits, args.targets, args.loss_file, args.logits_grad_file)
    elif args.task == "apply_adamw":
        grads = load_from_disk(args.grads)
        apply_adamw(args.layer_idx, grads, args.learning_rate, args.beta1, args.beta2, args.epsilon, args.weight_decay, args.t)
