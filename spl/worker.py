import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import save_to_disk, load_from_disk, load_layer_state_dict, save_layer_state_dict, model_args, tokenizer
from tokenizer import Tokenizer
from model import TransformerBlock, VocabParallelEmbedding, ColumnParallelLinear, RMSNorm, precompute_freqs_cis
import logging
import os
import torch.distributed as dist
from fairscale.nn.model_parallel.initialize import initialize_model_parallel, model_parallel_is_initialized

logging.basicConfig(level=logging.DEBUG)

def check_for_nans(tensor, name):
    if torch.isnan(tensor).any():
        logging.error(f"NaNs detected in {name}")

def initialize_distributed_environment():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

def embed_task(batch_file, embedding_file, inputs_file):
    batch = load_from_disk(batch_file)
    vocab_size = tokenizer.get_vocab_size()

    logging.debug(f"Batch for embedding (token IDs): {batch}")

    embedding = VocabParallelEmbedding(vocab_size, model_args.dim)
    embedding.load_state_dict(load_layer_state_dict(embedding_file))

    inputs = embedding(batch)
    check_for_nans(inputs, "embedding outputs")
    logging.debug(f"Embedded inputs: {inputs}")

    save_to_disk(inputs, inputs_file)

def forward_task(layer_idx, inputs_file, state_dict_file, freqs_cis_file, logits_file, mask_file):
    inputs = load_from_disk(inputs_file)
    if torch.isnan(inputs).any() or torch.isinf(inputs).any():
        raise ValueError(f"NaNs or Infs detected in inputs for layer {layer_idx}")

    state_dict = load_layer_state_dict(state_dict_file)
    for param in state_dict.values():
        if torch.isnan(param).any() or torch.isinf(param).any():
            raise ValueError(f"NaNs or Infs detected in weights of layer {layer_idx}")

    freqs_cis = load_from_disk(freqs_cis_file)
    mask = load_from_disk(mask_file)

    layer = TransformerBlock(layer_idx, model_args)
    layer.load_state_dict(state_dict)

    start_pos = 0  # Adjust as necessary
    seqlen = inputs.shape[1]
    freqs_cis = freqs_cis[start_pos: start_pos + seqlen]

    outputs = layer(inputs, start_pos, freqs_cis, mask)
    check_for_nans(outputs, f"layer {layer_idx} outputs")
    logging.debug(f"Outputs after layer {layer_idx}: {outputs}")

    save_to_disk(outputs, logits_file)

def backward_task(layer_idx, error_file, inputs_file, state_dict_file, error_output_file, mask_file):
    error = load_from_disk(error_file)
    state_dict = load_layer_state_dict(state_dict_file)
    
    layer = TransformerBlock(layer_idx, model_args)
    layer.load_state_dict(state_dict)
    
    inputs = load_from_disk(inputs_file)
    inputs.requires_grad = True
    
    error = error.view(inputs.shape)  # Ensure the gradient tensor matches the output tensor shape
    inputs.backward(error)
    
    grads = [param.grad for param in layer.parameters()]
    logging.debug(f"Gradients for layer {layer_idx}: {grads}")
    save_to_disk((inputs.grad, grads), error_output_file)

def final_logits_task(inputs_file, state_dict_file, logits_file):
    inputs = load_from_disk(inputs_file)
    state_dict = load_layer_state_dict(state_dict_file)

    if state_dict is None:
        raise ValueError("Failed to load state dict for the output layer")

    output_layer = ColumnParallelLinear(model_args.dim, state_dict['weight'].shape[0], bias=False)
    output_layer.load_state_dict(state_dict)

    logits = output_layer(inputs)
    check_for_nans(logits, "final logits")
    logging.debug(f"Final logits: {logits}")

    save_to_disk(logits, logits_file)

def final_logits_backward_task(error_file, inputs_file, state_dict_file, error_output_file):
    error = load_from_disk(error_file)
    state_dict = load_layer_state_dict(state_dict_file)
    
    if state_dict is None:
        raise ValueError("Failed to load state dict for the output layer")

    output_layer = ColumnParallelLinear(model_args.dim, state_dict['weight'].shape[0], bias=False)
    output_layer.load_state_dict(state_dict)
    
    inputs = load_from_disk(inputs_file)
    inputs.requires_grad = True
    logits = output_layer(inputs)
    
    logits.backward(error)
    
    grads = [param.grad for param in output_layer.parameters()]
    logging.debug(f"Gradients for final logits: {grads}")
    save_to_disk((inputs.grad, grads), error_output_file)

def embed_backward_task(error_file, batch_file, embedding_file, error_output_file):
    error = load_from_disk(error_file)
    logging.debug(f"Error tensor shape: {error.shape}")

    vocab_size = tokenizer.get_vocab_size()
    
    embedding = VocabParallelEmbedding(vocab_size, model_args.dim)
    embedding.load_state_dict(load_layer_state_dict(embedding_file))
    
    batch = load_from_disk(batch_file)
    logging.debug(f"Batch tensor shape: {batch.shape}")

    embeddings = embedding(batch)
    embeddings.retain_grad()
    logging.debug(f"Embeddings tensor shape: {embeddings.shape}")

    error = error.view(embeddings.shape)  # Ensure the gradient tensor matches the output tensor shape
    logging.debug(f"Reshaped error tensor shape: {error.shape}")
    
    embeddings.backward(error)
    
    grads = [param.grad for param in embedding.parameters()]
    logging.debug(f"Gradients for embedding: {grads}")
    save_to_disk(grads, error_output_file)

def loss_task(logits_file, targets_file, loss_file, logits_grad_file):
    logits = load_from_disk(logits_file)
    targets = load_from_disk(targets_file)

    logging.debug(f"Logits for loss: {logits}")
    logging.debug(f"Targets for loss: {targets}")

    pad_id = tokenizer.pad_id

    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=pad_id)
    save_to_disk(loss.item(), loss_file)

    logits.retain_grad()
    loss.backward()
    check_for_nans(logits.grad, "logits gradients")
    logging.debug(f"Logits gradients for loss: {logits.grad}")

    save_to_disk(logits.grad, logits_grad_file)

def apply_adamw(layer_idx, grads, learning_rate, beta1, beta2, epsilon, weight_decay, t):
    max_grad_norm = 1.0

    if layer_idx == -1:
        state_dict_file = "data/output.pt"
        m_file = "data/adam_m_output.pt"
        v_file = "data/adam_v_output.pt"
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
            if param.grad is not None:
                param.grad.zero_()

            param.data -= learning_rate * weight_decay * param.data

            m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
            v[i] = beta2 * v[i] + (1 - beta2) * (grads[i] ** 2)

            m_hat = m[i] / (1 - beta1 ** t)
            v_hat = v[i] / (1 - beta2 ** t)

            param.data -= learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)

            torch.nn.utils.clip_grad_norm_(param, max_grad_norm)

    logging.debug(f"Updated state dict for layer {layer_idx}: {state_dict}")

    save_layer_state_dict(state_dict, state_dict_file)
    save_to_disk(m, m_file)
    save_to_disk(v, v_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=[
        "embed", "forward", "backward", "final_logits", "final_logits_backward", 
        "embed_backward", "loss", "apply_adamw"
    ])
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
    parser.add_argument("--freqs_cis", type=str, required=False)
    parser.add_argument("--mask", type=str, required=False)
    args = parser.parse_args()

    initialize_distributed_environment()
    initialize_model_parallel(model_parallel_size_=1)

    if args.task == "embed":
        embed_task(args.batch, args.embedding_file, args.inputs)
    elif args.task == "forward":
        forward_task(args.layer_idx, args.inputs, args.state_dict, args.freqs_cis, args.logits_file, args.mask)
    elif args.task == "backward":
        backward_task(args.layer_idx, args.error, args.inputs, args.state_dict, args.error_output_file, args.mask)
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

    dist.destroy_process_group()
