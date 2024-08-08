import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import json
import time
import os
import requests
from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3
import subprocess
from model import ModelArgs, Transformer
from common import Model, wait_for_sot, tensor_to_model, initialize_distributed_environment_and_globals, model_args, tokenizer
from device import device

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s', handlers=[
    logging.StreamHandler()
])

# Constants
SOT_URL = 'http://localhost:5001'
MASTER_WALLETS_FILE = 'master_wallets.json'

# Simple linear model
class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # Single input and single output

    def forward(self, x):
        return self.linear(x)

# Stable AdamW update function
def stable_adamw_update(params, grads, m, v, lr=0.01, weight_decay=0.01, beta1=0.9, beta2=0.999, eps=1e-8, step=1):
    grads = torch.clamp(grads, -1.0, 1.0)  # Clip gradients

    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * grads * grads

    m_hat = m / (1 - beta1 ** step)
    v_hat = v / (1 - beta2 ** step)

    param_update = m_hat / (torch.sqrt(v_hat) + eps)
    param_update += weight_decay * params

    params = params - lr * param_update

    return params, m, v

# Base class for model adapters
class ModelAdapter:
    def __init__(self, model):
        self.model = model

    def random_init_all_params(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                nn.init.normal_(param, mean=0.0, std=0.02)

    def generate_synthetic_data(self, num_samples=100):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def compute_loss(self, outputs, targets):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def train_step(self, data, targets, lr, weight_decay, epochs, accumulation_steps):
        raise NotImplementedError("This method should be implemented by subclasses.")

# Simple Linear Network Adapter
class SimpleLinearAdapter(ModelAdapter):
    def __init__(self):
        super().__init__(SimpleLinearModel())

    def generate_synthetic_data(self, num_samples=100):
        x = torch.randn(num_samples, 1)  # Random input data
        y = 2 * x + 3  # Linear function y = 2x + 3
        return x, y

    def compute_loss(self, outputs, targets):
        return ((outputs - targets) ** 2).mean()  # Mean Squared Error loss

    def train_step(self, data, targets, lr, weight_decay, epochs, accumulation_steps):
        self.model.train()
        params = torch.cat([param.view(-1) for param in self.model.parameters() if param.requires_grad])
        m = torch.zeros_like(params)
        v = torch.zeros_like(params)
        
        microbatch_size = data.size(0) // accumulation_steps
        
        for epoch in range(epochs):
            grads_accumulated = torch.zeros_like(params)
            total_loss = 0.0
            
            for step in range(accumulation_steps):
                self.model.zero_grad()  # Zero the gradients
                
                start_idx = step * microbatch_size
                end_idx = start_idx + microbatch_size
                
                outputs = self.model(data[start_idx:end_idx])  # Forward pass
                loss = self.compute_loss(outputs, targets[start_idx:end_idx])
                loss.backward()  # Backward pass to compute gradients
                
                grads = torch.cat([param.grad.view(-1) for param in self.model.parameters() if param.grad is not None])
                grads_accumulated += grads
                total_loss += loss.item()
                
            grads_accumulated /= accumulation_steps
            
            params, m, v = stable_adamw_update(params, grads_accumulated, m, v, lr, weight_decay, step=epoch+1)
            
            # Update model parameters
            idx = 0
            for param in self.model.parameters():
                if param.requires_grad:
                    param_size = param.numel()
                    param.data = params[idx:idx + param_size].view_as(param).data
                    idx += param_size

            if epoch % 10 == 0:
                logging.info(f"Epoch [{epoch}/{epochs}], Loss: {total_loss/accumulation_steps:.4f}")

# Transformer LLM Adapter
class TransformerLLMAdapter(ModelAdapter):
    def __init__(self):
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        initialize_distributed_environment_and_globals('nccl')

        model = Model(model_args).to(device)
        super().__init__(model)
        self.model.to(device)

    def generate_synthetic_data(self, num_samples=100):
        batch_size = model_args.max_batch_size  # Ensure it matches model_args.max_batch_size
        data = torch.randint(0, model_args.vocab_size, (batch_size, 50), dtype=torch.long).to(device)  # Random input data
        targets = data.clone()  # For simplicity, the targets can be the same as inputs in a dummy example
        return data, targets

    def compute_loss(self, outputs, targets):
        return F.cross_entropy(outputs.view(-1, model_args.vocab_size), targets.view(-1), ignore_index=tokenizer.pad_id)

    def train_step(self, data, targets, lr, weight_decay, epochs, accumulation_steps):
        self.model.train()
        params = torch.cat([param.view(-1) for param in self.model.parameters() if param.requires_grad])
        m = torch.zeros_like(params)
        v = torch.zeros_like(params)
        
        microbatch_size = data.size(0) // accumulation_steps
        
        for epoch in range(epochs):
            grads_accumulated = torch.zeros_like(params)
            total_loss = 0.0
            
            for step in range(accumulation_steps):
                self.model.zero_grad()  # Zero the gradients
                
                start_idx = step * microbatch_size
                end_idx = start_idx + microbatch_size
                
                outputs = self.model(data[start_idx:end_idx], start_pos=0)  # Forward pass
                loss = self.compute_loss(outputs, targets[start_idx:end_idx])
                loss.backward()  # Backward pass to compute gradients
                
                grads = torch.cat([param.grad.view(-1) for param in self.model.parameters() if param.grad is not None])
                grads_accumulated += grads
                total_loss += loss.item()
                
                # Detach cache tensors
                if hasattr(self.model, 'layers'):
                    for layer in self.model.layers:
                        if hasattr(layer, 'attention'):
                            layer.attention.cache_k = layer.attention.cache_k.detach()
                            layer.attention.cache_v = layer.attention.cache_v.detach()
                
            grads_accumulated /= accumulation_steps
            
            params, m, v = stable_adamw_update(params, grads_accumulated, m, v, lr, weight_decay, step=epoch+1)
            
            # Update model parameters
            idx = 0
            for param in self.model.parameters():
                if param.requires_grad:
                    param_size = param.numel()
                    param.data = params[idx:idx + param_size].view_as(param).data
                    idx += param_size

            if epoch % 10 == 0:
                logging.info(f"Epoch [{epoch}/{epochs}], Loss: {total_loss/accumulation_steps:.4f}")

# Function to train the selected model
def train_model(adapter_class, lr=0.01, weight_decay=0.01, epochs=100, accumulation_steps=4):
    adapter = adapter_class()
    adapter.random_init_all_params()

    data, targets = adapter.generate_synthetic_data()
    adapter.train_step(data, targets, lr, weight_decay, epochs, accumulation_steps)

def main():
    model_type = "transformer_llm"  # Change this to "transformer_llm" to switch models
    if model_type == "simple_linear":
        adapter_class = SimpleLinearAdapter
    elif model_type == "transformer_llm":
        adapter_class = TransformerLLMAdapter
    else:
        raise ValueError("Invalid model type specified.")

    train_model(adapter_class, lr=0.01, weight_decay=0.01, epochs=100, accumulation_steps=4)

if __name__ == "__main__":
    main()
