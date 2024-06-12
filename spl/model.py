import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLayer(nn.Module):
    def __init__(self, dim, n_heads, ffn_dim):
        super(TransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=0.1)
        self.linear1 = nn.Linear(dim, ffn_dim)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(ffn_dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x2 = self.norm1(x)
        x2, _ = self.self_attn(x2, x2, x2)
        x = x + x2
        x2 = self.norm2(x)
        x2 = self.linear2(self.dropout(F.relu(self.linear1(x2))))
        x = x + x2
        return x

class Transformer(nn.Module):
    def __init__(self, model_args):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.layers = nn.ModuleList([
            TransformerLayer(model_args.dim, model_args.n_heads, model_args.dim * model_args.ffn_dim_multiplier)
            for _ in range(model_args.n_layers)
        ])
        self.fc = nn.Linear(model_args.dim, model_args.vocab_size)

    def forward(self, x, start_pos=0):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, d_model)
        for layer in self.layers:
            x = layer(x)
        x = x.permute(1, 0, 2)  # Back to (batch, seq_len, d_model)
        x = self.fc(x)
        return x

class ModelArgs:
    def __init__(self, vocab_size, dim, n_layers, n_heads, ffn_dim_multiplier):
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ffn_dim_multiplier = ffn_dim_multiplier
