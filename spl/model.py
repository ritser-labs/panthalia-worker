import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, model_args):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.transformer = nn.Transformer(
            d_model=model_args.dim, 
            nhead=model_args.n_heads, 
            num_encoder_layers=model_args.n_layers,
            num_decoder_layers=model_args.n_layers,
            dim_feedforward=model_args.dim * model_args.ffn_dim_multiplier,
            dropout=0.1
        )
        self.fc = nn.Linear(model_args.dim, model_args.vocab_size)

    def forward(self, x, start_pos=0):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, d_model)
        x = self.transformer(x, x)
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
