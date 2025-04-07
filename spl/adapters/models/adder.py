import torch
import torch.nn as nn

class Adder(nn.Module):
    def __init__(self):
        super(Adder, self).__init__()
        # A single linear layer with 2 input features (the two numbers to add) and 1 output feature (their sum)
        self.linear = nn.Linear(2, 1, bias=False)

    def forward(self, x):
        # Forward pass through the linear layer
        return self.linear(x)
