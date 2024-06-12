import torch
import torch.nn as nn
import torch.optim as optim
from util import safe_serialize, safe_deserialize, download_file, upload_file

class InputLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(InputLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        activations = self.linear(x)
        # Securely serialize and upload activations
        serialized_data = safe_serialize(activations)
        url = upload_file(serialized_data)
        return url

def run_input_layer():
    layer1 = InputLayer(10, 20)
    opt1 = optim.SGD(layer1.parameters(), lr=0.01)

    # Simulated download of input training data
    training_data_url = "url/to/training_data"
    training_data = safe_deserialize(download_file(training_data_url))

    for epoch in range(10):
        opt1.zero_grad()
        url = layer1(training_data)
        # Assuming the existence of a method to receive gradient URL
        grad_url = "url/to/received/grads"  # Placeholder
        gradients = safe_deserialize(download_file(grad_url))
        for param, grad in zip(layer1.parameters(), gradients):
            param.grad = grad.clone()
        opt1.step()

# Simulate running
run_input_layer()
