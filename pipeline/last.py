import torch
import torch.nn as nn
import torch.optim as optim
from util import safe_serialize, safe_deserialize, download_file, upload_file

class OutputLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(OutputLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, activations):
        output = self.linear(activations)
        return output

def run_output_layer(activation_url):
    layer2 = OutputLayer(20, 5)
    opt2 = optim.SGD(layer2.parameters(), lr=0.01)

    activations = safe_deserialize(download_file(activation_url))
    output = layer2(activations)
    target_url = "url/to/output_data"  # Placeholder for actual URL
    target = safe_deserialize(download_file(target_url))

    criterion = nn.MSELoss()
    loss = criterion(output, target)
    loss.backward()
    # Serialize and upload gradients
    gradients = safe_serialize([param.grad for param in layer2.parameters()])
    grad_url = upload_file(gradients)
    return grad_url

# Simulate running
activation_url = "url/to/activations"
run_output_layer(activation_url)
