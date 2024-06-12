from util import safe_serialize, safe_deserialize, download_file, upload_file
import torch
import torch.nn as nn
import torch.optim as optim
from constants import ACTIVATIONS_URL, NEW_WEIGHTS_2_URL, OUTPUT_DATA_URL, GRADIENTS_URL
from model import OutputLayer

def forward_and_backward_output_layer(model, activation_url, target_url, weights_url, grad_url):
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    
    # Try to load existing weights if available
    try:
        model_state = safe_deserialize(download_file(weights_url))
        model.load_state_dict(model_state)
        print("Loaded existing model state.")
    except Exception as e:
        print(f"No existing model state or unable to load, generating new weights.")
    
    activations = safe_deserialize(download_file(activation_url)).requires_grad_(True)
    target = safe_deserialize(download_file(target_url))
    output = model(activations)
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    print(loss)
    optimizer.zero_grad()
    loss.backward()

    # Capture the gradients with respect to the input of this layer
    input_grad = activations.grad.data
    
    optimizer.step()
    grad_url = upload_file(safe_serialize(input_grad), grad_url)
    # Serialize and upload the model state after updates
    weights_url = upload_file(safe_serialize(model.state_dict()), weights_url)
    
    return grad_url, weights_url

# Example instantiation and use
output_layer_model = OutputLayer(20, 10)
# Assume URLs to activations and target data are given
grad_url, updated_weights_url = forward_and_backward_output_layer(output_layer_model, ACTIVATIONS_URL, OUTPUT_DATA_URL, NEW_WEIGHTS_2_URL, GRADIENTS_URL)
