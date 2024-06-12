from util import safe_serialize, safe_deserialize, download_file, upload_file
import torch
import torch.nn as nn
import torch.optim as optim
from constants import GRADIENTS_URL, MODEL_1_STATE_URL, ACTIVATIONS_URL
from model import InputLayer

# Stage 3
def update_weights_input_layer(model_url, grad_url, activations_url):
    input_layer_model = InputLayer(10, 20)
    optimizer = optim.SGD(input_layer_model.parameters(), lr=0.01)
    
    # Load model state
    model_state = safe_deserialize(download_file(model_url))
    input_layer_model.load_state_dict(model_state)
    
    # Load input gradients
    input_grads = safe_deserialize(download_file(grad_url))
    
    # Load previously computed activations from Stage 1
    activations = safe_deserialize(download_file(activations_url))
    activations.requires_grad_(True)  # Ensure they're set to require grad

    # Use the activations directly for the backward pass
    # We don't compute any new forward pass here; we use the stored activations
    optimizer.zero_grad()
    activations.backward(input_grads)  # Apply the gradients received from Stage 2 directly

    optimizer.step()
    
    new_weights_url = upload_file(safe_serialize(input_layer_model.state_dict()), model_url)
    return new_weights_url

new_weights_url = update_weights_input_layer(MODEL_1_STATE_URL, GRADIENTS_URL, ACTIVATIONS_URL)
