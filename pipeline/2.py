from util import safe_serialize, safe_deserialize, download_file, upload_file
import torch
import torch.nn as nn
import torch.optim as optim
from constants import ACTIVATIONS_URL, NEW_WEIGHTS_2_URL, OUTPUT_DATA_URL
from model import OutputLayer

def forward_and_backward_output_layer(model, activation_url, target_url, weights_url):
  optimizer = optim.SGD(model.parameters(), lr=0.01)
  activations = safe_deserialize(download_file(activation_url))
  target = safe_deserialize(download_file(target_url))
  output = model(activations)
  criterion = nn.MSELoss()
  loss = criterion(output, target)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  # Optionally serialize and upload new model weights
  weights_url = upload_file(safe_serialize(model.state_dict()), weights_url)
  gradients = safe_serialize([param.grad for param in model.parameters()])
  grad_url = upload_file(gradients)
  return grad_url, weights_url

output_layer_model = OutputLayer(20, 5)
# Assume URLs to activations and target data are given
grad_url, updated_weights_url = forward_and_backward_output_layer(output_layer_model, ACTIVATIONS_URL, OUTPUT_DATA_URL, NEW_WEIGHTS_2_URL)
