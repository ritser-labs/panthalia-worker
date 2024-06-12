from util import safe_serialize, safe_deserialize, download_file, upload_file
import torch
import torch.nn as nn
import torch.optim as optim
from constants import GRADIENTS_URL, MODEL_1_STATE_URL
from model import InputLayer

def update_weights_input_layer(model_url, grad_url):
  optimizer = optim.SGD(input_layer_model.parameters(), lr=0.01)
  # Download and load model state
  model_state = safe_deserialize(download_file(model_url))
  input_layer_model.load_state_dict(model_state)
  gradients = safe_deserialize(download_file(grad_url))
  with torch.no_grad():
    for param, grad in zip(input_layer_model.parameters(), gradients):
      param.grad = grad.clone()
  optimizer.step()
  # Optionally serialize and upload updated model weights
  new_weights_url = upload_file(safe_serialize(input_layer_model.state_dict()))
  return new_weights_url


input_layer_model = InputLayer(10, 20)
new_weights_url = update_weights_input_layer(MODEL_1_STATE_URL, GRADIENTS_URL)
