import torch
import torch.nn as nn
import torch.optim as optim
from util import safe_serialize, safe_deserialize, download_file, upload_file
from constants import INPUT_DATA_URL, MODEL_1_STATE_URL, ACTIVATIONS_URL
from model import InputLayer

def forward_pass_input_layer(model, data, activations_url, model_url):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    try:
        # Attempt to load a previously saved model state
        model_state = safe_deserialize(download_file(model_url))
        model.load_state_dict(model_state)
        print("Loaded existing model state.")
    except Exception as e:
        print(f"No existing model state or unable to load: {e}")
    
    optimizer.zero_grad()
    activations = model(data)
    serialized_activations = safe_serialize(activations)
    url_activations = upload_file(serialized_activations, activations_url)
    
    # Serialize and re-upload the current model state, may be useful if modifications were made
    model_state = safe_serialize(model.state_dict())
    model_url = upload_file(model_state, model_url)
    
    return url_activations, model_url

# Example of usage
input_layer_model = InputLayer(10, 20)
input_data = safe_deserialize(download_file(INPUT_DATA_URL))
activations_url, model_url = forward_pass_input_layer(input_layer_model, input_data, ACTIVATIONS_URL, MODEL_1_STATE_URL)
