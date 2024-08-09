import subprocess
import requests
import os
import time
import torch
from model import ModelArgs, Transformer
from common import tokenizer, wait_for_sot, model_adapter
from device import device
from io import BytesIO

# The URL where the SOT service is running
SOT_URL = 'http://localhost:5001'


# Load the model from the SOT service
def load_model():
    flattened_params = torch.load(os.path.join('data', 'state', 'model.pt'), map_location=device)
    model = model_adapter.tensor_to_model(flattened_params)
    model.train()
    return model

# Initialize the model
model = None

# Function to make a prediction
def make_prediction(text):
    tokens = tokenizer.encode(
        text,
        bos=False,
        eos=False,
        allowed_special=set(),
        disallowed_special=()
    )
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    with torch.inference_mode():
        outputs = model(input_tensor, start_pos=0)
    predictions = torch.argmax(outputs, dim=-1).cpu().numpy().tolist()[0]
    decoded_output = tokenizer.decode(predictions)
    return decoded_output

def main():
    global model
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    model_adapter.initialize_environment('nccl')

    # Load the model
    model = load_model()
    if model:
        print("Model loaded successfully.")
    else:
        print("Failed to load the model. Exiting...")
        return

    # Main prediction loop
    try:
        print("Enter text to generate predictions (type 'exit' to quit):")
        while True:
            user_input = input("Input: ")
            if user_input.lower() == 'exit':
                break
            prediction = make_prediction(user_input)
            print(f"Prediction: {prediction}")
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    main()
