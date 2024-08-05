import subprocess
import requests
import os
import time
import torch
from model import ModelArgs, Transformer
from common import tokenizer, wait_for_sot, tensor_to_model, initialize_distributed_environment_and_globals
from device import device
from io import BytesIO

# The URL where the SOT service is running
SOT_URL = 'http://localhost:5001'

# Start the SOT service
def start_sot():
    sot_process = subprocess.Popen(
        ['python', 'sot.py', '--public_keys_file', 'master_public_keys.json'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return sot_process

# Load the model from the SOT service
def load_model():
    response = requests.get(f'{SOT_URL}/latest_state', params={'tensor_name': 'model'})
    if response.status_code == 200:
        tensor_data = BytesIO(response.content)
        model_tensor = torch.load(tensor_data, map_location=device)
        model = tensor_to_model(model_tensor)
        model.eval()  # Set model to evaluation mode
        return model
    else:
        print(f"Failed to load model: {response.text}")
        return None

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
    
    initialize_distributed_environment_and_globals('nccl')

    # Start the SOT service
    sot_process = start_sot()

    wait_for_sot(SOT_URL)

    # Load the model
    model = load_model()
    if model:
        print("Model loaded successfully.")
    else:
        print("Failed to load the model. Exiting...")
        sot_process.terminate()
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
    finally:
        sot_process.terminate()

if __name__ == "__main__":
    main()
