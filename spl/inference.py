import os
import torch
from .common import tokenizer, model_adapter
from .device import device

# The URL where the SOT service is running
SOT_URL = 'http://localhost:5001'

script_dir = os.path.dirname(os.path.abspath(__file__))

print(f"Vocab size: {tokenizer.get_vocab_size()}")

# Load the model from the SOT service
def load_model():
    flattened_params = torch.load(os.path.join(script_dir, 'data', 'state', 'model.pt'), map_location=device)
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
        outputs = model_adapter.forward(model, input_tensor)
    print(f'Shape of outputs: {outputs.shape}')
    top_token = model_adapter.get_top_token(outputs)
    print(f'Top token: {top_token}')
    decoded_output = tokenizer.decode(top_token)
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
