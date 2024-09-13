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
    model_path = os.path.join(script_dir, 'data', 'state', 'model.pt')
    weights_tensor = None
    if not os.path.exists(model_path):
        print('Model not found, initializing new model')
        weights_tensor = model_adapter.init_tensor()
    else:
        print(f'Model loaded from {model_path}')
        weights_tensor = torch.load(model_path, map_location=device)
    model = model_adapter.tensor_to_model(weights_tensor)
    model.train()
    return model

# Initialize the model
model = None

# Main function to initialize and run the model
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
            tokenized_user_input = tokenizer.encode(user_input)
            tokenized_user_input_tensor = torch.tensor(tokenized_user_input, dtype=torch.long).unsqueeze(0).to(device)
            tokenized_output = model_adapter.generate(model, tokenized_user_input_tensor)
            output = tokenizer.decode(tokenized_output[0])
            print(f"Output: {output}")
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    main()
