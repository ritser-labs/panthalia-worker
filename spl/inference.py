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
    model_path = os.path.join(script_dir, 'data', 'state', 'model.safetensors')
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
        for i in range(10):
            user_input = 'This product is'
            tokenized_user_input = tokenizer.encode(user_input)
            tokenized_user_input_tensor = torch.tensor(tokenized_user_input, dtype=torch.long).unsqueeze(0).to(device)
            tokenized_output = model_adapter.generate(model, tokenized_user_input_tensor, max_new_tokens=100, top_k=200, temperature=0.8)
            output = tokenizer.decode(tokenized_output[0].tolist())
            print(output)
            print('\n' * 3)
            print('-' * 80)
            print('\n' * 3)
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    main()
