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

# Function to make a prediction and get the next token
def predict_next_token(tokens):
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    with torch.inference_mode():
        outputs = model_adapter.forward(model, input_tensor)
    # The last token in the output corresponds to the prediction for the next token in the sequence
    next_token = model_adapter.get_top_token(outputs[:, -1, :])
    return next_token

# Function to run continuous inference and keep generating tokens
def generate_tokens(text, max_length=50):
    tokens = tokenizer.encode(
        text,
        bos=False,
        eos=False,
        allowed_special=set(),
        disallowed_special=()
    )

    print(f"Initial tokens: {tokens}")

    # Keep generating until the desired length is reached or exit is triggered
    while True:
        # Predict the next token
        next_token = predict_next_token(tokens)
        print(f"Next token: {next_token}")

        # Add the next token to the sequence
        tokens.append(next_token)

        # If the sequence gets too long, truncate the first token
        if len(tokens) > max_length:
            tokens = tokens[1:]

        # Decode the current sequence to text and print
        decoded_text = tokenizer.decode(tokens)
        print(f"Generated text: {decoded_text}")

        # Optional: You can add a stopping condition here, e.g., based on a special token or length
        # To stop the loop manually, just use a KeyboardInterrupt (Ctrl+C)

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
            generate_tokens(user_input)
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    main()
