import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm  # For progress bar

# Import the relevant modules from your existing files
from ..common import model_adapter, model_config, dataset
from ..device import device
from ..inference import load_model
import math

# Directory to save model checkpoints
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_path = os.path.join(parent_dir, 'data', 'state', 'model.pt')

# Training configurations
BUFFER_SIZE = 1000
MAX_SEQ_LEN = 128
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 5

# Function to train the model
def train_model(model, train_loader, optimizer, model_adapter, epochs):
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_loss = 0.0

        # Use tqdm for a progress bar
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            # Convert to tensors and add batch dimension
            inputs = torch.tensor(inputs, dtype=torch.long).unsqueeze(0).to(device)
            targets = torch.tensor(targets, dtype=torch.long).unsqueeze(0).to(device)

            # Forward pass
            optimizer.zero_grad()
            loss = model_adapter.forward_and_loss(model, inputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if math.isnan(loss.item()):
                print(f'Input: {inputs}')
                print(f'Target: {targets}')
        avg_loss = total_loss / (batch_idx + 1)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        torch.save(model_adapter.model_to_tensor(model), save_path)
        print(f"Model checkpoint saved to {save_path}")

def main():
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize environment (if needed for distributed training)
    model_adapter.initialize_environment('nccl')

    # Load model
    model = load_model()
    if model:
        print("Model loaded successfully.")
    else:
        print("Failed to load the model. Exiting...")
        return

    # Define the optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    train_model(model, dataset, optimizer, model_adapter, EPOCHS)

if __name__ == "__main__":
    main()
