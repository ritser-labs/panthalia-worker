import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm  # For progress bar

# Import the relevant modules from your existing files
from ..common import model_adapter, model_config, dataset
from ..device import device
from ..inference import load_model
from ..sot import stable_adamw_update
import math
import logging

logging.basicConfig(level=logging.INFO)

# Directory to save model checkpoints
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_path = os.path.join(parent_dir, 'data', 'state', 'model.pt')

# Training configurations
BUFFER_SIZE = 1000
MAX_SEQ_LEN = 128
BATCH_SIZE = 16
LEARNING_RATE = 1e-2
EPOCHS = 5
ACCUMULATION_STEPS = 1

# Function to train the model
def train_model(model, train_loader, model_adapter, epochs):
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_loss = 0.0

        weights = model_adapter.model_to_tensor(model)
        
        m = torch.zeros_like(weights)
        v = torch.zeros_like(weights)

        # Use tqdm for a progress bar
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            # Convert to tensors and add batch dimension
            inputs = torch.tensor(inputs).unsqueeze(0).to(device)
            targets = torch.tensor(targets).unsqueeze(0).to(device)

            updates, loss = model_adapter.train_task(model, inputs, targets, ACCUMULATION_STEPS)
            

            
            weights, m, v = stable_adamw_update(weights, updates, m, v, lr=LEARNING_RATE, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, step=batch_idx+1)
            

            model = model_adapter.tensor_to_model(weights)

            #if math.isnan(loss.item()):
            if batch_idx % 100 == 0:
                #print(f'Input: {inputs}')
                #print(f'Target: {targets}')
                print(f'Loss: {loss}')
                #print(f'Updates: {updates}')
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

    # Train the model
    train_model(model, dataset, model_adapter, EPOCHS)

if __name__ == "__main__":
    main()
