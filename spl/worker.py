import argparse
import torch
from common import load_from_disk, save_to_disk, load_layer_state_dict, save_layer_state_dict, model_args  # Import global model_args
from model import TransformerLayer, Transformer
from tokenizer import Tokenizer


def run_layer_step(layer, x, is_forward=True, next_error=None, optimizer=None, scaler=None, loss_fn=None, pad_id=None, logits=None):
    if is_forward:
        with torch.cuda.amp.autocast():
            x = layer(x)
        return x
    else:
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            if logits is None:
                if x is None:
                    x = next_error  # Use the next_error as the input for backward pass
                print(f"Shape of x before view: {x.shape}")  # Debug: Check shape of x before view
                x = x.view(-1, layer.norm1.normalized_shape[0])
                print(f"Shape of x after view: {x.shape}")  # Debug: Check shape of x after view
                logits = layer(x)
                print(f"Shape of logits: {logits.shape}")  # Debug: Check shape of logits
            logits = logits.view(-1, logits.size(-1))  # Flatten logits for loss calculation
            print(f"Shape of reshaped logits: {logits.shape}")  # Debug: Check shape of reshaped logits
            
            # Convert next_error to class indices if needed and flatten it
            if next_error.dim() == 3 and next_error.size(-1) == model_args.vocab_size:
                next_error = torch.argmax(next_error, dim=-1)
            next_error = next_error.view(-1)
            print(f"Shape of reshaped next_error: {next_error.shape}")  # Debug: Check shape of reshaped next_error

            # Check for out-of-range values
            print(f"Min value in next_error: {next_error.min().item()}")
            print(f"Max value in next_error: {next_error.max().item()}")

            loss = loss_fn(logits, next_error)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        grads = [param.grad for param in layer.parameters()]
        return loss.item(), grads



def embed_task(batch_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = load_from_disk(batch_file)
    if batch is None:
        raise ValueError(f"Failed to load batch from {batch_file}")
    batch = batch.to(device)
    tokenizer = Tokenizer(encoding_name='cl100k_base')
    model = Transformer(model_args).to(device)
    inputs = model.embedding(batch).permute(1, 0, 2)
    save_to_disk(inputs.cpu(), "data/inputs.pt")

def forward_task(layer_idx, inputs_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    layer = TransformerLayer(dim=model_args.dim, n_heads=model_args.n_heads, ffn_dim=model_args.dim * model_args.ffn_dim_multiplier).to(device)
    state_dict = load_layer_state_dict(f"data/layer_{layer_idx}.pt")
    if state_dict is None:
        layer = TransformerLayer(model_args.dim, model_args.n_heads, model_args.dim * model_args.ffn_dim_multiplier).to(device)
        save_layer_state_dict(layer.state_dict(), f"data/layer_{layer_idx}.pt")
        state_dict = load_layer_state_dict(f"data/layer_{layer_idx}.pt")
        if state_dict is None:
            raise ValueError(f"Failed to initialize and load state dict for layer {layer_idx}")
    layer.load_state_dict(state_dict)
    inputs = load_from_disk(inputs_file)
    if inputs is None:
        raise ValueError(f"Failed to load inputs from {inputs_file}")
    inputs = inputs.to(device)
    with torch.cuda.amp.autocast():
        activations = layer(inputs)
    save_to_disk(activations.cpu(), f"data/logits_layer_{layer_idx}.pt")


def final_logits_task(inputs_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = load_from_disk(inputs_file)
    if inputs is None:
        raise ValueError(f"Failed to load inputs from {inputs_file}")
    inputs = inputs.to(device)
    model = Transformer(model_args).to(device)
    logits = model.fc(inputs.permute(1, 0, 2))
    save_to_disk(logits.cpu(), "data/logits.pt")

def backward_task(layer_idx, error_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    layer = TransformerLayer(dim=model_args.dim, n_heads=model_args.n_heads, ffn_dim=model_args.dim * model_args.ffn_dim_multiplier).to(device)
    state_dict = load_layer_state_dict(f"data/layer_{layer_idx}.pt")
    if state_dict is None:
        raise ValueError(f"Failed to load layer state dict for layer {layer_idx}")
    layer.load_state_dict(state_dict)
    error = load_from_disk(error_file)
    if error is None:
        raise ValueError(f"Failed to load error from {error_file}")
    error = error.to(device)
    print(f"Shape of error: {error.shape}")  # Debug: Check shape of error
    if len(error.shape) == 2:
        error = error.unsqueeze(0)
        print(f"Shape of error after unsqueeze: {error.shape}")  # Debug: Check shape of error after unsqueeze

    # Load precomputed logits
    logits = load_from_disk(f"data/logits_layer_{layer_idx}.pt")
    if logits is None:
        raise ValueError(f"Failed to load logits from data/logits_layer_{layer_idx}.pt")
    logits = logits.to(device)

    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    tokenizer = Tokenizer(encoding_name='cl100k_base')
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    print("Running backward step")
    
    # Ensure next_error is correctly shaped
    if error.dim() == 3 and error.size(-1) == model_args.vocab_size:
        error = torch.argmax(error, dim=-1)
    
    loss, grads = run_layer_step(layer, None, is_forward=False, next_error=error, optimizer=optimizer, scaler=scaler, loss_fn=loss_fn, pad_id=tokenizer.pad_id, logits=logits)
    print(f"Saving error and grads for layer {layer_idx}")
    error_grads = (loss, grads)
    print(f"Saving error_grads with shape {len(error_grads)}")
    save_to_disk(error_grads, f"data/error_layer_{layer_idx}.pt")
    save_layer_state_dict(layer.state_dict(), f"data/layer_{layer_idx}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["embed", "forward", "backward", "final_logits"])
    parser.add_argument("--layer_idx", type=int, required=False)
    parser.add_argument("--inputs", type=str, required=False)
    parser.add_argument("--error", type=str, required=False)
    parser.add_argument("--batch", type=str, required=False)
    args = parser.parse_args()

    if args.task == "embed":
        embed_task(args.batch)
    elif args.task == "forward":
        forward_task(args.layer_idx, args.inputs)
    elif args.task == "backward":
        backward_task(args.layer_idx, args.error)
    elif args.task == "final_logits":
        final_logits_task(args.inputs)
