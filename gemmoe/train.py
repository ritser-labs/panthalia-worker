import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
import numpy as np
from sklearn.metrics import accuracy_score
from modeling_gemmoe import GemmoeForCausalLM
from tokenization_gemmoe import GemmoeTokenizer
from configuration_gemmoe import GemmoeConfig
from torch import nn

NUM_PROC = 24

# Load the dataset
dataset = load_dataset("wikipedia", language="en", date="20240401", split='train[:5%]', trust_remote_code=True, num_proc=NUM_PROC)

# Initialize the tokenizer
tokenizer = GemmoeTokenizer("tokenizer.model", trust_remote_code=True)

config = GemmoeConfig().from_json_file("config.json")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=2048)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=NUM_PROC)



# Define a simple get_label function if your dataset has labels
# def get_label(example): 
#     # Your logic here to extract label
#     return example["label_field"]

# tokenized_datasets = tokenized_datasets.map(get_label)

# Data collator will dynamically pad the batch during training
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Initialize our model
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    model = GemmoeForCausalLM(config)
model_size = sum(t.numel() for t in model.parameters())
# Visualize model layers
print("Model layers:")
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
        print(f"{name}: {module}")

# Visualize model parameters
print("\nModel parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Visualize model state dict
print("\nModel state dict:")
for key, value in model.state_dict().items():
    print(f"{key}: {value.shape}")

print(f"Model size: {model_size / 1024**2:.2f} M params")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=500,
    weight_decay=0.01,
    bf16=True,                              # use bfloat16 precision
    tf32=True,
    logging_dir='./logs',
)

# Define the compute_metrics function for the evaluation
def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Optionally, plot the training loss
import matplotlib.pyplot as plt

training_stats = trainer.state.log_history

training_loss = [entry['loss'] for entry in training_stats if 'loss' in entry]
eval_loss = [entry['eval_loss'] for entry in training_stats if 'eval_loss' in entry]

plt.plot(training_loss, label='Training Loss')
plt.plot(eval_loss, label='Evaluation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
