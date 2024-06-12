import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import numpy as np
from sklearn.metrics import accuracy_score
from modeling_gemmoe import GemmoeForCausalLM
from tokenization_gemmoe import GemmoeTokenizer
from configuration_gemmoe import GemmoeConfig

# Load the dataset
dataset = load_dataset("wikipedia", language="sw", date="20240401", trust_remote_code=True)

# Initialize the tokenizer
tokenizer = GemmoeTokenizer("tokenizer.model", trust_remote_code=True)

config = GemmoeConfig(name_or_path="config.json")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define a simple get_label function if your dataset has labels
# def get_label(example): 
#     # Your logic here to extract label
#     return example["label_field"]

# tokenized_datasets = tokenized_datasets.map(get_label)

# Use a smaller dataset for training to speed things up
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(1000))

# Data collator will dynamically pad the batch during training
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Initialize our model
model = GemmoeForCausalLM(config, trust_remote_code=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    evaluate_during_training=True,
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
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
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
