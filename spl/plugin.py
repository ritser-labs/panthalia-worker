
from .adapters.dataloader import *
from .adapters.model_config import *
import os
from .adapters.model_adapter import *
from .tokenizer import Tokenizer
import math


BUFFER_SIZE = 100000  # Size of the buffer to shuffle data

current_dir = os.path.dirname(os.path.abspath(__file__))


# IMPORTANT: if you change tokenizer, dont forget to comment out the code in tokenizer.py
# that ignores non ascii characters
# and change pad_id
tokenizer_path = os.path.join(current_dir, 'tokenizers', 'char.tiktoken')

tokenizer = Tokenizer(tokenizer_path)


model_params = GPTConfig(
    block_size=256,
    vocab_size=tokenizer.get_vocab_size(),
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.2,
    bias=True,
    pad_token_id=tokenizer.pad_id
)

model_config = NanoGPTConfig(tokenizer, model_params)

dataset = ShakespeareDataLoader(model_config, buffer_size=BUFFER_SIZE, max_seq_len=model_params.block_size)

model_adapter = NanoGPTModelAdapter(model_config)

class StandardPlugin:
    def __init__(
        self,
        model_adapter,
        model_config,
        dataset,
        tokenizer,
        num_microbatches,
        example_per_microbatch,
        max_lr=0.003,
        min_lr=0.0003,
        tensor_version_interval=30,
        expected_worker_time=24
    ):
        self.model_adapter = model_adapter
        self.model_config = model_config
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_microbatches = num_microbatches
        self.batch_size = num_microbatches * example_per_microbatch
        self.accumulation_steps = num_microbatches
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.tensor_version_interval = tensor_version_interval
        self.expected_worker_time = expected_worker_time
    
    def get_master_learning_hyperparameters(self, current_master_iteration):
        return {
            'accumulation_steps': self.accumulation_steps,
        }
    
    def get_sot_learning_hyperparameters(self, current_iteration):
        """
        Calculate the learning rate using cosine annealing with warm restarts.

        Args:
            current_iteration (int): Current iteration number.

        Returns:
            dict: A dictionary containing the learning rate and Adam optimizer parameters.
        """
        T_0 = 200  # Initial number of iterations for the first cycle
        T_mult = 2  # Factor to increase the cycle length after each restart
        eta_max = self.max_lr * self.num_microbatches  # Initial learning rate (maximum)
        eta_min = self.min_lr * self.num_microbatches  # Minimum learning rate

        # Determine the current cycle length
        cycle_length = T_0
        t = current_iteration
        while current_iteration >= cycle_length:
            current_iteration -= cycle_length
            cycle_length *= T_mult

        # Calculate the learning rate using the cosine annealing formula
        lr = eta_min + (eta_max - eta_min) * (1 + math.cos(math.pi * current_iteration / cycle_length)) / 2

        return {
            'learning_rate': lr,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
            'weight_decay': 0.01,
            't': t,  # Add the current iteration as 't'
            'accumulation_steps': self.accumulation_steps
        }


NUM_MICROBATCHES = 256

EXAMPLES_PER_MICROBATCH = 32

exported_plugin = StandardPlugin(
    model_adapter,
    model_config,
    dataset,
    tokenizer,
    num_microbatches=NUM_MICROBATCHES,
    example_per_microbatch=EXAMPLES_PER_MICROBATCH
)

#model_config = AdderModelConfig()

#dataset = AddNumbersDataLoader()

#model_adapter = AdderModelAdapter(model_config)

#exported_plugin = StandardPlugin(model_adapter, model_config, dataset, tokenizer, num_microbatches=NUM_MICROBATCHES, example_per_microbatch=EXAMPLES_PER_MICROBATCH, expected_worker_time=4)