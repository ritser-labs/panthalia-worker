import os
import json
import random
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from .model_config import BaseModelConfig, TransformerModelConfig
import time


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datasets_dir = os.path.join(parent_dir, 'datasets')

class LanguageDataLoader(IterableDataset):
    def __init__(self, model_config: BaseModelConfig, buffer_size: int, max_seq_len: int):
        self.max_seq_len = max_seq_len
        self.model_config = model_config
        self.buffer_size = buffer_size
        self.buffer = []  # In-memory buffer to hold token pairs

        start_time = time.time()
        self.fill_buffer_with_token_pairs(self._text_generator(), self.max_seq_len)
        end_time = time.time()

    def __iter__(self):
        self.buffer_pos = 0  # Reset position when creating a new iterator
        return self

    def __next__(self):
        if self.buffer_pos >= len(self.buffer):
            # Buffer exhausted, refill
            buffer_refill_start_time = time.time()
            self.fill_buffer_with_token_pairs(self._text_generator(), self.max_seq_len)
            buffer_refill_end_time = time.time()

            if not self.buffer:
                raise StopIteration

            self.buffer_pos = 0  # Reset buffer position

        line_start_time = time.time()
        token_pair = self.buffer[self.buffer_pos]
        self.buffer_pos += 1  # Move to the next pair
        line_end_time = time.time()

        return token_pair

    def truncate_tokens(self, tokens, max_seq_len, pad_token):
        truncate_start_time = time.time()

        if len(tokens) < max_seq_len:
            tokens += [pad_token] * (max_seq_len - len(tokens))
        elif len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]

        truncate_end_time = time.time()
        return tokens

    def tokenize_and_split(self, text, max_seq_len):
        tokenize_start_time = time.time()
        tokens = self.model_config.tokenizer.encode(text)
        tokenize_end_time = time.time()

        split_start_time = time.time()
        token_pairs = []
        
        # Calculate how many chunks can be made based on the total length and max_seq_len
        num_chunks = len(tokens) // max_seq_len

        for _ in range(num_chunks):
            # Randomly pick a start position ensuring we have enough tokens for a sequence
            start_pos = random.randint(0, len(tokens) - max_seq_len - 1)

            # Extract the sequence of tokens from the random start position
            seq_len = max_seq_len
            substr = tokens[start_pos:start_pos + seq_len]

            # Create input-target pairs where targets are the next token shifted by one
            inputs = self.truncate_tokens(substr[:-1], max_seq_len, self.model_config.tokenizer.pad_id)
            targets = self.truncate_tokens(substr[1:], max_seq_len, self.model_config.tokenizer.pad_id)

            # Append the pair to the list
            token_pairs.append((inputs, targets))

        split_end_time = time.time()
        return token_pairs

    def fill_buffer_with_token_pairs(self, text_generator, max_seq_len):
        buffer_fill_start_time = time.time()
        self.buffer = []  # Clear the buffer before refilling

        tokenization_start_time = time.time()
        for text in text_generator:
            token_pairs = self.tokenize_and_split(text, max_seq_len)
            self.buffer.extend(token_pairs)
            if len(self.buffer) >= self.buffer_size:
                break
        tokenization_end_time = time.time()

        random.shuffle(self.buffer)

        buffer_fill_end_time = time.time()



class WikipediaDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size, max_seq_len):
        dataset_load_start_time = time.time()
        super().__init__(model_config, buffer_size, max_seq_len)
        self.dataset = load_dataset("wikipedia", "20220301.en", split='train', streaming=True)
        self.dataset_iter = iter(self.dataset)
        dataset_load_end_time = time.time()

    def _text_generator(self):
        for example in self.dataset_iter:
            yield example['text']


class ShakespeareDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size, max_seq_len, file_path=os.path.join(datasets_dir, 'shakespeare.txt'), block_size=124000):
        self.file_path = file_path
        self.block_size = block_size

        self.lines = self.load_lines()

        super().__init__(model_config, buffer_size, max_seq_len)

    def load_lines(self):
        with open(self.file_path, 'r') as f:
            return f.readlines()

    def _text_generator(self):
        num_lines = len(self.lines)
        start_index = 0

        while True:
            end_index = start_index + self.block_size
            if end_index > num_lines:
                start_index = 0
                end_index = self.block_size

            yield ''.join(self.lines[start_index:end_index]).strip()
            start_index = end_index

            if start_index >= num_lines:
                start_index = 0

class LowercaseAlphabetDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size: int, max_seq_len: int):
        self.alphabet = list('abcdefghijklmnopqrstuvwxyz')
        super().__init__(model_config, buffer_size, max_seq_len)

    def _text_generator(self):
        while True:
            start_index = random.randint(0, len(self.alphabet) - 1)
            end_index = random.randint(start_index, len(self.alphabet))
            yield ''.join(self.alphabet[start_index:end_index])

class FineWebDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size, max_seq_len):
        dataset_load_start_time = time.time()
        super().__init__(model_config, buffer_size, max_seq_len)
        self.dataset = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True)
        self.dataset_iter = iter(self.dataset)
        dataset_load_end_time = time.time()

    def _text_generator(self):
        for example in self.dataset_iter:
            yield example['text']

class AddNumbersDataLoader(IterableDataset):
    def __init__(self, min_value: int = -100, max_value: int = 100):
        """
        DataLoader for generating pairs of numbers and their sum indefinitely.
        
        Args:
            min_value (int): Minimum value for the random numbers.
            max_value (int): Maximum value for the random numbers.
        """
        self.min_value = min_value
        self.max_value = max_value

    def __iter__(self):
        """
        Create an iterator that yields pairs of numbers and their sum indefinitely.
        """
        return self

    def __next__(self):
        """
        Generate the next sample indefinitely.
        
        Returns:
            Tuple: A tuple containing a tensor of two input numbers and a tensor with their sum.
        """
        # Generate two random numbers within the specified range
        num1 = random.randint(self.min_value, self.max_value)
        num2 = random.randint(self.min_value, self.max_value)
        
        # Calculate their sum
        result = num1 + num2

        # Return a tuple of input tensor and output tensor
        return torch.tensor([num1, num2], dtype=torch.float32), torch.tensor([result], dtype=torch.float32)
