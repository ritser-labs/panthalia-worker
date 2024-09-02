import os
import json
import random
import tempfile
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from .model_config import BaseModelConfig, TransformerModelConfig

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datasets_dir = os.path.join(parent_dir, 'datasets')

class LanguageDataLoader(IterableDataset):
    def __init__(self, model_config: BaseModelConfig, buffer_size: int, max_seq_len: int):
        self.max_seq_len = max_seq_len
        self.model_config = model_config
        self.buffer_size = buffer_size
        self.buffer_file = tempfile.NamedTemporaryFile(delete=False, mode='w+b')  # Disk-based buffer file
        self.buffer_file_path = self.buffer_file.name
        self.buffer_file.close()
        self.fill_buffer_with_token_pairs(self._text_generator(), self.max_seq_len)

    def __iter__(self):
        self.buffer_pos = 0
        self.buffer = open(self.buffer_file_path, 'rb')  # Open file for reading
        return self

    def __next__(self):
        line = self.buffer.readline()
        if not line:
            # Buffer exhausted, close and refill
            self.buffer.close()
            self.fill_buffer_with_token_pairs(self._text_generator(), self.max_seq_len)
            self.buffer = open(self.buffer_file_path, 'rb')  # Reopen the buffer file
            line = self.buffer.readline()  # Read the next line after refilling

            if not line:
                self.buffer.close()
                raise StopIteration

        return json.loads(line.decode('utf-8'))

    def truncate_tokens(self, tokens, max_seq_len, pad_token):
        if len(tokens) < max_seq_len:
            tokens += [pad_token] * (max_seq_len - len(tokens))
        elif len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
        return tokens

    def tokenize_and_split(self, text, max_seq_len):
        tokens = self.model_config.tokenizer.encode(
            text,
            bos=False,
            eos=False,
            allowed_special=set(),
            disallowed_special=(),
        )

        start_pos = 0
        token_pairs = []
        while start_pos < len(tokens) - 1:
            upper_bound = min(len(tokens) - start_pos, max_seq_len)
            #seq_len = random.randint(2, upper_bound)
            seq_len = upper_bound
            substr = tokens[start_pos:start_pos + seq_len]
            inputs = self.truncate_tokens(substr[:-1], max_seq_len, self.model_config.tokenizer.pad_id)
            targets = self.truncate_tokens(substr[1:], max_seq_len, self.model_config.tokenizer.pad_id)
            token_pairs.append((inputs, targets))
            start_pos += seq_len
        return token_pairs

    def fill_buffer_with_token_pairs(self, text_generator, max_seq_len):
        with open(self.buffer_file_path, 'wb') as f:
            buffer = []
            for text in text_generator:
                token_pairs = self.tokenize_and_split(text, max_seq_len)
                buffer.extend(token_pairs)
                if len(buffer) >= self.buffer_size:
                    break

            random.shuffle(buffer)

            for pair in buffer:
                f.write(json.dumps(pair).encode('utf-8') + b'\n')

    def _text_generator(self):
        """Abstract method, should be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")


class WikipediaDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size, max_seq_len):
        super().__init__(model_config, buffer_size, max_seq_len)
        self.dataset = load_dataset("wikipedia", "20220301.en", split='train', streaming=True)
        self.dataset_iter = iter(self.dataset)

    def _text_generator(self):
        for example in self.dataset_iter:
            yield example['text']


class ShakespeareDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size, max_seq_len, file_path=os.path.join(datasets_dir, 'shakespeare.txt'), block_size=124000):
        super().__init__(model_config, buffer_size, max_seq_len)
        self.file_path = file_path
        self.lines = self.load_lines()
        self.block_size = block_size

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
        super().__init__(model_config, buffer_size, max_seq_len)
        self.dataset = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True)
        self.dataset_iter = iter(self.dataset)

    def _text_generator(self):
        for example in self.dataset_iter:
            yield example['text']

class AddNumbersDataLoader(IterableDataset):
    def __init__(self, min_value: int = 0, max_value: int = 100):
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
