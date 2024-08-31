import os
import json
import random
from torch.utils.data import IterableDataset
from datasets import load_dataset
from .model_config import BaseModelConfig, TransformerModelConfig

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datasets_dir = os.path.join(parent_dir, 'datasets')

class LanguageDataLoader(IterableDataset):
    def __init__(self, model_config: BaseModelConfig, buffer_size: int):
        self.model_config = model_config
        self.buffer_size = buffer_size
        self.buffer = []

    def __iter__(self):
        max_seq_len = self.model_config.model_args.max_seq_len
        try:
            while True:
                self.buffer = self.fill_buffer_with_token_pairs(self._text_generator(), max_seq_len)
                while self.buffer:
                    yield self.buffer.pop()
        except StopIteration:
            while self.buffer:
                yield self.buffer.pop()

    def __next__(self):
        if not self.buffer:
            self.buffer = list(self.generate_examples())
        if not self.buffer:
            raise StopIteration
        return self.buffer.pop(0)

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
        while start_pos < len(tokens):
            seq_len = random.randint(1, min(max_seq_len, len(tokens) - start_pos))
            substr = tokens[start_pos:start_pos + seq_len]
            inputs = self.truncate_tokens(substr[:-1], max_seq_len, self.model_config.tokenizer.pad_id)
            targets = self.truncate_tokens(substr[1:], max_seq_len, self.model_config.tokenizer.pad_id)
            token_pairs.append((inputs, targets))
            start_pos += seq_len
        return token_pairs

    def fill_buffer_with_token_pairs(self, text_generator, max_seq_len):
        buffer = []
        while len(buffer) < self.buffer_size:
            text = next(text_generator)
            buffer.extend(self.tokenize_and_split(text, max_seq_len))
        random.shuffle(buffer)
        return buffer

    def _text_generator(self):
        """Abstract method, should be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")


class WikipediaDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size):
        super().__init__(model_config, buffer_size)
        self.dataset = load_dataset("wikipedia", "20220301.en", split='train', streaming=True)
        self.dataset_iter = iter(self.dataset)

    def _text_generator(self):
        for example in self.dataset_iter:
            yield example['text']


class ShakespeareDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size, file_path=os.path.join(datasets_dir, 'shakespeare.txt')):
        super().__init__(model_config, buffer_size)
        self.file_path = file_path
        self.lines = self.load_lines()

    def load_lines(self):
        with open(self.file_path, 'r') as f:
            return f.readlines()

    def _text_generator(self):
        while True:
            yield random.choice(self.lines).strip()


class LowercaseAlphabetDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size: int):
        super().__init__(model_config, buffer_size)
        self.alphabet = list('abcdefghijklmnopqrstuvwxyz')

    def _text_generator(self):
        while True:
            start_index = random.randint(0, len(self.alphabet) - 1)
            end_index = random.randint(start_index, len(self.alphabet))
            yield ''.join(self.alphabet[start_index:end_index])

class FineWebDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size):
        super().__init__(model_config, buffer_size)
        self.dataset = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True)
        self.dataset_iter = iter(self.dataset)

    def _text_generator(self):
        for example in self.dataset_iter:
            yield example['text']