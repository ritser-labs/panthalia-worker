import os
import json
import random
from torch.utils.data import IterableDataset
from datasets import load_dataset
from model_config import BaseModelConfig, TransformerModelConfig

class LanguageDataLoader(IterableDataset):
    def __init__(self, model_config: BaseModelConfig, buffer_size: int):
        self.model_config = model_config
        self.buffer_size = buffer_size
        self.buffer = []
    
    def __iter__(self):
        raise NotImplementedError("This method needs to be implemented by subclasses")

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

class WikipediaDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size):
        super().__init__(model_config, buffer_size)
        self.dataset = load_dataset("wikipedia", "20220301.en", split='train', streaming=True)
        self.dataset_iter = iter(self.dataset)

    def __iter__(self):
        max_seq_len = self.model_config.model_args.max_seq_len
        buffer = []

        try:
            while True:
                while len(buffer) < self.buffer_size:
                    example = next(self.dataset_iter)
                    tokens = self.model_config.tokenizer.encode(
                        example['text'],
                        bos=False,
                        eos=False,
                        allowed_special=set(),
                        disallowed_special=(),
                    )

                    for seq_len in range(1, min(len(tokens), max_seq_len) + 1):
                        inputs = self.truncate_tokens(tokens[:seq_len], max_seq_len, self.model_config.tokenizer.pad_id)
                        targets = self.truncate_tokens(tokens[1:seq_len + 1], max_seq_len, self.model_config.tokenizer.pad_id)
                        buffer.append((inputs, targets))

                random.shuffle(buffer)
                while buffer:
                    yield buffer.pop()
        except StopIteration:
            while buffer:
                yield buffer.pop()

class ShakespeareDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size, file_path=os.path.join('datasets', 'shakespeare.txt')):
        super().__init__(model_config, buffer_size)
        self.file_path = file_path
        self.lines = self.load_lines()

    def load_lines(self):
        with open(self.file_path, 'r') as f:
            return f.readlines()

    def __iter__(self):
        max_seq_len = self.model_config.model_args.max_seq_len
        buffer = []

        try:
            while True:
                while len(buffer) < self.buffer_size:
                    line = random.choice(self.lines).strip()
                    tokens = self.model_config.tokenizer.encode(
                        line,
                        bos=False,
                        eos=False,
                        allowed_special=set(),
                        disallowed_special=(),
                    )

                    for seq_len in range(1, min(len(tokens), max_seq_len) + 1):
                        inputs = self.truncate_tokens(tokens[:seq_len], max_seq_len, self.model_config.tokenizer.pad_id)
                        targets = self.truncate_tokens(tokens[1:seq_len + 1], max_seq_len, self.model_config.tokenizer.pad_id)
                        buffer.append((inputs, targets))

                random.shuffle(buffer)
                while buffer:
                    yield buffer.pop()
        except StopIteration:
            while buffer:
                yield buffer.pop()


class LowercaseAlphabetDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size: int):
        super().__init__(model_config, buffer_size)
        self.alphabet = list('abcdefghijklmnopqrstuvwxyz')

    def __iter__(self):
        max_seq_len = self.model_config.model_args.max_seq_len
        buffer = []

        try:
            while True:
                while len(buffer) < self.buffer_size:
                    start_index = random.randint(0, len(self.alphabet) - 1)
                    end_index = random.randint(start_index, len(self.alphabet))
                    sequence = ''.join(self.alphabet[start_index:end_index])

                    tokens = self.model_config.tokenizer.encode(
                        sequence,
                        bos=False,
                        eos=False,
                        allowed_special=set(),
                        disallowed_special=(),
                    )

                    for seq_len in range(1, min(len(tokens), max_seq_len) + 1):
                        inputs = self.truncate_tokens(tokens[:seq_len], max_seq_len, self.model_config.tokenizer.pad_id)
                        targets = self.truncate_tokens(tokens[1:seq_len + 1], max_seq_len, self.model_config.tokenizer.pad_id)
                        buffer.append((inputs, targets))

                random.shuffle(buffer)
                while buffer:
                    yield buffer.pop()
        except StopIteration:
            while buffer:
                yield buffer.pop()
