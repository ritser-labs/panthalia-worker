import os
import json
import random
from torch.utils.data import IterableDataset
from datasets import load_dataset
from common import tokenizer, model_args

BUFFER_SIZE = 1000

class DataLoaderBase(IterableDataset):
    def __init__(self, buffer_size=BUFFER_SIZE):
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

    def truncate_tokens(self, tokens, max_seq_len, pad_token=tokenizer.pad_id):
        if len(tokens) < max_seq_len:
            tokens += [pad_token] * (max_seq_len - len(tokens))
        elif len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
        return tokens

class WikipediaDataLoader(DataLoaderBase):
    def __init__(self, buffer_size=BUFFER_SIZE):
        super().__init__(buffer_size)
        self.dataset = load_dataset("wikipedia", "20220301.en", split='train', streaming=True)
        self.dataset_iter = iter(self.dataset)

    def __iter__(self):
        max_seq_len = model_args.max_seq_len
        buffer = []

        try:
            while True:
                while len(buffer) < self.buffer_size:
                    example = next(self.dataset_iter)
                    tokens = tokenizer.encode(
                        example['text'],
                        bos=False,
                        eos=False,
                        allowed_special=set(),
                        disallowed_special=(),
                    )

                    for seq_len in range(1, min(len(tokens), max_seq_len) + 1):
                        inputs = self.truncate_tokens(tokens[:seq_len], max_seq_len)
                        targets = self.truncate_tokens(tokens[1:seq_len + 1], max_seq_len)
                        buffer.append((inputs, targets))

                random.shuffle(buffer)
                while buffer:
                    yield buffer.pop()
        except StopIteration:
            while buffer:
                yield buffer.pop()

class ShakespeareDataLoader(DataLoaderBase):
    def __init__(self, buffer_size=BUFFER_SIZE, file_path=os.path.join('datasets', 'shakespeare.txt')):
        super().__init__(buffer_size)
        self.file_path = file_path
        self.lines = self.load_lines()

    def load_lines(self):
        with open(self.file_path, 'r') as f:
            return f.readlines()

    def __iter__(self):
        max_seq_len = model_args.max_seq_len
        buffer = []

        try:
            while True:
                while len(buffer) < self.buffer_size:
                    line = random.choice(self.lines).strip()
                    tokens = tokenizer.encode(
                        line,
                        bos=False,
                        eos=False,
                        allowed_special=set(),
                        disallowed_special=(),
                    )

                    for seq_len in range(1, min(len(tokens), max_seq_len) + 1):
                        inputs = self.truncate_tokens(tokens[:seq_len], max_seq_len)
                        targets = self.truncate_tokens(tokens[1:seq_len + 1], max_seq_len)
                        buffer.append((inputs, targets))

                random.shuffle(buffer)
                while buffer:
                    yield buffer.pop()
        except StopIteration:
            while buffer:
                yield buffer.pop()
