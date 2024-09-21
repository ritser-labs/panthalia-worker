import os
import json
import random
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from .model_config import BaseModelConfig, TransformerModelConfig
import time
import asyncio
import aiofiles
import concurrent.futures
import multiprocessing
from ..plugin import exported_plugin

batch_size = exported_plugin.batch_size

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datasets_dir = os.path.join(parent_dir, 'datasets')
executor = concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())



def truncate_tokens(tokens, max_seq_len, pad_token):
    if len(tokens) < max_seq_len:
        tokens += [pad_token] * (max_seq_len - len(tokens))
    elif len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]
    return tokens


def _tokenize_and_split_sync_batch(texts, max_seq_len, tokenizer, pad_id):
    """Synchronous batch tokenization function."""
    token_pairs = []
    for text in texts:
        tokens = tokenizer.encode(text)
        # Truncate or pad tokens
        if len(tokens) < max_seq_len:
            tokens += [pad_id] * (max_seq_len - len(tokens))
        else:
            tokens = tokens[:max_seq_len]
        
        # Create input-target pairs
        inputs = tokens[:-1]
        targets = tokens[1:]
        token_pairs.append((inputs, targets))
    return token_pairs


class LanguageDataLoader:
    def __init__(self, model_config: BaseModelConfig, buffer_size: int, max_seq_len: int):
        self.max_seq_len = max_seq_len
        self.model_config = model_config
        self.buffer_size = buffer_size
        self.buffer = []  # In-memory buffer to hold token pairs

    def __aiter__(self):
        return self

    async def prefetch(self):
        while True:
            if len(self.buffer) < self.buffer_size:
                await self.fill_buffer_with_token_pairs(self._text_generator(), self.max_seq_len)
            await asyncio.sleep(0.1)  # Adjust sleep time as needed

    async def __anext__(self):
        if not hasattr(self, 'buffer_pos'):
            self.buffer_pos = 0
        if self.buffer_pos >= len(self.buffer):
            # Wait until prefetching fills the buffer
            await asyncio.sleep(0.1)
            if self.buffer_pos >= len(self.buffer):
                raise StopAsyncIteration

        token_pair = self.buffer[self.buffer_pos]
        self.buffer_pos += 1
        return token_pair

    async def tokenize_and_split(self, texts, max_seq_len):
        """Use ProcessPoolExecutor for CPU-bound batch tokenization."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            _tokenize_and_split_sync_batch,  # Use the batch function
            texts,
            max_seq_len,
            self.model_config.tokenizer,
            self.model_config.tokenizer.pad_id
        )

    async def fill_buffer_with_token_pairs(self, text_generator, max_seq_len):
        self.buffer = []
        batch = []
        async for text in text_generator:
            batch.append(text)
            if len(batch) == batch_size:
                token_pairs = await self.tokenize_and_split(batch, max_seq_len)
                self.buffer.extend(token_pairs)
                batch = []
                if len(self.buffer) >= self.buffer_size:
                    break
        if batch:
            token_pairs = await self.tokenize_and_split(batch, max_seq_len)
            self.buffer.extend(token_pairs)
        random.shuffle(self.buffer)

class WikipediaDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size, max_seq_len):
        super().__init__(model_config, buffer_size, max_seq_len)
        print("Loading Wikipedia dataset...")
        self.dataset = load_dataset("wikipedia", "20220301.en", split='train', streaming=True)

    async def _text_generator(self):
        for example in self.dataset:
            yield example['text']

class ShakespeareDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size, max_seq_len, file_path=os.path.join(datasets_dir, 'shakespeare.txt'), block_size=124000):
        self.file_path = file_path
        self.block_size = block_size
        self.file_handle = None  # Initialize the file handle for on-demand reading
        super().__init__(model_config, buffer_size, max_seq_len)

    async def _open_file(self):
        """Open the file lazily when needed."""
        if self.file_handle is None:
            try:
                self.file_handle = await aiofiles.open(self.file_path, 'r')
            except Exception as e:
                logging.error(f"Failed to open file {self.file_path}: {e}", exc_info=True)
                raise

    async def _close_file(self):
        """Close the file handle to clean up."""
        if self.file_handle is not None:
            await self.file_handle.close()
            self.file_handle = None

    async def _text_generator(self):

        await self._open_file()  # Ensure the file is opened

        while True:
            try:
                chunk = await self.file_handle.read(self.block_size)  # Read the file in chunks
                if not chunk:
                    await self.file_handle.seek(0)  # Reset to the beginning if EOF is reached
                    continue  # Continue reading from the start
                yield chunk.strip()  # Yield the chunk as the text to process
            except Exception as e:
                logging.error(f"Error reading file {self.file_path}: {e}", exc_info=True)
                raise


class LowercaseAlphabetDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size: int, max_seq_len: int):
        self.alphabet = list('abcdefghijklmnopqrstuvwxyz')
        super().__init__(model_config, buffer_size, max_seq_len)

    async def _text_generator(self):
        while True:
            start_index = random.randint(0, len(self.alphabet) - 1)
            end_index = random.randint(start_index, len(self.alphabet))
            yield ''.join(self.alphabet[start_index:end_index])


class FineWebDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size, max_seq_len):
        print("Loading FineWeb dataset...")
        self.dataset = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True)
        super().__init__(model_config, buffer_size, max_seq_len)

    async def _text_generator(self):
        # Wrap the synchronous dataset iteration in an async generator
        for example in await asyncio.to_thread(self.iterate_dataset):
            yield example['text']

    def iterate_dataset(self):
        """ Synchronous iteration over the dataset """
        for example in self.dataset:
            yield example


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

    async def __iter__(self):
        return self

    async def __anext__(self):
        num1 = random.randint(self.min_value, self.max_value)
        num2 = random.randint(self.min_value, self.max_value)
        
        result = num1 + num2

        return torch.tensor([num1, num2], dtype=torch.float32), torch.tensor([result], dtype=torch.float32)
