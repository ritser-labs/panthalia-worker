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
import logging

DATALOADER_BATCH_SIZE = 64

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datasets_dir = os.path.join(parent_dir, 'datasets')
executor = concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())


def _tokenize_and_split_sync_batch(texts, max_seq_len, tokenizer):
    """Synchronous batch tokenization function."""
    token_pairs = []
    for text in texts:
        tokens = tokenizer.encode(text)
        # Truncate or pad tokens
        if len(tokens) < max_seq_len:
            tokens += [tokenizer.pad_id] * (max_seq_len - len(tokens))
        else:
            tokens = tokens[:max_seq_len]
        
        # Create input-target pairs
        inputs = tokens[:-1]
        targets = tokens[1:]
        token_pairs.append((inputs, targets))
    return token_pairs


class LanguageDataLoader:
    def __init__(self, model_config, buffer_size, max_seq_len, batch_size=64):
        """
        Generalized class for handling both streaming and file-based datasets.
        Handles tokenization, prefetching, and buffering of text data.
        
        Args:
            model_config (TransformerModelConfig): Model configuration including tokenizer.
            buffer_size (int): Buffer size to hold tokenized text pairs.
            max_seq_len (int): Maximum length of tokenized sequences.
            batch_size (int): Number of text chunks to process in each batch.
        """
        self.max_seq_len = max_seq_len
        self.model_config = model_config
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = []  # Buffer for tokenized data
        self.dataset_iterator = None  # Will be set by subclasses
    
    def init_dataset(self):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not hasattr(self, 'buffer_pos'):
            self.buffer_pos = 0
        if self.buffer_pos >= len(self.buffer):
            # Buffer exhausted, refill
            await self.fill_buffer_with_token_pairs(self._text_generator(), self.max_seq_len)

            if not self.buffer:
                raise StopAsyncIteration

            self.buffer_pos = 0  # Reset buffer position

        # Fetch a batch
        batch_size = min(self.batch_size, len(self.buffer) - self.buffer_pos)
        token_pairs = self.buffer[self.buffer_pos:self.buffer_pos + batch_size]
        self.buffer_pos += batch_size

        return token_pairs  # Returning a list of token pairs

    async def _text_generator(self):
        """
        Abstract method to be implemented by subclasses for generating text data.
        This method should yield text data.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    async def _load_file_in_chunks(self, file_path, block_size):
        """
        Read a file asynchronously in chunks.

        Args:
            file_path (str): Path to the file to read.
            block_size (int): Size of each chunk to read.

        Yields:
            str: Chunks of text from the file.
        """
        try:
            async with aiofiles.open(file_path, 'r') as f:
                while True:
                    chunk = await f.read(block_size)
                    if not chunk:
                        break
                    yield chunk.strip()
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}", exc_info=True)
            raise

    async def _load_dataset_in_chunks(self):
        """
        Yield text data from a streaming dataset asynchronously.

        Yields:
            str: Text examples from the dataset.
        """
        while True:
            try:
                example = next(self.dataset_iterator)
                text = example.get('text', None)
                if text:
                    yield text
            except StopIteration:
                logging.info("Dataset iterator exhausted. Restarting iterator.")
                self.dataset_iterator = iter(self.dataset)
                await asyncio.sleep(1)  # Avoid tight loop
            except Exception as e:
                logging.error(f"Error in _load_dataset_in_chunks: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def fill_buffer_with_token_pairs(self, text_generator, max_seq_len):
        """
        Fill buffer with tokenized text pairs using a given text generator.

        Args:
            text_generator (generator): An async generator that yields text data.
            max_seq_len (int): The maximum sequence length for tokenization.
        """
        self.buffer = []
        batch = []
        async for text in text_generator:
            if text is None:
                logging.warning("Received None text from generator.")
                continue
            batch.append(text)
            if len(batch) == self.batch_size:
                token_pairs = await self.tokenize_and_split(batch, max_seq_len)
                self.buffer.extend(token_pairs)
                logging.debug(f"Added {len(token_pairs)} token pairs to buffer. Buffer size: {len(self.buffer)}")
                batch = []
                if len(self.buffer) >= self.buffer_size:
                    break
        if batch:
            token_pairs = await self.tokenize_and_split(batch, max_seq_len)
            self.buffer.extend(token_pairs)
            logging.debug(f"Added remaining {len(token_pairs)} token pairs to buffer. Buffer size: {len(self.buffer)}")
        random.shuffle(self.buffer)

    async def tokenize_and_split(self, texts, max_seq_len):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor, _tokenize_and_split_sync_batch, texts, max_seq_len, self.model_config.tokenizer
        )


class WikipediaDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size, max_seq_len, batch_size=64):
        """
        WikipediaDataLoader for loading the Wikipedia dataset.
        Inherits common functionality from LanguageDataLoader.
        """
        super().__init__(model_config, buffer_size, max_seq_len, batch_size)
    
    def init_dataset(self):
        self.dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        self.dataset_iterator = iter(self.dataset)

    async def _text_generator(self):
        """
        Override the _text_generator method to yield text from the Wikipedia dataset.
        """
        async for text in self._load_dataset_in_chunks():
            yield text

class FineWebDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size, max_seq_len, batch_size=64):
        """
        FineWebDataLoader for loading the FineWeb dataset.
        Inherits common functionality from LanguageDataLoader.
        """
        super().__init__(model_config, buffer_size, max_seq_len, batch_size)
    
    def init_dataset(self):
        self.dataset = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True)
        self.dataset_iterator = iter(self.dataset)

    async def _text_generator(self):
        """
        Override the _text_generator method to yield text from the FineWeb dataset.
        """
        async for text in self._load_dataset_in_chunks():
            yield text


class ShakespeareDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size, max_seq_len, file_path=os.path.join(datasets_dir, 'shakespeare.txt'), block_size=124000, batch_size=64):
        """
        ShakespeareDataLoader for loading text data from a file.
        Inherits common functionality from LanguageDataLoader.
        
        Args:
            file_path (str): Path to the Shakespeare text file.
            block_size (int): Size of each chunk to read from the file.
            batch_size (int): Number of text chunks to process in each batch.
        """
        super().__init__(model_config, buffer_size, max_seq_len, batch_size)
        self.file_path = file_path
        self.block_size = block_size

    async def _text_generator(self):
        """
        Override the _text_generator method to yield text chunks from the Shakespeare file.
        """
        async for chunk in self._load_file_in_chunks(self.file_path, self.block_size):
            yield chunk


class LowercaseAlphabetDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size: int, max_seq_len: int):
        self.alphabet = list('abcdefghijklmnopqrstuvwxyz')
        super().__init__(model_config, buffer_size, max_seq_len)

    async def _text_generator(self):
        while True:
            start_index = random.randint(0, len(self.alphabet) - 1)
            end_index = random.randint(start_index, len(self.alphabet))
            yield ''.join(self.alphabet[start_index:end_index])


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
