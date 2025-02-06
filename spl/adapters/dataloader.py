import os
import random
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from .model_config import TransformerModelConfig
import asyncio
import aiofiles
import concurrent.futures
import multiprocessing
import logging
import time

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datasets_dir = os.path.join(parent_dir, 'datasets')


def _tokenize_and_split_sync_batch(text, max_seq_len, tokenizer):
    """
    Synchronously tokenizes a single text and splits it into multiple token pairs.
    """
    token_pairs = []
    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)

    # Calculate number of samples, ensure at least 1
    num_samples = max(1, total_tokens // max_seq_len)

    # We want (max_seq_len + 1) tokens per sample, so we can form (input, target)
    desired_sample_len = max_seq_len + 1

    for i in range(num_samples):
        start = i * max_seq_len
        end = start + desired_sample_len
        sample_tokens = tokens[start:end]

        # Pad if not enough tokens
        if len(sample_tokens) < desired_sample_len:
            sample_tokens += [tokenizer.pad_id] * (desired_sample_len - len(sample_tokens))

        # inputs are first max_seq_len, targets are shifted by 1
        inputs = sample_tokens[:max_seq_len]
        targets = sample_tokens[1:desired_sample_len]
        token_pairs.append((inputs, targets))

    return token_pairs


class LanguageDataLoader:
    """
    Common base for all text-like loaders. Each child class must implement
    or override `_text_generator` to produce text strings. Then we:
      1) Tokenize/split them into (input, target) pairs
      2) Fill an async buffer with batches
      3) Let clients consume them via __anext__
    """

    def __init__(self, model_config, buffer_size, max_seq_len, batch_size):
        self.max_seq_len = max_seq_len
        self.model_config = model_config
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.buffer = asyncio.Queue(maxsize=self.buffer_size)  # to hold the final (inputs,targets) batches
        self.dataset_iterator = None

        self._stop_event = asyncio.Event()
        self._filler_task = None  # the background filler
        self._token_pair_buffer = []  # accumulates token pairs until we form a batch
        self.executor = None  # for run_in_executor

    async def _buffer_filler(self):
        """
        Background task: repeatedly gather text => tokenize => form full batch => put in self.buffer.
        """
        while not self._stop_event.is_set():
            try:
                # Accumulate token pairs until we have enough for a full batch
                while len(self._token_pair_buffer) < self.batch_size:
                    text = await self._get_text()
                    if text is None:
                        await asyncio.sleep(0.1)
                        continue

                    token_pairs = await self.tokenize_and_split(text, self.max_seq_len)
                    if not token_pairs:
                        continue

                    random.shuffle(token_pairs)
                    self._token_pair_buffer.extend(token_pairs)

                # Now slice off a batch
                batch = self._token_pair_buffer[:self.batch_size]
                self._token_pair_buffer = self._token_pair_buffer[self.batch_size:]

                await self.buffer.put(batch)
                logging.debug(f"Added a batch to buffer. Buffer size: {self.buffer.qsize()}")

            except Exception as e:
                logging.error(f"Error in buffer_filler: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def _get_text(self):
        """
        Fetches one chunk of text from self._text_generator().
        """
        try:
            text = await self._text_generator().__anext__()
            return text
        except StopAsyncIteration:
            logging.info("Text generator exhausted. Waiting for more data...")
            await asyncio.sleep(1)
            return None
        except Exception as e:
            logging.error(f"Error in _get_text: {e}", exc_info=True)
            await asyncio.sleep(1)
            return None

    async def __aiter__(self):
        return self

    async def __anext__(self):
        """
        Grab the next batch from the buffer. Raises StopAsyncIteration if empty.
        """
        if self.buffer.empty():
            await asyncio.sleep(0.1)
            if self.buffer.empty():
                raise StopAsyncIteration

        batch = await self.buffer.get()
        logging.debug(f"Retrieved a batch from buffer. Remaining buffer size: {self.buffer.qsize()}")
        return batch

    async def _text_generator(self):
        """
        Must be overridden by child classes to yield text strings asynchronously.
        """
        raise NotImplementedError("Subclasses must override _text_generator.")

    async def initialize_dataset(self):
        """
        Public async init. Child classes can override but should do:
            await super().initialize_dataset()
        to start the filler task.
        """
        if self.executor is None:
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=multiprocessing.cpu_count()
            )
        self._filler_task = asyncio.create_task(self._buffer_filler())

    def stop(self):
        """
        Cancels the buffer filler task.
        """
        if self._filler_task:
            self._stop_event.set()
            self._filler_task.cancel()

    async def _load_file_in_chunks(self, file_path, block_size):
        """
        Asynchronously read a file in chunks, yield strings.
        """
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
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
        Yields text from self.dataset in a loop; re-init if exhausted.
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
                await asyncio.sleep(1)
            except Exception as e:
                logging.error(f"Error in _load_dataset_in_chunks: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def tokenize_and_split(self, text, max_seq_len):
        """
        Offload tokenize/split to ProcessPoolExecutor for speed.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            _tokenize_and_split_sync_batch,
            text, max_seq_len, self.model_config.tokenizer
        )


class WikipediaDataLoader(LanguageDataLoader):
    """
    Streams data from the Wikipedia dataset (huggingface).
    """
    def __init__(self, model_config: TransformerModelConfig, buffer_size, max_seq_len, batch_size):
        super().__init__(model_config, buffer_size, max_seq_len, batch_size)
        self.dataset = None
        self.dataset_iterator = None

    async def initialize_dataset(self):
        # Make sure parent logic runs first (starts the filler, sets executor, etc.)
        await super().initialize_dataset()
        self.dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True).shuffle()
        self.dataset_iterator = iter(self.dataset)

    async def _text_generator(self):
        async for text in self._load_dataset_in_chunks():
            yield text


class FineWebDataLoader(LanguageDataLoader):
    """
    Streams data from the FineWeb dataset.
    """
    def __init__(self, model_config: TransformerModelConfig, buffer_size, max_seq_len, batch_size):
        super().__init__(model_config, buffer_size, max_seq_len, batch_size)
        self.dataset = None
        self.dataset_iterator = None

    async def initialize_dataset(self):
        await super().initialize_dataset()
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name="CC-MAIN-2024-10",
            split="train",
            streaming=True
        ).shuffle()
        self.dataset_iterator = iter(self.dataset)

    async def _text_generator(self):
        async for text in self._load_dataset_in_chunks():
            yield text


class ShakespeareDataLoader(LanguageDataLoader):
    """
    Reads a local Shakespeare text file in chunks, then tokenizes.
    """
    def __init__(self, model_config: TransformerModelConfig, buffer_size, max_seq_len, batch_size,
                 file_path=None, block_size=124000):
        super().__init__(model_config, buffer_size, max_seq_len, batch_size)
        self.file_path = file_path if file_path else os.path.join(datasets_dir, 'shakespeare.txt')
        self.block_size = block_size

    async def initialize_dataset(self):
        # This calls the parent's logic to start the filler
        await super().initialize_dataset()

    async def _text_generator(self):
        async for chunk in self._load_file_in_chunks(self.file_path, self.block_size):
            yield chunk


class LowercaseAlphabetDataLoader(LanguageDataLoader):
    """
    Generates random slices of the lowercase alphabet on the fly.
    """
    def __init__(self, model_config: TransformerModelConfig, buffer_size: int, max_seq_len: int, batch_size):
        super().__init__(model_config, buffer_size, max_seq_len, batch_size)
        self.alphabet = list('abcdefghijklmnopqrstuvwxyz')

    async def initialize_dataset(self):
        await super().initialize_dataset()

    async def _text_generator(self):
        while True:
            start_index = random.randint(0, len(self.alphabet) - 1)
            end_index = random.randint(start_index + 1, len(self.alphabet))
            yield ''.join(self.alphabet[start_index:end_index])


class AddNumbersDataLoader(IterableDataset):
    """
    A completely separate toy class that doesn't use LanguageDataLoader at all.
    Generates random (num1, num2) => sum, buffered asynchronously.
    """

    def __init__(self, min_value: int = -100, max_value: int = 100,
                 buffer_size: int = 1024, batch_size: int = 32):
        self.min_value = min_value
        self.max_value = max_value
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = asyncio.Queue(maxsize=self.buffer_size)
        self._stop_event = asyncio.Event()
        self._filler_task = asyncio.create_task(self._fill_buffer())

    async def _fill_buffer(self):
        while not self._stop_event.is_set():
            try:
                batch = []
                for _ in range(self.batch_size):
                    num1 = random.randint(self.min_value, self.max_value)
                    num2 = random.randint(self.min_value, self.max_value)
                    result = num1 + num2
                    batch.append((
                        torch.tensor([num1, num2], dtype=torch.float32),
                        torch.tensor([result], dtype=torch.float32)
                    ))
                await self.buffer.put(batch)
                logging.debug(f"Added a number batch to buffer. Size: {self.buffer.qsize()}")
            except Exception as e:
                logging.error(f"AddNumbersDataLoader _fill_buffer error: {e}", exc_info=True)
                await asyncio.sleep(1)

    def __iter__(self):
        return self

    def __next__(self):
        if self.buffer.empty():
            time.sleep(0.01)
            if self.buffer.empty():
                raise StopIteration
        try:
            batch = self.buffer.get_nowait()
            return batch
        except asyncio.QueueEmpty:
            raise StopIteration

    def stop(self):
        self._stop_event.set()
        self._filler_task.cancel()

    def __len__(self):
        return self.buffer_size
