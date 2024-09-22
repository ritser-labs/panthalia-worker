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

# Initialize global variables
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datasets_dir = os.path.join(parent_dir, 'datasets')
executor = concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())

def _tokenize_and_split_sync_batch(text, max_seq_len, tokenizer):
    """
    Synchronously tokenizes a single text and splits it into multiple token pairs.

    Args:
        text (str): The text to tokenize.
        max_seq_len (int): The maximum sequence length for each token pair.
        tokenizer: Tokenizer with `encode` and `pad_id` attributes.

    Returns:
        list of tuples: Each tuple contains (inputs, targets) token lists.
    """
    token_pairs = []
    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)

    # Calculate number of samples, ensure at least 1
    num_samples = max(1, (total_tokens + max_seq_len - 1) // max_seq_len)

    for i in range(num_samples):
        start = i * max_seq_len
        end = start + max_seq_len
        sample_tokens = tokens[start:end]

        # Ensure the sample has exactly max_seq_len tokens
        if len(sample_tokens) < max_seq_len:
            sample_tokens += [tokenizer.pad_id] * (max_seq_len - len(sample_tokens))
        else:
            sample_tokens = sample_tokens[:max_seq_len]

        inputs = sample_tokens[:-1]
        targets = sample_tokens[1:]
        token_pairs.append((inputs, targets))

    return token_pairs

class LanguageDataLoader:
    def __init__(self, model_config, buffer_size, max_seq_len, batch_size):
        """
        Generalized class for handling both streaming and file-based datasets.
        Handles tokenization, prefetching, and buffering of text data.

        Args:
            model_config (TransformerModelConfig): Model configuration including tokenizer.
            buffer_size (int): Number of batches to hold in the buffer.
            max_seq_len (int): Maximum length of tokenized sequences.
            batch_size (int): Number of token pairs per batch.
        """
        self.max_seq_len = max_seq_len
        self.model_config = model_config
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.buffer = asyncio.Queue(maxsize=self.buffer_size)  # Buffer to hold batches
        self.dataset_iterator = None  # To be initialized by subclasses

        self._stop_event = asyncio.Event()  # Event to signal stopping the buffer filler
        self._filler_task = None  # Background buffer filler task
        self._token_pair_buffer = []  # Temporary buffer to accumulate token pairs

    async def _buffer_filler(self):
        """
        Background task to continuously fill the buffer with batches.
        Accumulates token pairs across multiple texts until a full batch is ready.
        """
        while not self._stop_event.is_set():
            try:
                # Accumulate token pairs until we have enough for a full batch
                while len(self._token_pair_buffer) < self.batch_size:
                    text = await self._get_text()
                    if text is None:
                        await asyncio.sleep(0.1)
                        continue

                    # Tokenize and split the text into multiple token pairs
                    token_pairs = await self.tokenize_and_split(text, self.max_seq_len)
                    if not token_pairs:
                        continue

                    # Shuffle token pairs to ensure randomness
                    random.shuffle(token_pairs)

                    # Add token pairs to the temporary buffer
                    self._token_pair_buffer.extend(token_pairs)
                    #logging.debug(f"Accumulated {len(token_pairs)} token pairs. Total in buffer: {len(self._token_pair_buffer)}")

                # Form a batch from the accumulated token pairs
                batch = self._token_pair_buffer[:self.batch_size]
                self._token_pair_buffer = self._token_pair_buffer[self.batch_size:]

                # Put the batch into the buffer
                await self.buffer.put(batch)
                logging.debug(f"Added a batch to buffer. Buffer size: {self.buffer.qsize()}")

            except Exception as e:
                logging.error(f"Error in buffer_filler: {e}", exc_info=True)
                await asyncio.sleep(1)  # Prevent tight loop on error

    async def _get_text(self):
        """
        Fetches a single text from the generator.

        Returns:
            str or None: The fetched text or None if unavailable.
        """
        try:
            text = await self._text_generator().__anext__()
            #logging.debug("Fetched a new text from generator.")
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
        Retrieves the next batch from the buffer.

        Returns:
            list of tuples: Each tuple contains (inputs, targets) token lists.

        Raises:
            StopAsyncIteration: If the buffer is empty and no more data is available.
        """
        if self.buffer.empty():
            await asyncio.sleep(0.1)  # Wait for the buffer to be filled
            if self.buffer.empty():
                raise StopAsyncIteration

        batch = await self.buffer.get()
        logging.debug(f"Retrieved a batch from buffer. Remaining buffer size: {self.buffer.qsize()}")
        return batch

    async def _text_generator(self):
        """
        Abstract method to be implemented by subclasses for generating text data.
        This method should yield text data asynchronously.
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

    async def tokenize_and_split(self, text, max_seq_len):
        """
        Tokenize and split a single text into multiple input-target pairs.

        Args:
            text (str): The text to tokenize and split.
            max_seq_len (int): The maximum sequence length for each token pair.

        Returns:
            list of tuples: Each tuple contains (inputs, targets) token lists.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor, _tokenize_and_split_sync_batch, text, max_seq_len, self.model_config.tokenizer
        )

    def init_dataset(self):
        """
        Abstract method to initialize the dataset.
        Should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def stop(self):
        """
        Signals the buffer filler to stop and waits for the task to finish.
        """
        if self._filler_task:
            self._stop_event.set()
            self._filler_task.cancel()

class WikipediaDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size, max_seq_len, batch_size):
        """
        WikipediaDataLoader for loading the Wikipedia dataset.
        Inherits common functionality from LanguageDataLoader.

        Args:
            model_config (TransformerModelConfig): Model configuration including tokenizer.
            buffer_size (int): Number of batches to hold in the buffer.
            max_seq_len (int): Maximum length of tokenized sequences.
            batch_size (int): Number of token pairs per batch.
        """
        super().__init__(model_config, buffer_size, max_seq_len, batch_size)
        self.dataset = None
        self.dataset_iterator = None

    def init_dataset(self):
        """
        Initialize the Wikipedia streaming dataset and its iterator.
        """
        self.dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        self.dataset_iterator = iter(self.dataset)
        self._filler_task = asyncio.create_task(self._buffer_filler())

    async def _text_generator(self):
        """
        Override the _text_generator method to yield text from the Wikipedia dataset.
        """
        async for text in self._load_dataset_in_chunks():
            yield text

class FineWebDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size, max_seq_len, batch_size):
        """
        FineWebDataLoader for loading the FineWeb dataset.
        Inherits common functionality from LanguageDataLoader.

        Args:
            model_config (TransformerModelConfig): Model configuration including tokenizer.
            buffer_size (int): Number of batches to hold in the buffer.
            max_seq_len (int): Maximum length of tokenized sequences.
            batch_size (int): Number of token pairs per batch.
        """
        super().__init__(model_config, buffer_size, max_seq_len, batch_size)
        self.dataset = None
        self.dataset_iterator = None

    def init_dataset(self):
        """
        Initialize the FineWeb streaming dataset and its iterator.
        """
        self.dataset = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True)
        self.dataset_iterator = iter(self.dataset)
        self._filler_task = asyncio.create_task(self._buffer_filler())

    async def _text_generator(self):
        """
        Override the _text_generator method to yield text from the FineWeb dataset.
        """
        async for text in self._load_dataset_in_chunks():
            yield text

class ShakespeareDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size, max_seq_len, batch_size, file_path=None, block_size=124000):
        """
        ShakespeareDataLoader for loading text data from a file.
        Inherits common functionality from LanguageDataLoader.

        Args:
            model_config (TransformerModelConfig): Model configuration including tokenizer.
            buffer_size (int): Number of batches to hold in the buffer.
            max_seq_len (int): Maximum length of tokenized sequences.
            batch_size (int): Number of token pairs per batch.
            file_path (str, optional): Path to the Shakespeare text file. Defaults to None.
            block_size (int, optional): Size of each chunk to read from the file. Defaults to 124000.
        """
        self.file_path = file_path if file_path else os.path.join(datasets_dir, 'shakespeare.txt')
        self.block_size = block_size
        super().__init__(model_config, buffer_size, max_seq_len, batch_size)

    def init_dataset(self):
        """
        ShakespeareDataLoader does not use a streaming dataset.
        Initialization is handled differently.
        """
        # No dataset to initialize for file-based loading
        self._filler_task = asyncio.create_task(self._buffer_filler())

    async def _text_generator(self):
        """
        Override the _text_generator method to yield text chunks from the Shakespeare file.
        """
        async for chunk in self._load_file_in_chunks(self.file_path, self.block_size):
            yield chunk

class LowercaseAlphabetDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size: int, max_seq_len: int, batch_size):
        """
        LowercaseAlphabetDataLoader for generating random lowercase alphabet sequences.
        Inherits common functionality from LanguageDataLoader.

        Args:
            model_config (TransformerModelConfig): Model configuration including tokenizer.
            buffer_size (int): Number of batches to hold in the buffer.
            max_seq_len (int): Maximum length of tokenized sequences.
            batch_size (int): Number of token pairs per batch.
        """
        self.alphabet = list('abcdefghijklmnopqrstuvwxyz')
        super().__init__(model_config, buffer_size, max_seq_len, batch_size)

    def init_dataset(self):
        """
        LowercaseAlphabetDataLoader does not use an external dataset.
        Initialization is not required.
        """
        # No dataset to initialize
        self._filler_task = asyncio.create_task(self._buffer_filler())

    async def _text_generator(self):
        """
        Override the _text_generator method to yield random lowercase alphabet sequences.
        """
        while True:
            start_index = random.randint(0, len(self.alphabet) - 1)
            end_index = random.randint(start_index + 1, len(self.alphabet))  # Ensure at least one character
            yield ''.join(self.alphabet[start_index:end_index])

class AddNumbersDataLoader(IterableDataset):
    def __init__(self, min_value: int = -100, max_value: int = 100, buffer_size: int = 1024, batch_size: int = 32):
        """
        DataLoader for generating pairs of numbers and their sum indefinitely.

        Args:
            min_value (int): Minimum value for the random numbers.
            max_value (int): Maximum value for the random numbers.
            buffer_size (int): Number of batches to buffer.
            batch_size (int): Number of samples per batch.
        """
        self.min_value = min_value
        self.max_value = max_value
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = asyncio.Queue(maxsize=self.buffer_size)  # Buffer to hold batches
        self._stop_event = asyncio.Event()  # Event to signal stopping the buffer filler
        self._filler_task = asyncio.create_task(self._fill_buffer())

    async def _fill_buffer(self):
        """
        Background task to continuously fill the buffer with number batches.
        """
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
                logging.debug(f"Added a number batch to buffer. Buffer size: {self.buffer.qsize()}")
            except Exception as e:
                logging.error(f"Error in AddNumbersDataLoader _fill_buffer: {e}", exc_info=True)
                await asyncio.sleep(1)

    def __iter__(self):
        return self

    def __next__(self):
        if self.buffer.empty():
            time.sleep(0.01)  # Wait for the buffer to fill
            if self.buffer.empty():
                raise StopIteration
        try:
            batch = self.buffer.get_nowait()
            logging.debug(f"Retrieved a number batch from buffer. Remaining buffer size: {self.buffer.qsize()}")
            return batch
        except asyncio.QueueEmpty:
            raise StopIteration

    def stop(self):
        """
        Signals the buffer filler to stop and waits for the task to finish.
        """
        self._stop_event.set()
        self._filler_task.cancel()

    def __len__(self):
        return self.buffer_size
