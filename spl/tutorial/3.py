from ..adapters.dataloader import LanguageDataLoader, datasets_dir


# Previous code


# Defining the data loader
# This defines the training data we'll use
class ShakespeareDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size, max_seq_len, batch_size, file_path=None, block_size=124000):
        self.file_path = file_path if file_path else os.path.join(datasets_dir, 'shakespeare.txt')
        self.block_size = block_size
        super().__init__(model_config, buffer_size, max_seq_len, batch_size)

    def init_dataset(self):
        # No dataset to initialize for file-based loading
        self._filler_task = asyncio.create_task(self._buffer_filler())

    async def _text_generator(self):
        async for chunk in self._load_file_in_chunks(self.file_path, self.block_size):
            yield chunk