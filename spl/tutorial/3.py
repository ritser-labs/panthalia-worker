from ..adapters.dataloader import LanguageDataLoader, datasets_dir


# Previous code


# Defining the data loader
# This defines the training data we'll use
class ShakespeareDataLoader(LanguageDataLoader):
    def __init__(
            self,
            model_config: TransformerModelConfig,
            buffer_size, max_seq_len,
            file_path=os.path.join(
                datasets_dir, 'shakespeare.txt'),
            block_size=124000
        ):
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
