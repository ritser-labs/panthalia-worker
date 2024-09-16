from .adapters.dataloader import LanguageDataLoader, datasets_dir
from .adapters.model_config import TransformerModelConfig
from .adapters.models.nanogpt import GPT, GPTConfig
from .adapters.model_adapter import TransformerModelAdapter
from .adapters.plugins import StandardPlugin
import math
import os

class CharacterLevelTokenizer:
    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id
        
    def encode(self, text: str) -> list:
        return [ord(char) for char in text]
    
    def decode(self, ascii_list: list) -> str:
        return ''.join([chr(value) for value in ascii_list])

    def get_vocab_size(self) -> int:
        return 128

tokenizer = CharacterLevelTokenizer()

model_params = GPTConfig(
    block_size=256,
    vocab_size=tokenizer.get_vocab_size(),
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.2,
    bias=False,
    pad_token_id=tokenizer.pad_id
)

class NanoGPTConfig(TransformerModelConfig):
    def __init__(self, tokenizer, params):
        self.tokenizer = tokenizer
        self.args = [params]
        self.kwargs = {}
        self.model_class = GPT
        self.vocab_size = params.vocab_size
        self.max_seq_len = params.block_size
    
    def encode(text):
        return tokenizer.encode(text)
    
    def decode(tokens):
        return tokenizer.decode(tokens)

class NanoGPTModelAdapter(TransformerModelAdapter):
    def forward_and_loss(self, model, inputs, targets):
        return model.forward(inputs, targets)[1]
    
    def forward(self, model, inputs):
        return model.forward(inputs)[0]

    def generate(self, model, input, max_new_tokens=None, temperature=0.8, top_k=200):
        if max_new_tokens is None:
            max_new_tokens = self.model_config.get_max_seq_len()
        return model.generate(input, max_new_tokens, temperature=temperature, top_k=top_k)

class ShakespeareDataLoader(LanguageDataLoader):
    def __init__(self, model_config: TransformerModelConfig, buffer_size, max_seq_len, file_path=os.path.join(datasets_dir, 'shakespeare.txt'), block_size=124000):
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

model_config = NanoGPTConfig(tokenizer, model_params)

model_adapter = NanoGPTModelAdapter(model_config)

dataset = ShakespeareDataLoader(model_config, buffer_size=100_000, max_seq_len=model_params.block_size)

NUM_MICROBATCHES = 200

EXAMPLES_PER_MICROBATCH = 512

exported_plugin = StandardPlugin(
    model_adapter,
    model_config,
    dataset,
    tokenizer,
    num_microbatches=NUM_MICROBATCHES,
    example_per_microbatch=EXAMPLES_PER_MICROBATCH,
    outer_max_lr=1,
    outer_min_lr=1,
    tensor_version_interval=45,
    expected_worker_time=40,
    max_concurrent_iterations=4,
    inner_max_lr=0.001,
    inner_min_lr=0.0001,
    inner_T_0=200
)