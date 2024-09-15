from ..adapters.models.nanogpt import GPT
from ..adapters.model_config import TransformerModelConfig

# Previous code

tokenizer = CharacterLevelTokenizer()


# Tell Panthalia how we're configuring the model
# and how to tokenize
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

