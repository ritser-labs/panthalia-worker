from ..adapters.models.nanogpt import GPT, GPTConfig
from ..adapters.model_config import TransformerModelConfig

# Previous code

tokenizer = CharacterLevelTokenizer()

# Configuring the model itself
model_params = GPTConfig(
    block_size=256,
    vocab_size=tokenizer.get_vocab_size(),
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.2,
    bias=True,
    pad_token_id=tokenizer.pad_id
)


# Configuring Panthalia's model adapter
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

