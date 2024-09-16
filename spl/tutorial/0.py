from ..adapters.models.nanogpt import GPTConfig

VOCAB_SIZE = 128
PAD_TOKEN_ID = 0

# Configuring the model itself
model_params = GPTConfig(
    block_size=256,
    vocab_size=VOCAB_SIZE,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.2,
    bias=False,
    pad_token_id=PAD_TOKEN_ID
)

