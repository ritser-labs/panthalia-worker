from .adapters.plugins import StandardPlugin

# ... Previous code

# Now we configure Panthalia

model_config = NanoGPTConfig(tokenizer, model_params)

model_adapter = NanoGPTModelAdapter(model_config)

dataset = ShakespeareDataLoader(
    model_config,
    buffer_size=100_000,
    max_seq_len=model_params.block_size
)

NUM_MICROBATCHES = 450 # 256,512

EXAMPLES_PER_MICROBATCH = 32 # 32,512

# Defining hyperparemeters for both inner and outer optimizers
# of DiLoCo
exported_plugin = StandardPlugin(
    model_adapter,
    model_config,
    dataset,
    tokenizer,
    num_microbatches=NUM_MICROBATCHES,
    example_per_microbatch=EXAMPLES_PER_MICROBATCH,
    outer_max_lr=0.7,
    outer_min_lr=0.7,
    tensor_version_interval=60,
    expected_worker_time=55,
    max_concurrent_iterations=4,
    inner_max_lr=0.001,
    inner_min_lr=0.0001,
    inner_T_0=200
)
