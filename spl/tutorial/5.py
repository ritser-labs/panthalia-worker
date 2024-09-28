from .adapters.plugins import StandardPlugin

# ... Previous code

# Now we configure Panthalia

model_config = NanoGPTConfig(tokenizer, model_params)

model_adapter = NanoGPTModelAdapter(model_config)

NUM_STEPS = 285

EXAMPLES_PER_STEP = 64

dataset = ShakespeareDataLoader(
    model_config,
    buffer_size=100_000,
    batch_size=NUM_STEPS * EXAMPLES_PER_STEP,
    max_seq_len=model_params.block_size
)

exported_plugin = StandardPlugin(
    model_adapter,
    model_config,
    dataset,
    tokenizer,
    num_steps=NUM_STEPS,
    examples_per_step=EXAMPLES_PER_STEP,
    outer_max_lr=1,
    outer_min_lr=1,
    outer_weight_decay=0.0,
    tensor_version_interval=36,
    expected_worker_time=31,
    max_concurrent_iterations=4,
    inner_max_lr=0.001,
    inner_min_lr=0.0001,
    inner_T_0=200,
    preload_batch_count=4
)