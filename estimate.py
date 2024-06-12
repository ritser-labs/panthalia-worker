from transformers import AutoModel
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from configuration_gemmoe import GemmoeConfig
from modeling_gemmoe import GemmoeForCausalLM
config = GemmoeConfig().from_json_file("config.json")
print(config.num_experts_per_tok)
model = GemmoeForCausalLM(config)
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)
