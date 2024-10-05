import sys
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from modeling_llama_gritlm import (
    LlamaGritConfig, 
    LlamaGritModel, 
    LlamaForCausalLMGrit
)

AutoConfig.register("llama_grit", LlamaGritConfig)
AutoModel.register(LlamaGritConfig, LlamaGritModel)
AutoModelForCausalLM.register(
    LlamaGritConfig, LlamaForCausalLMGrit)

model = AutoModelForCausalLM.from_pretrained(
    sys.argv[1],
    torch_dtype="auto",
)
output_path = sys.argv[2]
model.save_pretrained(
    output_path,
    max_shard_size="5GB",
    safe_serialization=False,
)
