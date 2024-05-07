from eetq import AutoEETQForCausalLM
from transformers import AutoTokenizer
import torch

model_name = "/data2/llm3-8"
quant_path = "/data/eetq_llm3_8"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# max_memory_mapping = {"cpu": "30GB", 0: "5GB", 1: "5GB", 2: "5GB"}
model = AutoEETQForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True,
                                            device_map="auto",
                                            # max_memory=max_memory_mapping
                                            )
model.quantize(quant_path)
tokenizer.save_pretrained(quant_path)
