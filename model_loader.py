import torch
import transformers
# from peft import (
#     LoraConfig,
#     get_peft_model,
#     get_peft_model_state_dict,
#     prepare_model_for_int8_training,
#     set_peft_model_state_dict,
# )
from typing import List

from utils.compression import compress_module



def load_model(model_path, torch_dtype=torch.float16, **kv):
    model_map = {0: [0, 1, 2, 3, 4, 5], 1: [6, 7, 8, 9, 10, 11]}
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
        # load_in_8bit=True,
        **kv
    )
    return model


# def lora_model(model, lora_r: int = 4,
#                lora_alpha: int = 8,
#                lora_dropout: float = 0.05,
#                lora_target_modules: List[str] = ["q_proj", "v_proj", 'gate_proj', 'down_proj', 'up_proj', 'k_proj', "o_proj1"]):
#                #  lora_target_modules: List[str] = []):
#     config = LoraConfig(
#         r=lora_r,
#         lora_alpha=lora_alpha,
#         target_modules=lora_target_modules,
#         lora_dropout=lora_dropout,
#         bias="none",
#         task_type="CAUSAL_LM",
#     )
#     model = get_peft_model(model, config)
#     return model


def compress_8bit(model):
    compress_module(model)
    return


if __name__ == "__main__":
    pass
