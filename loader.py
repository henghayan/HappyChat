import torch
import transformers
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from typing import List

from compression import compress_module


def load_model(model_path, torch_dtype=torch.bfloat16, **kv):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        **kv
    )
    return model


def lora_model(model, lora_r: int = 8,
               lora_alpha: int = 16,
               lora_dropout: float = 0.05,
               lora_target_modules: List[str] = ["q_proj", "v_proj"]):
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    return model


def compress_8bit(model, device="cuda:0"):
    compress_module(model, device)
    return


if __name__ == "__main__":
    pass