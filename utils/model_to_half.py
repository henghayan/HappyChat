import torch
import transformers
from transformers import AutoTokenizer


def get_half_model(model_path, save_path, dtype=torch.bfloat16):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model.half()
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


if __name__ == '__main__':
    get_half_model('/data2/llm3-70', '/data2/llm3-70-half')
