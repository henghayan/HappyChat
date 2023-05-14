import sys

import transformers
from transformers import AutoTokenizer
import torch


def main(model_path):
    print("model_path", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("token ok")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.to(device='cuda:0')
    model.eval()
    input_text = "hello"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to('cuda:0')
    generated_text_ids = model.generate(input_ids, max_length=20)
    generated_text = tokenizer.decode(generated_text_ids[0], skip_special_tokens=True)
    print(generated_text)


if __name__ == "__main__":
    # Set model to evaluation mode
    main(sys.argv[1])
    # main("/D/hchat/mpt")