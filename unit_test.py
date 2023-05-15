import sys

import transformers
from transformers import AutoTokenizer
import torch

from train import load_train_data


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


def test_load_json(path, tokenizer):
    load_train_data(path, tokenizer)


def test_train(base_model, data_path):
    print("start train")
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
    print("tokenizer load ok, mode load ...")
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        load_8bit=True
    )

    print("model load ok")
    print("model load ok")
    train_data = load_train_data(data_path, tokenizer, 128)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=int(1),
            gradient_accumulation_steps=2,
            num_train_epochs=int(1),
            learning_rate=float(0.01),
            fp16=True,
            output_dir="/hy-tmp/out"
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True),
    )

    trainer.train()


if __name__ == "__main__":
    # Set model to evaluation mode
    # main(sys.argv[1])
    # main("/D/hchat/mpt")
    token_path = "/hy-tmp/fs7b"
    data_path = '/hy-tmp/test.json'
    # print("token_path", token_path)
    # tokenizer = AutoTokenizer.from_pretrained(token_path, trust_remote_code=True, use_fast=False)
    # print("token ok")
    # test_load_json(data_path, tokenizer)
    test_train(token_path, data_path)
