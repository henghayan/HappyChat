import sys
import transformers
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

from loader import load_model
from prompter import Prompter
from compression import compress_module, decompress_module


def train(
        base_model: str = "", data_path: str = "", output_dir: str = "", c_8bit=False, device="cuda",
        batch_size=32, micro_batch_size=1, num_epochs=3, learning_rate=3e-3, cutoff_len=128,
):
    gradient_accumulation_steps = int(batch_size) // int(micro_batch_size)
    print("start train")
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
    train_data = load_train_data(data_path, tokenizer, int(cutoff_len))
    print("tokenizer load ok, mode load ...")
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # todo 应该读取config.json 中的类型，目前写死
    dtype = torch.float16
    model = load_model(base_model, torch_dtype=dtype)

    print("model load ok")
    if c_8bit:
        compress_module(model, device)
    print("model compress ok")
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=int(micro_batch_size),
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=int(num_epochs),
            learning_rate=float(learning_rate),
            #fp16=True,
            output_dir=output_dir
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True),
    )

    trainer.train()
    if c_8bit:
        decompress_module(model, dtype)
    model.save_pretrained(output_dir)


def data_format_func(tokenizer, cutoff_len=256, add_eos_token=True):
    prompter = Prompter()

    def data_format(data_point):
        full_prompt = prompter.generate_prompt(data_point["input"], data_point["output"])
        result = tokenizer(full_prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None)
        if result["input_ids"][-1] != tokenizer.eos_token_id and len(
                result["input_ids"]) < cutoff_len and add_eos_token:
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    return data_format


def load_train_data(data_path, tokenizer, cutoff_len=256):
    data = load_dataset('json', data_files=data_path)
    train_data = data["train"].map(data_format_func(tokenizer, cutoff_len))
    return train_data


def parse_args():
    args = sys.argv[1:]
    params = {}
    for arg in args:
        key, value = arg.split("=")
        params[key] = value
    return params


if __name__ == "__main__":
    args = parse_args()
    train(**args)
