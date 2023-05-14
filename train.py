import sys
import transformers
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

from loader import load_model


def train(
        base_model: str = "",
        data_path: str = "",
        output_dir: str = "",
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
):
    gradient_accumulation_steps = batch_size // micro_batch_size

    model = load_model(base_model, torch_dtype=torch.float16)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None)
        if result["input_ids"][-1] != tokenizer.eos_token_id and len(
                result["input_ids"]) < cutoff_len and add_eos_token:
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = data_point["input"] + " " + data_point["output"]
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    data = load_dataset(data_path)
    train_data = data["train"].map(generate_and_tokenize_prompt)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            output_dir=output_dir,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True),
    )

    trainer.train()
    model.save_pretrained(output_dir)


def parse_args():
    args = sys.argv[1:]  # 忽略脚本名称
    params = {}
    for arg in args:
        key, value = arg.split("=")
        params[key] = value
    return params


if __name__ == "__main__":
    args = parse_args()
    train(**args)
