import os
import sys
import gc
import time

import transformers
from transformers import Trainer, TrainingArguments
import torch

import sys
sys.path.append("../")
print("sys.path", sys.path)
from data_loader import load_train_data


# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
# os.environ["RANK"] = "0"
# os.environ["LOCAL_RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"
# os.environ["NUM_PROCESSES"] = "3"

# os.environ["DS_SKIP_CUDA_CHECK"] = "1"
# os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'


def train(base_model: str = "", data_path: str = "", output_dir: str = "", c_8bit=False, lora=False, device="cuda:0",
          batch_size=32, micro_batch_size=8, num_epochs=2, learning_rate=0.0003, cutoff_len=512, gui=False, save=True):
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token_id = tokenizer.pad_token_id
    tokenizer.bos_token_id = tokenizer.bos_token_id
    tokenizer.eos_token_id = tokenizer.eos_token_id

    tokenizer.pad_token = tokenizer.eos_token

    # tokenizer.pad_token_id = 0

    # tokenizer.padding_side = "right"

    train_data = load_train_data(data_path, tokenizer, int(cutoff_len))

    dtype = torch.bfloat16
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        # device_map="auto"
    )

    training_args = TrainingArguments(
        deepspeed="/data/HappyChat/deepspeed_func/config.json",
        output_dir=output_dir,
        gradient_accumulation_steps=8,
        num_train_epochs=int(num_epochs),
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True)
    )
    trainer.train()


if __name__ == "__main__":
    path = "/data2/llm3-8"
    data_path = "/data/HappyChat/train_data/vir.json"
    train(path, data_path, "/data/output", c_8bit=True)
