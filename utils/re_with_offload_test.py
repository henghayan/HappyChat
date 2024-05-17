import time
import gc
import torch
import transformers
from offload_manager import OffloadManager
from mem_optimize import model_to_recompute_mode
from data_loader import load_train_data
from transformers import Adafactor

import sys
sys.path.append("/data/HappyChat")
print("sys.path", sys.path)


def train_with_re_offload(
        base_model: str = "", data_path: str = "", output_dir: str = "", c_8bit=False, lora=False, device="cuda:0",
        batch_size=256, micro_batch_size=128, num_epochs=8, learning_rate=0.00003, cutoff_len=512, gui=False, save=True
):

    gradient_accumulation_steps = int(batch_size) // int(micro_batch_size)
    print("start train")
    # tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token_id = tokenizer.pad_token_id
    tokenizer.bos_token_id = tokenizer.bos_token_id
    tokenizer.eos_token_id = tokenizer.eos_token_id

    tokenizer.pad_token = tokenizer.eos_token

    train_data = load_train_data(data_path, tokenizer, int(cutoff_len))
    print("tokenizer load ok, mode load ...")

    dtype = torch.bfloat16
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto"
    )

    # offload_mgr = OffloadManager(model, 3)
    model_to_recompute_mode(model, None)
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=int(micro_batch_size),
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=int(num_epochs),
            learning_rate=float(learning_rate),
            output_dir=output_dir,
            # optim="adafactor"
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True),
    )

    start_time = time.time()

    trainer.train()
    end_time = time.time()
    print("train use time:", end_time - start_time)



if __name__ == '__main__':
    path = "/data2/llm3-8"
    data_path = "/data/HappyChat/train_data/vir.json"
    train_with_re_offload(path, data_path, "/data/output")

    # test_use_time_for_offload(path)