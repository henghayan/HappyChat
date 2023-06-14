import sys
import gc
import transformers

import torch

from model_loader import load_model
from data_loader import load_train_data
from utils.mem_optimize import model_to_recompute_mode
from chat_gui import GUI
import time


def train(
        base_model: str = "", data_path: str = "", output_dir: str = "", c_8bit=False, lora=False, device="cuda:0",
        batch_size=128, micro_batch_size=2, num_epochs=5, learning_rate=3e-3, cutoff_len=256, gui=False, save=True
):
    gradient_accumulation_steps = int(batch_size) // int(micro_batch_size)
    print("start train")
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
    train_data = load_train_data(data_path, tokenizer, int(cutoff_len))
    print("tokenizer load ok, mode load ...")
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # todo 应该读取config.json 中的类型，目前写死
    dtype = torch.bfloat16
    model = load_model(base_model, torch_dtype=dtype)
    # model.to(device)
    print("model load ok")
    # if lora:
    #     print("lora")
    #     model = lora_model(model)
    #
    # if c_8bit:
    #     print("compress 8bit")
    #     compress_module(model)

    if c_8bit:
        print("model_to_recompute_mode")
        model_to_recompute_mode(model)

    gc.collect()
    torch.cuda.empty_cache()
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
    # shield_layer(model, ["k", "q", "v"])

    start_time = time.time()
    # trainer.train()
    end_time = time.time()
    print("train use time:", end_time-start_time)
    # if c_8bit:
    #     decompress_module(model, dtype, device=device)
    if gui:
        model.eval()
        GUI(model, tokenizer, device)
    if save:
        model.save_pretrained(output_dir)


def shield_layer(model, layer_name_list=None):
    if layer_name_list is None:
        return
    params_layers = list(model.named_parameters())
    for layer in params_layers:
        pass








def parse_args():
    args = sys.argv[1:]
    params = {}
    for arg in args:
        key, value = arg.split("=")
        params[key] = value
    return params


if __name__ == "__main__":
    # print("wqer", transformers.__file__)
    args = parse_args()
    train(**args)
