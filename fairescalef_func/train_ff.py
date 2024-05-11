import os
import sys
import gc
import time
from torch.utils.data.dataloader import DataLoader
import transformers
from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
from fairscale.experimental.nn.offload import OffloadModel
import sys

from torch.nn.utils.rnn import pad_sequence

from torch import Tensor
sys.path.append("/data/HappyChat")
print("sys.path", sys.path)
from data_loader import load_train_data


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    input_ids = pad_sequence([Tensor(item) for item in input_ids], batch_first=True).long()
    attention_mask = pad_sequence([Tensor(item) for item in attention_mask], batch_first=True).long()
    labels = pad_sequence([Tensor(item) for item in labels], batch_first=True).long()

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def train(base_model: str = "", data_path: str = "", output_dir: str = "", c_8bit=False, lora=False, device="cuda:0",
          batch_size=32, micro_batch_size=8, num_epochs=1, learning_rate=0.0003, cutoff_len=512, gui=False, save=True):
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token_id = tokenizer.pad_token_id
    tokenizer.bos_token_id = tokenizer.bos_token_id
    tokenizer.eos_token_id = tokenizer.eos_token_id

    tokenizer.pad_token = tokenizer.eos_token
    train_data = load_train_data(data_path, tokenizer, int(cutoff_len))

    dtype = torch.bfloat16
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        # device_map="auto"
    )
    print(model)
    for name, p in model.named_parameters():
        print(name)
    # sequential_layers = nn.Sequential(
    #     model.model.embed_tokens
    # )
    # for l in model.model.layers:
    #     sequential_layers.append(l)
    # sequential_layers.append(model.model.norm)
    # sequential_layers.append(model.lm_head)
    #
    # offload_model = OffloadModel(
    #     model=sequential_layers,
    #     device=torch.device("cuda"),
    #     offload_device=torch.device("cpu"),
    #     num_slices=1,
    #     checkpoint_activation=True,
    #     num_microbatches=1,
    # )
    #
    # torch.cuda.set_device(0)
    # device = torch.device("cuda")
    #
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(offload_model.parameters(), lr=0.001)
    # train_dataloader = DataLoader(train_data, batch_size=8, collate_fn=collate_fn)
    # # To train 1 epoch.
    # offload_model.train()
    # for data_batch in train_dataloader:
    #     input_ids = data_batch['input_ids'].to(device)
    #     attn_mask = data_batch['attention_mask'].to(device)
    #     labels = data_batch['labels'].to(device)
    #
    #     batch_inputs, batch_outputs = input_ids.to("cuda"), labels.to("cuda")
    #     start = time.time_ns()
    #     optimizer.zero_grad()
    #     # inputs = batch_inputs.reshape(-1, num_inputs * num_inputs)
    #     with torch.cuda.amp.autocast():
    #         output = offload_model(batch_inputs)
    #         loss = criterion(output, target=batch_outputs)
    #         loss.backward()
    #         print("output", output)
    #     optimizer.step()




if __name__ == "__main__":
    path = "/data2/llm3-8"
    data_path = "/data/HappyChat/train_data/vir.json"
    train(path, data_path, "/data/output", c_8bit=True)
