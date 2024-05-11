import colossalai
from colossalai.nn.optimizer import HybridAdam
# from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, HybridParallelPlugin, Plugin
from colossalai.lazy import LazyInitContext
from colossalai.zero.low_level import LowLevelZeroOptimizer

import time

import transformers
from transformers import Trainer, TrainingArguments
import torch
from loss_func import LabelSmoother

import sys

sys.path.append("/data/HappyChat")
print("sys.path", sys.path)
from data_loader import load_train_data


def train(base_model: str = "", data_path: str = "", output_dir: str = "", c_8bit=False, lora=False, device="cuda:0",
          batch_size=32, micro_batch_size=8, num_epochs=1, learning_rate=0.0003, cutoff_len=512, gui=False, save=True):
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token_id = tokenizer.pad_token_id
    tokenizer.bos_token_id = tokenizer.bos_token_id
    tokenizer.eos_token_id = tokenizer.eos_token_id

    tokenizer.pad_token = tokenizer.eos_token

    train_data = load_train_data(data_path, tokenizer, int(cutoff_len))

    dtype = torch.bfloat16

    # with LazyInitContext(default_device=torch.device('cuda')):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        # device_map="auto"
    )

    criterion = LabelSmoother()
    optimizer = HybridAdam(model.parameters(), lr=1e-3, nvme_offload_fraction=1.0, nvme_offload_dir='/swp/cs')
    plugin = LowLevelZeroPlugin(stage=2)
    booster = Booster(plugin=plugin)
    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

    torch.cuda.synchronize()
    model.train()

    start = time.time()
    # for step in range(3):
    #
    #     data = train_data[:10]
    #     outputs = model(**data)
    #     loss = criterion(outputs.logits, data['input_ids'])
    #     booster.backward(loss, optimizer)
    #     optimizer.step()
    #     optimizer.zero_grad()
    #     print(f'[{step}] loss: {loss.item():.3f}')
    # print(f'Time: {time.time() - start:.3f} s')

    return


if __name__ == "__main__":
    colossalai.launch(0, 1, "localhost", 8080)
    path = "/data2/llm3-8"
    data_path = "/data/HappyChat/train_data/vir.json"
    train(path, data_path, "/data/output", c_8bit=True)
