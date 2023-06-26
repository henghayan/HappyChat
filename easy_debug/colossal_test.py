import time

import torch
import transformers
from torch.utils.data import DataLoader

from data_loader import load_train_data, collate_fn
from sample_transformer import TransformerTest
from chat_gui import GUI
from model_loader import load_model

from utils.mem_optimize import model_to_recompute_mode

from torch.utils.checkpoint import checkpoint


def colossal(model, optimizer, criterion):
    booster = Booster(plugin=plugin)
    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)
    return model, optimizer, criterion, booster


def create_model(d_model=1024, num_heads=12, num_layer=12, vocab_size=32000, dtype=torch.float16):
    model = TransformerTest(d_model, num_heads, num_layer, vocab_size, dtype=dtype)
    return model


def train(model, optimizer, data_path, num_epochs, device="cuda:0"):
    tokenizer = transformers.AutoTokenizer.from_pretrained('/data/fs7', trust_remote_code=True, use_fast=False)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    train_data = load_train_data(data_path, tokenizer, 512)
    train_dataloader = DataLoader(train_data, batch_size=2, collate_fn=collate_fn)
    criterion = CrossEntropyLoss()
    # train_dataloader = DataLoader(train_data, batch_size=2,
    #                               collate_fn=transformers.DataCollatorForSeq2Seq(tokenizer, return_tensors="pt",
    #                                                                              padding=True))

    # model, optimizer, criterion, booster = colossal(model, optimizer, criterion)
    step = 0
    for i in range(num_epochs):
        for data_batch in train_dataloader:

            input_ids = data_batch['input_ids'].to(device)
            attn_mask = data_batch['attention_mask'].to(device)
            labels = data_batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attn_mask)

            loss = criterion(outputs.logits, labels)
            torch.cuda.synchronize()

            loss.backward()
            # booster.backward(loss, optimizer)

            optimizer.step()

            if (step + 1) % 1 == 0:
                print(f"Epoch {i + 1}/{num_epochs}, Step {step + 1}, Loss: {loss.item()}")

            step += 1
    model.eval()
    print("model", model)
    GUI(model, tokenizer, device)


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        # 错位并 重塑
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        res_logits = shift_logits.view(-1, shift_logits.shape[-1])
        res_labels = shift_labels.view(-1)
        return self.criterion(res_logits, res_labels)


if __name__ == "__main__":
    dtype = torch.bfloat16
    model = load_model('/data/fs7', torch_dtype=dtype)

    # model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    model_to_recompute_mode(model)
    # model, optimizer, criterion = "", "", ""
    train(model, optimizer, "/data/HappyChat/train_data/test.json", 1)
