import time

import torch
import torch.optim as optim
import torch.nn as nn
import transformers

from data_loader import load_train_data
from model_loader import load_model

from transformers import Trainer

'''
可行性：针对性 训练时可行的，将emd 以及全连接层展开，可以看到每个词在初始及结束阶段与隐藏层是隔离的，仅与对应下标参数有关；
     在逻辑向量世界，可能影响较小，因为在计算机世界，隐藏层的传递仅与内部向量有关，与第几个词衍生无关，
    
影响面：但是在最后它还是有影响的，因为在softmax时，每个词之间的正相关性，将以数值的形势直接坐落在最后的解码层，也就是，越大的数值，概率越高；

规划： 只要逻辑向量隐藏层世界是正常的，“道”就还在，下一步，需要研究，不同范畴知识分布区域，知识分布一定是连续或离散相关的；

疑惑： 全连接层是否具备知识记忆，我认为应该是没有的，它类似人的反射弧，不需要思考，坐落该层的一定是词性自身的概念；
      且  y = wa 的线性层，纯纯的加权，不可能有记忆；
'''


# 针对点token进行训练
class PointTokenTrainer(Trainer):
    def __init__(self, mask, *ars, **kwargs):
        super().__init__(*ars, **kwargs)
        self.mask = mask.unsqueeze(1)

    def training_step(self, model, inputs):
        # 调用原始的training_step
        outputs = super().training_step(model, inputs)

        with torch.no_grad():
            # a = list(model.named_parameters())
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'embed' in name or 'fc' in name or 'lm_head' in name:
                        temp_mask = self.mask.to(param.grad.device)
                        param.grad *= temp_mask

        return outputs


# 针对性训练token
def train_batch_point_token(train_data, model, token_mask):
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.CrossEntropyLoss()
    # Step 3: 在训练循环中应用Mask
    for epoch in range(10):  # 这里只是一个示例
        optimizer.zero_grad()
        for train_data in train_data['train']:
            input_ids = train_data['input_ids']
            outputs = model(input_ids)
            logits = outputs.logits
            loss = criterion(logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                             input_ids[:, 1:].contiguous().view(-1))
            loss.backward()
            with torch.no_grad():
                model.embedding.weight.grad *= token_mask
                model.fc.weight.grad *= token_mask

            optimizer.step()


def get_token_mask(vocab_size, start_index, end_index):
    token_mask = torch.zeros(vocab_size)
    token_mask[start_index:end_index] = 1
    return token_mask


def get_train_data(tokenizer, train_file, max_length=56):
    return load_train_data(train_file, tokenizer, max_length)


def train_llama_point(data_path, base_model, output_dir, start_index, end_index=None, auto_grad=False, batch_size=16,
                      micro_batch_size=2, num_epochs=1, learning_rate=0.0003, cutoff_len=512):
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    train_data = load_train_data(data_path, tokenizer, int(cutoff_len))
    print("tokenizer load ok, mode load ...")

    dtype = torch.bfloat16
    model = load_model(base_model, torch_dtype=dtype)

    first_param = None
    last_param = None

    # 重新启用第一个和最后一个参数的梯度
    for param in model.parameters():
        param.requires_grad = auto_grad
        if first_param is None:
            first_param = param
        last_param = param

    first_param.requires_grad = True
    last_param.requires_grad = True

    token_mask = get_token_mask(first_param.shape[0], start_index, end_index)

    gradient_accumulation_steps = int(batch_size) // int(micro_batch_size)
    # trainer = Trainer(
    trainer = PointTokenTrainer(
        mask=token_mask,
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=int(micro_batch_size),
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=int(num_epochs),
            learning_rate=float(learning_rate),
            output_dir=output_dir,
            logging_dir="/log/HappyChat/",
            logging_steps=2,
            save_steps=1000
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True)
    )
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    print("train use time:", end_time - start_time)
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    train_llama_point(data_path="/data/train_data/wiki0.json",
                      base_model="/data/x_win_40706_cn/",
                      start_index=32000, output_dir="/data/x_win_40706_cn1/")


    a= "CUDA_VISIBLE_DEVICES=0,1"