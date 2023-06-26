import os

import torch
import torch.nn as nn
import deepspeed

import time

if '/usr/local/cuda-11.7/bin:/root/anaconda3/envs/env1/bin' not in os.environ['PATH']:
    os.environ['PATH'] += ':/usr/local/cuda-11.7/bin:/root/anaconda3/envs/env1/bin'


# 定义一个简单的线性模型
class SimpleLinearModel(nn.Module):

    def __init__(self):
        super(SimpleLinearModel, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(2048, 2048) for _ in range(128)])

        self.fc = nn.Linear(2048, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


os.environ['RANK'] = "0"
os.environ['WORLD_SIZE'] = "2"
# 初始化模型
model = SimpleLinearModel()

config = {
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 8,
    "steps_per_print": 2000,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 0.001
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 0.001,
            "warmup_num_steps": 1000
        }
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    # "fp16": {
    #     "enabled": True
    # }
}

# 为了启用流水线并行处理，我们需要使用DeepSpeed的PipelineEngine
engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)

# engine = model.to("cuda:0")

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 你的训练数据和标签
# 这里只是一个例子，你需要根据你的需求进行修改
inputs = torch.randn(64, 2048).to('cuda:0')
labels = torch.randn(64, 1).to('cuda:0')

# 训练循环
start_time = time.time()
for epoch in range(100):
    # 正向传播
    outputs = engine(inputs)
    # 计算损失
    loss = loss_fn(outputs, labels)

    # loss.backward()
    # optimizer.step()

    engine.backward(loss)
    engine.step()


    print("i", epoch)
print("use_time:", time.time() - start_time)