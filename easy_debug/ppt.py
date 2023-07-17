import torch
from torch import nn
import time
from fairscale.nn import Pipe
import os
import torch.distributed.pipeline.sync as ppsync
from utils.mem_optimize import model_to_recompute_mode, RecomputeLinearFunction, TimeLinear

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '29500'
# torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)


class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        # self.layers = nn.ModuleList([TimeLinear(4096 * 2, 4096 * 2, i, c) for i in range(16)])
        self.layers = nn.ModuleList([nn.Linear(4096*2, 4096*2) for i in range(12)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


multi_gpu = 1


class tempTM1(nn.Module):
    def __init__(self, ly1, ly2):
        super().__init__()
        self.fc1 = ly1
        self.fc2 = ly2

    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x.to(f"cuda:{multi_gpu}"))


# 定义设备和模型切片

# model_to_recompute_mode(model)

# 训练模型
if __name__ == "__main__":
    devices = ["cuda:0", "cuda:1"]
    layer1 = Layer(0).to('cuda:0')
    layer2 = Layer(multi_gpu).to(f'cuda:{multi_gpu}')

    model = nn.Sequential(layer1, layer2)
    # a = list(model.named_children())
    print("a")
    model = tempTM1(layer1, layer2)
    # model = Pipe(model, chunks=1, balance=[1, 1], devices=devices)

    # model = ppsync.Pipe(model, chunks=4)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    temp_i = 1
    inputs = torch.randn(int(16/temp_i), 512, 4096 * 2).to('cuda:0')
    target = torch.randn(int(16/temp_i), 512, 4096 * 2).to(f'cuda:{multi_gpu}')
    inputs.requires_grad = True
    # outputs = model(inputs)

    start_time = time.time()

    for j in range(temp_i):
        for i in range(10):
            outputs = model(inputs)
            # optimizer.zero_grad()
            loss = criterion(outputs, target)
            # loss = criterion(outputs.to_here(), target)  # Compute the loss
            # loss.backward()
            loss.backward(retain_graph=True)
            # torch.cuda.synchronize()
            # print("loss", loss)
            # optimizer.step()
            print("i", i)
    print("Pipeline parallel training time: ", time.time() - start_time)

