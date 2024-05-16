import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from offload_manager import OffloadManager
from mem_optimize import model_to_recompute_mode
import gc

# 定义线性层模型
class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, devices, dtype=torch.bfloat16):
        super(LinearModel, self).__init__()
        self.devices = devices
        self.num_devices = len(devices)
        self.num_layers = num_layers
        self.dtype = dtype

        self.layers = nn.ModuleList()
        self.layers_per_device = num_layers // self.num_devices
        extra_layers = num_layers % self.num_devices

        for i in range(num_layers):
            temp_index = i // self.layers_per_device
            device = self.devices[temp_index if temp_index < self.num_devices else -1]
            layer = nn.Linear(input_dim, input_dim, bias=False, dtype=self.dtype).to(device)
            self.layers.append(layer)

        self.prelu = nn.PReLU(dtype=dtype).to(devices[-1], dtype=self.dtype)
        self.fc = nn.Linear(input_dim, output_dim, dtype=dtype, bias=False).to(devices[-1], dtype=self.dtype)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            temp_index = i // self.layers_per_device
            device = self.devices[temp_index if temp_index < self.num_devices else -1]
            x = x.to(device)
            x = layer(x)

        x = self.prelu(x)
        x = self.fc(x)
        return x

# 生成一些示例数据
input_dim = 2**12
output_dim = 2**12
num_layers = 256
batch_size = 32
data_size = 1024

# 创建数据集
inputs = torch.randn(data_size, input_dim)
targets = torch.randn(data_size, output_dim)
dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 获取可用的 GPU 设备
devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
if not devices:
    devices = [torch.device('cpu')]


use_dtype = torch.bfloat16

# 初始化模型
model = LinearModel(input_dim, output_dim, num_layers, devices, dtype=use_dtype)
offload_mgr = OffloadManager(model, 3)
model_to_recompute_mode(model, offload_mgr)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

torch.cuda.synchronize()
gc.collect()
torch.cuda.empty_cache()

time_start = time.time()
num_epochs = 1
for epoch in range(num_epochs):
    model.train()

    inputs = torch.randn(batch_size, 512, input_dim, dtype=use_dtype).to(devices[0])
    labels = torch.randint(0, 512, (batch_size, output_dim), dtype=torch.long).to(devices[-1])
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, use_time: {time.time() - time_start:.2f} seconds')
print(f'训练完成！Total time: {time.time() - time_start:.2f} seconds')
