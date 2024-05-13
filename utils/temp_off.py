import torch
import torch.nn as nn
import torch.optim as optim
from offload_manager import OffloadManager
from mem_optimize import model_to_recompute_mode
import time
from torch.utils.checkpoint import checkpoint
import gc

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_memory_usage(description):
    torch.cuda.empty_cache()
    print(f"{description} - Allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")


hidden_num = 2 ** 14
dtype = None


# 简单的MLP模型
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(hidden_num, hidden_num, dtype=dtype)
        self.fc2 = nn.Linear(hidden_num, hidden_num, dtype=dtype)
        self.fc3 = nn.Linear(hidden_num, hidden_num, dtype=dtype)
        self.fc4 = nn.Linear(hidden_num, hidden_num, dtype=dtype)
        self.fc7 = nn.Linear(hidden_num, hidden_num, dtype=dtype)
        self.fc8 = nn.Linear(hidden_num, hidden_num, dtype=dtype)
        self.fc5 = nn.Linear(hidden_num, hidden_num, dtype=dtype)
        self.fc6 = nn.Linear(hidden_num, hidden_num, dtype=dtype)


        self.prelu = nn.PReLU(dtype=dtype)
        self.fc = nn.Linear(hidden_num, 2048, dtype=dtype)

    def forward(self, x):
        x1 = self.fc1(x)
        # x.data = torch.empty(0, dtype=x.dtype, device=x.device)
        # torch.cuda.synchronize()
        # temp_x = x.detach().clone()
        # del x
        # gc.collect()
        # torch.cuda.empty_cache()

        x2 = self.fc2(x1)
        # x1.data = torch.empty(0, dtype=x1.dtype, device=x1.device)
        # del x1
        # torch.cuda.synchronize()
        # gc.collect()
        # torch.cuda.empty_cache()
        # temp_x1 = x1.detach().clone()




        x3 = self.fc3(x2)
        # x2.data = torch.empty(0, dtype=x2.dtype, device=x2.device)
        # del x2
        # torch.cuda.synchronize()
        # # temp_x2 = x2.detach().clone()
        #
        # gc.collect()
        # torch.cuda.empty_cache()

        x4 = self.fc4(x3)
        # x3.data = torch.empty(0, dtype=x3.dtype, device=x3.device)
        # torch.cuda.synchronize()
        # temp_x3 = x2.detach().clone()
        # del x2
        # gc.collect()
        # torch.cuda.empty_cache()

        x5 = self.prelu(x4)


        x = self.fc5(x5)

        x = self.fc6(x)

        x = self.fc7(x)

        x = self.fc8(x)

        x6 = self.fc(x)
        # x5.data = torch.empty(0, dtype=x5.dtype, device=x5.device)
        # torch.cuda.synchronize()
        # temp_x5 = x5.detach().clone()
        # del x5
        # gc.collect()
        # torch.cuda.empty_cache()
        return x6


def forward_by_offload(forward_tensor, input):
    out = forward_tensor(input)
    return out


# 实例化模型并移到GPU
model = SimpleMLP().to("cuda:0")
offload_mgr = OffloadManager(model, 1)
model_to_recompute_mode(model, offload_mgr)


print_memory_usage("Param Use")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 32
inputs = torch.randn(batch_size, 512, hidden_num, dtype=dtype).to(device)
labels = torch.randint(0, 512, (batch_size, 2048), dtype=dtype).to(device)

# 训练模型
model.train()
print_memory_usage("Before training")

time_start = time.time()
for epoch in range(1):
    optimizer.zero_grad()
    print_memory_usage("After zero_grad")

    outputs = model(inputs)
    print_memory_usage("After forward pass")

    loss = criterion(outputs, labels)
    print_memory_usage("After loss computation")

    loss.backward()
    print_memory_usage("After backward pass")

    optimizer.step()
    print_memory_usage("After optimizer step")

print("Time train use: ", time.time() - time_start)

# 去除梯度后的显存使用
optimizer.zero_grad()
print_memory_usage("After clearing gradients")
