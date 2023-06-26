import time
import gc

import torch
import torch.optim as optim
from torch.utils.checkpoint import checkpoint

from utils.mem_optimize import model_to_recompute_mode, RecomputeLinear
from utils.compression import *


class SimpleLinearModel(nn.Module):

    def __init__(self):
        super(SimpleLinearModel, self).__init__()

        self.layers = nn.ModuleList([TempLinear(4096, 4096) for _ in range(32)])

        self.fc = nn.Linear(4096, 512)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.fc(x)
        return x


class TempLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super(TempLinear, self).__init__()
        self.fc = torch.nn.Linear(in_f, out_f)

    def forward(self, x):
        return self.fc(x)


#################################################################################################################
#################################################################################################################



# 参数设置
d_model = 4096
num_layers = 16
n_epochs = 1
print_interval = 10


# 实例化模型


# 训练模型
def train():
    model = SimpleLinearModel()
    model.to("cuda:0")
    # model = TransformerTest(d_model, num_heads, num_layers, 512, dtype=dtype)
    criterion = nn.MSELoss()  # Loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    model_to_recompute_mode(model)

    input_seq_tensor = torch.rand(8192, 2, 4096, dtype=torch.float32).to("cuda:0")
    input_seq_tensor.requires_grad = True
    target_seq_tensor = torch.randn(8192, 2, 512, dtype=torch.float32).to("cuda:0")


    start_time = time.time()


    for epoch in range(n_epochs):
        step = 0
        for _ in range (15):
            # input_seq_tensor = torch.tensor(input_seq).unsqueeze(0)
            # target_seq_tensor = torch.tensor(target_seq).unsqueeze(0)

            # optimizer.zero_grad()
            output = model(input_seq_tensor)
            # res = output.squeeze(0)
            # target = target_seq_tensor.squeeze(0)
            loss = criterion(output, target_seq_tensor)
            loss.backward()
            optimizer.step()

            step += 1

            print(f"Epoch {epoch + 1}/{n_epochs}, Step {step + 1}")
            print("use_time1", time.time() - start_time)


if __name__ == "__main__":

    train()



