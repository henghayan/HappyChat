import torch
import torch.nn as nn
import time


class GpuModule(nn.Module):
    def __init__(self, n_layers, input_dim, output_dim, device):
        super(GpuModule, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim).to(device) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = GpuModule(128, 2048, 2048, device="cuda:0")
        self.fc2 = GpuModule(128, 2048, 2048, device="cuda:1")

    def forward(self, x, num_splits):
        # 按照指定的段数将输入数据分割
        splits = torch.split(x, x.size(0) // num_splits)

        outputs = []
        data0 = splits[0].to("cuda:0")
        data1 = splits[0].to("cuda:1")

        for split in splits:
            # split = split.to("cuda:0")
            # y = self.fc1(split)
            # y = self.fc2(y.to("cuda:1"))
            # outputs.append(y)

            y = self.fc1(data0)
            y = self.fc2(data1)
            # outputs.append(y)

        return outputs


def train():
    # 假设我们有一些输入数据
    input_data = torch.randn(256, 2048)

    model = MyModel()
    start_time = time.time()

    for i in range(100):
        # 执行模型并交错处理数据
        output_data = model(input_data, 4)  # 将输入数据分割成4个部分
        print("i", i)
    end_time = time.time()

    print("Elapsed time: ", end_time - start_time)


if __name__ == "__main__":
    train()