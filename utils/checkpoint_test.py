import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(100000, 10000)
        self.linear2 = nn.Linear(10000, 1000)
        self.linear3 = nn.Linear(1000, 100)
        self.linear4 = nn.Linear(100, 10)

    def forward(self, x):

        # x = checkpoint(self.linear1, x)
        # x = checkpoint(self.linear2, x)
        # x = checkpoint(self.linear3, x)  # 使用 checkpoint 只在 linear2 这一层
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x


model_checkpointed = SimpleModel().to(device)

input_tensor = Variable(torch.randn(1, 100000)).to(device)
input_tensor.requires_grad = True
output_checkpointed = model_checkpointed(input_tensor)
output_checkpointed.backward(torch.ones(1, 10).to(device))

print("Check model gradients:")
