import os
import torch
import torch.nn as nn
import torch.distributed.pipeline.sync as pipe_sync
from torch.distributed.rpc import init_rpc
import time

class GpuModule(nn.Module):
    def __init__(self, n_layers, input_dim, output_dim, device):
        super(GpuModule, self).__init__()
        self.device = device
        self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim).to(device) for _ in range(n_layers)])

    def forward(self, x):
        start_time = time.time()
        for layer in self.layers:
            x = layer(x)
        end_time = time.time()
        print(f"Module on {self.device} started at {start_time} and ended at {end_time}")
        return x

def cat_module(input, module1, module2, device=0):
    output1 = module1(input)
    output1 = output1.to("cuda:"+str(device))
    return module2(output1)



if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)

    module1 = GpuModule(128, 4096, 4096, device=torch.device('cuda:0'))
    module2 = GpuModule(128, 4096, 4096, device=torch.device('cuda:1'))

    model = nn.Sequential(module1, module2)
    model = pipe_sync.Pipe(model, chunks=1)

    criterion = nn.MSELoss()  # Loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Optimizer

    total_samples = 256 * 100
    batch_size = 256  # You can adjust this value according to your GPU memory.

    start_time = time.time()
    for i in range(total_samples // batch_size):
        optimizer.zero_grad()  # Reset the gradients
        input = torch.rand(batch_size, 4096).cuda(0)  # Adjust the size of the input tensor
        target = torch.rand(batch_size, 4096).cuda(1)

        output_rref = model(input)# Generate a target tensor
        loss = criterion(output_rref.to_here(), target)  # Compute the loss

        # output_rref = cat_module(input, module1, module2, 1)# Generate a target tensor
        # loss = criterion(output_rref, target)  # Compute the loss

        loss.backward()  # Backpropagation
        optimizer.step()  # Update the weights
        print("output_rref", i)

    print("use time", time.time() - start_time)








def run(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    init_rpc(f"worker{rank}", rank=rank, world_size=world_size)
