import os
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from torch.distributed.pipeline.sync import Pipe
import time

from utils.mem_optimize import model_to_recompute_mode, RecomputeLinear


class SimpleLinearModel(nn.Module):

    def __init__(self):
        super(SimpleLinearModel, self).__init__()

        self.layers = nn.ModuleList([TempLinear(4096, 4096) for _ in range(16)])

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


def run(rank, world_size):
    batch_size = 32
    input = torch.rand(batch_size, 2, 4096).to("cuda:0")
    input.requires_grad = True
    target = torch.rand(batch_size, 2, 512).to("cuda:0")

    model = SimpleLinearModel().to("cuda:0")

    criterion = nn.MSELoss()  # Loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Optimizer

    # Define the number of epochs and batch size.
    epochs = 1

    # model_to_recompute_mode(model)
    start_time = time.time()
    for epoch in range(epochs):
        for i in range(120):  # You can adjust this value according to the size of your dataset.
            optimizer.zero_grad()  # Reset the gradients

            output = model(input)
            loss = criterion(output, target)

            loss.backward()  # Backpropagation
            optimizer.step()  # Update the weights

            if i % 10 == 0:  # Print loss every 10 iterations.
                print(f"Epoch {epoch + 1}/{epochs}, Iteration {i + 1}/100")
    print(f"use time: {time.time() - start_time}")



if __name__ == "__main__":
    run(0, 1)
