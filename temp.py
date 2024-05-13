import torch
import time

a = torch.rand([2**15, 2**15]).to("cuda")
b = torch.rand([2**15, 2**15]).to("cuda")

start = time.time()

c = torch.nn.functional.linear(a, b)

print("use time", time.time()-start)