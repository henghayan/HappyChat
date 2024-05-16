import torch
import time

tensor_size = (1024, 1024, 1024)  # Adjust as needed
tensor1 = torch.randn(tensor_size, pin_memory=True)
# Synchronous transfer
pipe = torch.cuda.Stream("cuda:0")
pipe1 = torch.cuda.Stream("cuda:0")
start = time.time()
with torch.cuda.stream(pipe):
    tensor_cpu1 = tensor1.to('cuda', non_blocking=False)


torch.cuda.synchronize()
sync_time = time.time() - start
print(f'CPU to GPU time: {sync_time:.5f} seconds')

# Asynchronous transfer
start1 = time.time()
with torch.cuda.stream(pipe1):
    tensor_cpu1 = tensor_cpu1.to('cpu', non_blocking=False)
torch.cuda.synchronize()
gpu_to_cpu_time = time.time() - start1
print(f'GPU to CPU transfer time: {gpu_to_cpu_time:.5f} seconds')
