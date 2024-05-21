import torch
import time


# 函数：测试多 GPU 同步传输速度
def test_multi_gpu_sync_transfer(size):
    num_gpus = torch.cuda.device_count()
    tensors = [torch.randn(size, size).cuda(i) for i in range(num_gpus)]

    # 测试从多个 GPU 到 CPU 的同步传输时间
    start_time = time.time()
    tensors_cpu = [tensor.cpu() for tensor in tensors]
    multi_gpu_to_cpu_sync_time = time.time() - start_time

    # 测试从 CPU 到多个 GPU 的同步传输时间
    tensors_cpu = [torch.randn(size, size) for _ in range(num_gpus)]
    start_time = time.time()
    tensors_gpu = [tensor.cuda(i) for i, tensor in enumerate(tensors_cpu)]
    multi_cpu_to_gpu_sync_time = time.time() - start_time

    return multi_gpu_to_cpu_sync_time, multi_cpu_to_gpu_sync_time


# 函数：测试多 GPU 异步传输速度
def test_multi_gpu_async_transfer(size):
    num_gpus = torch.cuda.device_count()
    tensors = [torch.randn(size, size).cuda(i) for i in range(num_gpus)]

    # 测试从多个 GPU 到 CPU 的异步传输时间
    streams = [torch.cuda.Stream(device=i) for i in range(num_gpus)]
    start_time = time.time()
    for i, tensor in enumerate(tensors):
        with torch.cuda.stream(streams[i]):
            tensor.data = tensor.to("cpu", non_blocking=True)
    torch.cuda.synchronize()
    multi_gpu_to_cpu_async_time = time.time() - start_time

    # 测试从 CPU 到多个 GPU 的异步传输时间
    tensors_cpu = [torch.randn(size, size) for _ in range(num_gpus)]
    streams = [torch.cuda.Stream(device=i) for i in range(num_gpus)]
    start_time = time.time()
    for i, tensor in enumerate(tensors_cpu):
        with torch.cuda.stream(streams[i]):
            tensor.cuda(i, non_blocking=True)

    torch.cuda.synchronize()
    multi_cpu_to_gpu_async_time = time.time() - start_time

    return multi_gpu_to_cpu_async_time, multi_cpu_to_gpu_async_time


# 设定测试的张量大小
tensor_size = 2**14

# 多个 GPU 的同步传输速度测试
if torch.cuda.device_count() > 1:
    multi_gpu_to_cpu_sync_time, multi_cpu_to_gpu_sync_time = test_multi_gpu_sync_transfer(tensor_size)
    print(f"Multi-GPU to CPU sync transfer time: {multi_gpu_to_cpu_sync_time:.6f} seconds")
    print(f"Multi-CPU to GPU sync transfer time: {multi_cpu_to_gpu_sync_time:.6f} seconds")

    # 多个 GPU 的异步传输速度测试
    multi_gpu_to_cpu_async_time, multi_cpu_to_gpu_async_time = test_multi_gpu_async_transfer(tensor_size)
    print(f"Multi-GPU to CPU async transfer time: {multi_gpu_to_cpu_async_time:.6f} seconds")
    print(f"Multi-CPU to GPU async transfer time: {multi_cpu_to_gpu_async_time:.6f} seconds")
else:
    print("Only one GPU is available. Multi-GPU transfer test is skipped.")
