import torch
import time


def benchmark_bandwidth(size):
    # 创建CPU和GPU的张量
    cpu_tensor = torch.randn(size, pin_memory=True)
    gpu_tensor = torch.randn(size, device='cuda')

    # 测量CPU到GPU的传输时间
    start_time = time.time()
    gpu_tensor.copy_(cpu_tensor, non_blocking=True)
    torch.cuda.synchronize()
    end_time = time.time()
    cpu_to_gpu_time = end_time - start_time

    # 测量GPU到CPU的传输时间
    start_time = time.time()
    cpu_tensor.copy_(gpu_tensor, non_blocking=True)
    torch.cuda.synchronize()
    end_time = time.time()
    gpu_to_cpu_time = end_time - start_time

    # 计算带宽（GB/s）
    data_size_gb = cpu_tensor.numel() * cpu_tensor.element_size() / 1e9
    cpu_to_gpu_bandwidth = data_size_gb / cpu_to_gpu_time
    gpu_to_cpu_bandwidth = data_size_gb / gpu_to_cpu_time

    print(f"CPU to GPU bandwidth: {cpu_to_gpu_bandwidth:.2f} GB/s")
    print(f"GPU to CPU bandwidth: {gpu_to_cpu_bandwidth:.2f} GB/s")


def test_async_read_write(size):
    # 创建CPU和GPU的张量
    cpu_to_gpu_tensor = torch.randn(size, pin_memory=True)
    gpu_tensor = torch.randn(size, device='cuda')

    gpu_to_cpu_tensor = torch.empty(size, pin_memory=True)
    cpu_tensor = torch.randn(size, device='cuda')

    # 创建两个CUDA流
    stream1 = torch.cuda.Stream("cuda:0")
    stream2 = torch.cuda.Stream("cuda:0")

    # 测试异步读写性能
    start_time = time.time()

    # 异步CPU到GPU传输
    with torch.cuda.stream(stream1):
        gpu_tensor.copy_(cpu_to_gpu_tensor, non_blocking=True)

    # 异步GPU到CPU传输
    with torch.cuda.stream(stream2):
        gpu_to_cpu_tensor.copy_(cpu_tensor, non_blocking=True)

    # 等待两个流完成
    stream1.synchronize()
    stream2.synchronize()

    end_time = time.time()
    print(f'Async read/write time: {end_time - start_time:.4f} seconds')




def test_sync_read(size):
    # 创建CPU和GPU的张量
    cpu_to_gpu_tensor = torch.randn(size, pin_memory=True)
    gpu_tensor = torch.randn(size, device='cuda')

    start_time = time.time()
    gpu_tensor.copy_(cpu_to_gpu_tensor, non_blocking=False)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f'Sync CPU to GPU time: {end_time - start_time:.4f} seconds')

def test_sync_write(size):
    # 创建GPU和CPU的张量
    gpu_to_cpu_tensor = torch.empty(size, pin_memory=True)
    cpu_tensor = torch.randn(size, device='cuda')

    start_time = time.time()
    gpu_to_cpu_tensor.copy_(cpu_tensor, non_blocking=False)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f'Sync GPU to CPU time: {end_time - start_time:.4f} seconds')

# 测试样本大小
size = (1024 * 1024 * 1024)
#
# 运行单独的同步读写测试
test_sync_read(size)
test_sync_write(size)
# 运行异步读写测试
test_async_read_write(size)

# benchmark_bandwidth(size)