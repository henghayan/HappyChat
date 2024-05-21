import torch
import time


def test_multi_gpu_sync(gpu_num, size=(1024, 1024, 1024)):
    tensors_sync_list = [torch.randn(size) for i in range(gpu_num)]

    start = time.time()
    for i in range(gpu_num):
        tensors = tensors_sync_list[i]
        stream = torch.cuda.Stream("cuda:%s" % i)
        with torch.cuda.stream(stream):
            tensors.data = tensors.to("cuda:%s" % i, non_blocking=False)
    time_sync = time.time() - start
    torch.cuda.synchronize()
    print(f"Time sync CPU->GPU: {time_sync:.5f} seconds, GPU num:{gpu_num}")

    start = time.time()
    for i in range(gpu_num):
        tensors = tensors_sync_list[i]
        stream = torch.cuda.Stream("cuda:%s" % i)
        with torch.cuda.stream(stream):
            tensors.data = tensors.to("cpu", non_blocking=False)
    time_sync = time.time() - start
    torch.cuda.synchronize()
    print(f"Time sync CPU<-GPU: {time_sync:.5f} seconds, GPU num:{gpu_num}")


def test_multi_gpu_async(gpu_num, size=(1024, 1024, 1024)):
    tensors_async_list = [torch.randn(size) for i in range(gpu_num)]

    start_async = time.time()
    for i in range(gpu_num):
        tensors = tensors_async_list[i]
        stream = torch.cuda.Stream("cuda:%s" % i)
        with torch.cuda.stream(stream):
            tensors.data = tensors.to("cuda:%s" % i, non_blocking=True)
    torch.cuda.synchronize()
    time_sync = time.time() - start_async
    print(f"Time async CPU->GPU: {time_sync:.5f} seconds, GPU num:{gpu_num}")

    start_async = time.time()
    for i in range(gpu_num):
        tensors = tensors_async_list[i]
        stream = torch.cuda.Stream("cuda:%s" % i)
        with torch.cuda.stream(stream):
            tensors.data = tensors.to("cpu", non_blocking=True)
    torch.cuda.synchronize()
    time_sync = time.time() - start_async
    print(f"Time async CPU<-GPU: {time_sync:.5f} seconds, GPU num:{gpu_num}")


if __name__ == '__main__':
    # num_gpus = torch.cuda.device_count()
    num_gpus = 3
    test_multi_gpu_sync(num_gpus)
    test_multi_gpu_async(num_gpus)
