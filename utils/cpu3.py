import torch
import time

# Set tensor size
tensor_size = (4096, 4096)  # Adjust as needed

# Allocate pinned memory for the tensor on CPU


# Allocate a stream for asynchronous operations
stream = torch.cuda.Stream()


# Function to transfer data from GPU to pre-allocated CPU buffer
def transfer_to_cpu(buffer, tensor):
    with torch.cuda.stream(stream):
        buffer.copy_(tensor, non_blocking=True)
    stream.synchronize()


def transfer_to_gpu(t_gpu):
    # Synchronous transfer from GPU to CPU for comparison
    start = time.time()
    with torch.cuda.stream(stream):
        tensor_cpu_sync = t_gpu.to('cpu', non_blocking=False)
    stream.synchronize()
    sync_time = time.time() - start
    print(f'Synchronous GPU to CPU transfer time: {sync_time:.5f} seconds')


# Asynchronous transfer from GPU to pre-allocated CPU buffer
def transfer_pin(item_num, dtype=torch.bfloat16):
    test_list = [torch.randn(tensor_size, device='cuda', dtype=dtype) for i in range(item_num)]
    t_pin_list = [torch.empty(tensor_size, pin_memory=True, dtype=torch.bfloat16) for i in range(item_num)]
    start = time.time()
    for i in range(len(test_list)):
        t_new = test_list[i]
        t_pin_new = t_pin_list[i]
        transfer_to_cpu(t_pin_new, t_new)
        async_time = time.time() - start
    print(f'Asynchronous GPU to CPU transfer time with pinned memory: {async_time:.5f} seconds')


def test_async_w_r(size):
    w_tensor = torch.randn(size, device='cuda', dtype=torch.bfloat16)
    w_pin = torch.empty(size, pin_memory=True, dtype=torch.bfloat16)

    r_tensor = torch.randn(size, device='cpu', dtype=torch.bfloat16)
    stream = torch.cuda.Stream("cuda:0")
    stream2 = torch.cuda.Stream("cuda:0")

    start_time = time.time()
    with torch.cuda.stream(stream):
        w_pin.copy_(w_tensor, non_blocking=True)

    with torch.cuda.stream(stream2):
        r_tensor.to("cuda:0", non_blocking=True)

    torch.cuda.synchronize()
    end_time = time.time()
    print('end_time:', end_time - start_time)


def test_sync_w_r(size):
    w_tensor = torch.randn(size, device='cuda', dtype=torch.bfloat16)
    w_pin = torch.empty(size, pin_memory=True, dtype=torch.bfloat16)

    r_tensor = torch.randn(size, device='cpu', dtype=torch.bfloat16)
    stream = torch.cuda.Stream("cuda:0")

    start_time = time.time()
    with torch.cuda.stream(stream):
        w_pin.copy_(w_tensor, non_blocking=False)
    torch.cuda.synchronize()
    w_end_time = time.time()
    print('w_pin:', w_end_time - start_time)

    stream2 = torch.cuda.Stream("cuda:0")
    with torch.cuda.stream(stream2):
        r_tensor.to("cuda:0", non_blocking=False)
    torch.cuda.synchronize()
    r_end_time = time.time()
    print('r_end_time:', r_end_time - w_end_time)
    print("total:", r_end_time - start_time)


def test_async_w_r_multi_gpu(size):
    w_tensor = torch.randn(size, device='cuda', dtype=torch.bfloat16)
    w_pin = torch.empty(size, pin_memory=True, dtype=torch.bfloat16)

    r_tensor = torch.randn(size, device='cpu', dtype=torch.bfloat16)
    stream = torch.cuda.Stream("cuda:0")
    stream2 = torch.cuda.Stream("cuda:1")

    start_time = time.time()
    with torch.cuda.stream(stream):
        w_pin.copy_(w_tensor, non_blocking=True)

    with torch.cuda.stream(stream2):
        r_tensor.to("cuda:1", non_blocking=True)

    torch.cuda.synchronize()
    end_time = time.time()
    print('end_time:', end_time - start_time)


# # Verify that the data is transferred correctly
# assert torch.allclose(tensor_cpu_pinned, tensor_cpu_sync)
# print("Data verification successful.")

if __name__ == '__main__':

    # tensor_gpu = torch.randn(tensor_size, device='cuda', dtype=torch.bfloat16)
    # transfer_to_gpu(tensor_gpu)
    #
    # tensor_gpu_pin = torch.randn(tensor_size, device='cuda', dtype=torch.bfloat16)
    # tensor_cpu_pinned = torch.empty(tensor_size, pin_memory=True, dtype=torch.bfloat16)
    # transfer_pin(256)

    test_sync_w_r((128, 2048, 2048))
    print("-------------------\n")
    test_async_w_r((128, 2048, 2048))
    print("-------------------\n")
    test_async_w_r_multi_gpu((128, 2048, 2048))