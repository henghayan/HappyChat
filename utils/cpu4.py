import torch
import time


def test_async_w_r(size):
    w_tensor = torch.randn(size, device='cuda', dtype=torch.float32)
    w_pin = torch.empty(size, pin_memory=True, dtype=torch.float32)
    r_tensor = torch.randn(size, device='cpu', dtype=torch.float32)

    stream_write = torch.cuda.Stream()
    stream_read = torch.cuda.Stream()

    start_time = time.time()

    with torch.cuda.stream(stream_write):
        w_pin.copy_(w_tensor, non_blocking=True)

    with torch.cuda.stream(stream_read):
        r_tensor = r_tensor.to("cuda", non_blocking=True)

    write_event = torch.cuda.Event()
    read_event = torch.cuda.Event()

    stream_write.record_event(write_event)
    stream_read.record_event(read_event)

    write_event.synchronize()
    read_event.synchronize()

    end_time = time.time()
    print('Async end_time:', end_time - start_time)


def test_sync_w_r(size):
    w_tensor = torch.randn(size, device='cuda', dtype=torch.float32)
    w_pin = torch.empty(size, pin_memory=True, dtype=torch.float32)
    r_tensor = torch.randn(size, device='cpu', dtype=torch.float32)

    stream = torch.cuda.Stream()

    start_time = time.time()

    with torch.cuda.stream(stream):
        w_pin.copy_(w_tensor, non_blocking=False)
    stream.synchronize()
    w_end_time = time.time()
    print('Sync w_pin:', w_end_time - start_time)

    with torch.cuda.stream(stream):
        r_tensor = r_tensor.to("cuda", non_blocking=False)
    stream.synchronize()
    r_end_time = time.time()
    print('Sync r_end_time:', r_end_time - w_end_time)
    print("Sync total:", r_end_time - start_time)


def test_async_w_r_multi_gpu(size):
    w_tensor = torch.randn(size, device='cuda:0', dtype=torch.float32)
    w_pin = torch.empty(size, pin_memory=True, dtype=torch.float32)
    r_tensor = torch.randn(size, device='cpu', dtype=torch.float32)

    stream_write = torch.cuda.Stream(device='cuda:0')
    stream_read = torch.cuda.Stream(device='cuda:1')

    start_time = time.time()

    with torch.cuda.stream(stream_write):
        w_pin.copy_(w_tensor, non_blocking=True)

    with torch.cuda.stream(stream_read):
        r_tensor = r_tensor.to("cuda:1", non_blocking=True)

    write_event = torch.cuda.Event()
    read_event = torch.cuda.Event()

    stream_write.record_event(write_event)
    stream_read.record_event(read_event)

    write_event.synchronize()
    read_event.synchronize()

    end_time = time.time()
    print('Multi-GPU Async end_time:', end_time - start_time)


def test_async_w_r_chunked(size, chunk_size):
    w_tensor = torch.randn(size, device='cuda', dtype=torch.float32)
    w_pin = torch.empty(size, pin_memory=True, dtype=torch.float32)
    r_tensor = torch.randn(size, device='cpu', dtype=torch.float32)

    stream_write = torch.cuda.Stream()
    stream_read = torch.cuda.Stream()

    start_time = time.time()

    num_chunks = size[0] // chunk_size
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size

        with torch.cuda.stream(stream_write):
            w_pin[start_idx:end_idx].copy_(w_tensor[start_idx:end_idx], non_blocking=True)

        with torch.cuda.stream(stream_read):
            r_tensor[start_idx:end_idx] = r_tensor[start_idx:end_idx].to("cuda", non_blocking=True)

    write_event = torch.cuda.Event()
    read_event = torch.cuda.Event()

    stream_write.record_event(write_event)
    stream_read.record_event(read_event)

    write_event.synchronize()
    read_event.synchronize()

    end_time = time.time()
    print('Chunked Async end_time:', end_time - start_time)


# 测试样本大小和块大小
size = (4096, 4096, 4096)  # 更大的数据集
chunk_size = 1024  # 块大小

print("Testing Async Write and Read")
test_async_w_r(size)

print("\nTesting Sync Write and Read")
test_sync_w_r(size)

print("\nTesting Multi-GPU Async Write and Read")
test_async_w_r_multi_gpu(size)

print("\nTesting Chunked Async Write and Read")
test_async_w_r_chunked(size, chunk_size)
