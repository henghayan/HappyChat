import torch
import time
import gc

# Set tensor size
tensor_size = (256, 4096, 4096)
# Adjust as needed

stream = torch.cuda.Stream("cuda:0")


def transfer_to_cpu(buffer, tensor):
    with torch.cuda.stream(stream):
        buffer.copy_(tensor, non_blocking=False)
    stream.synchronize()


# Allocate pinned memory for the tensor on CPU
def get_tensor_cpu_pinned(cuda_tensor, dtype=torch.bfloat16):
    res = torch.empty(cuda_tensor.size(), pin_memory=True)
    return res


def test_list_transformer_pin(item_num, dtype=torch.bfloat16):
    gc.collect()
    torch.cuda.empty_cache()
    test_list = [torch.randn(tensor_size, device='cuda', dtype=dtype) for i in range(item_num)]

    start_list = time.time()
    pin_list = []
    for t in test_list:
        pin = get_tensor_cpu_pinned(t)
        pin_list.append(pin)

    pin_time = time.time()
    print(f'[{item_num}] pin use: {pin_time - start_list:.5f} seconds')

    for i in range(len(test_list)):
        transfer_to_cpu(pin_list[i], test_list[i])

    total = time.time() - pin_time
    print(f'[{item_num}] test_list GPU to CPU transfer time with pinned memory: {total:.5f} seconds\n')


def test_list_transformer_native(item_num, dtype=torch.bfloat16):
    test_list = [torch.randn(tensor_size, device='cuda', dtype=dtype) for i in range(item_num)]

    start_time = time.time()

    for i in range(len(test_list)):
        param = test_list[i]
        with torch.cuda.stream(stream):
            param = param.to("cpu", non_blocking=False)
        stream.synchronize()
    total = time.time() - start_time
    print(f'[native] [{item_num}]:test_list GPU to CPU transfer time: {total:.5f} seconds\n')

# Verify that the data is transferred correctly
# assert torch.allclose(tensor_cpu_pinned, tensor_cpu_sync)
# print("Data verification successful.")

if __name__ == '__main__':
    # test_list_transformer_pin(1)
    #
    # test_list_transformer_pin(2)
    #
    # test_list_transformer_pin(4)
    #
    # test_list_transformer_pin(8)
    #
    # test_list_transformer_pin(16)
    #
    # test_list_transformer_pin(32)
    #
    # test_list_transformer_pin(64)
    #
    # test_list_transformer_pin(128)
    #
    # test_list_transformer_pin(256)

    test_list_transformer_native(1)
    test_list_transformer_pin(1)