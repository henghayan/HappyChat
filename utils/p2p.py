import torch
import time


def enable_p2p(gpu0, gpu1):
    if torch.cuda.device_count() < 2:
        print("Requires at least two GPUs to test P2P.")
        return False

    torch.cuda.set_device(gpu0)
    try:
        torch.cuda.nccl.ncclGetP2PAccess(gpu1)
        torch.cuda.set_device(gpu1)
        torch.cuda.nccl.ncclGetP2PAccess(gpu0)
        return True
    except RuntimeError as e:
        print(f"P2P not supported between GPU {gpu0} and GPU {gpu1}: {e}")
        return False


def test_p2p_transfer(size, gpu0, gpu1):
    tensor = torch.randn(size, size, device=f'cuda:{gpu0}')

    # P2P传输
    start_time = time.time()
    tensor_p2p = tensor.to(f'cuda:{gpu1}')
    p2p_time = time.time() - start_time

    return p2p_time


# 设定测试的张量大小和GPU编号
tensor_size = 1024
gpu0 = 0
gpu1 = 1

if enable_p2p(gpu0, gpu1):
    p2p_time = test_p2p_transfer(tensor_size, gpu0, gpu1)
    print(f"P2P transfer time from GPU {gpu0} to GPU {gpu1}: {p2p_time:.6f} seconds")
else:
    print("P2P transfer not enabled.")
