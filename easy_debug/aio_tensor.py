import torch
from tensornvme import DiskOffloader
import time


def sync_api() :

    y = torch.rand(2**10, 2**10, 2**10, dtype=torch.bfloat16)
    offloader = DiskOffloader('/swp/tensornvme/')
    size = y.element_size() * y.nelement()
    start_time = time.time()
    print("memory:", size)

    offloader.sync_write(y)

    print("write use time: ", time.time() - start_time)
    # x is saved to a file on disk (in ./offload folder) and the memory of x is freed
    offloader.sync_read(y)
    # x is restored
    # offloader.sync_writev([x, y])
    # # x and y are offloaded
    # offloader.sync_readv([x, y])
    # x and y are restored.
    # sync_writev() and sync_readv() are order sensitive
    # E.g. sync_writev([x, y]) and sync_writev([y, x]) are different

    print("read use time: ", time.time() - start_time)

    y = y.to("cuda")
    torch.cuda.synchronize()

    print("to cuda use time: ", time.time() - start_time)


if __name__ == '__main__':
    sync_api()