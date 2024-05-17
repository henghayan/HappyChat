import gc

import torch
import transformers
import time

wait_cache = 0
class OffloadManager:
    def __init__(self, model, cuda_num, offload_device='cpu'):
        self.model = model
        self.mark_model_params()

        self.streams = {}
        self._init_streams(cuda_num)

        self.offload_device = offload_device
        self._offload_param_map = {}  # "param_index": {"offload_index": xx, "param": xx, "cpu_pin_tensor": xxx}
        self._offload_param_index_sort_list = []  # 参与offload参数下标排序,代替双向链表
        self._clear_cache_step = 8
        self.cur_step = 0

        self.pre_load_stream = None


    def _update_offload_sort(self):
        self._offload_param_index_sort_list = sorted(self._offload_param_index_sort_list)
        for i in range(len(self._offload_param_index_sort_list)):
            param_index = self._offload_param_index_sort_list[i]
            self._offload_param_map[param_index]['offload_index'] = i

    def _init_streams(self, cuda_num):
        for i in range(cuda_num):
            self.streams['cpu_to_cuda:%s' % str(i)] = torch.cuda.Stream(device="cuda:%s" % i)
            self.streams['cuda:%s_to_cpu' % str(i)] = torch.cuda.Stream(device="cuda:%s" % i)

    @staticmethod
    def _cpu_tensor_allocate(cuda_tensor, dtype=torch.bfloat16):
        tensor_size = cuda_tensor.size()
        dtype = cuda_tensor.dtype
        tensor_cpu_pinned = torch.empty(tensor_size, pin_memory=True, dtype=dtype)
        return tensor_cpu_pinned

    def register_and_offload_param(self, weight_tensor):
        assert weight_tensor.is_cuda
        device_index = weight_tensor.device.index
        param_index = weight_tensor.__dict__['param_index']

        pin_tensor = self._cpu_tensor_allocate(weight_tensor)  # 在cpu中占位，防止碎片化，加速

        self._offload_param_map[param_index] = {'offload_index': len(self._offload_param_index_sort_list),
                                                'param': weight_tensor, 'cpu_pin_tensor': pin_tensor}

        self._offload_param_index_sort_list.append(param_index)
        self._update_offload_sort()

        self.param_offload(weight_tensor, init=True) # todo 初始化需要先加载一次，否则损失计算会爆炸，bug，原因可能与快速首次传入值可能存在丢失

    def mark_model_params(self):
        param_index = 0
        for name, param in self.model.named_parameters():
            if param.is_cuda:
                origin_device_index = param.device.index
            else:
                origin_device_index = -1  # cpu
            param.__dict__["param_index"] = param_index
            param.__dict__["param_name"] = name
            param.__dict__["origin_device_index"] = origin_device_index
            param_index += 1

    def param_load(self, target_param, non_blocking: bool = True) -> None:
        # global wait_cache
        param_index = target_param.__dict__.get('param_index', -1)
        origin_device_index = target_param.__dict__.get('origin_device_index', -1)

        assert param_index > -1
        assert origin_device_index > -1
        #
        # need_synchronize_index_list = [self._offload_param_index_sort_list[0], self._offload_param_index_sort_list[-1]]
        #
        # if param_index in need_synchronize_index_list:
        #     torch.cuda.synchronize()
        #     torch.cuda.empty_cache()

        origin_device = torch.device("cuda:%s" % origin_device_index)

        # stream_key = 'cpu_to_cuda:%s' % origin_device_index
        # stream = self.streams[stream_key]
        new_stream_key = 'cuda:%s' % origin_device_index  # 卸载过快会导致原self.streams不稳定，不采用stream池
        stream = torch.cuda.Stream(new_stream_key)

        if not non_blocking:  # 非预加载
            if self.pre_load_stream:  #将上次预加载完全
                self.pre_load_stream.synchronize()

            with torch.cuda.stream(stream):
                if not target_param.is_cuda:
                    # target_param.data = target_param.to(origin_device, non_blocking=non_blocking)

                    param_info = self._offload_param_map[param_index]
                    pin_tensor = param_info.get('cpu_pin_tensor', None)
                    assert pin_tensor is not None
                    target_param.data = pin_tensor.to(origin_device, non_blocking=non_blocking)

        else:  # 预加载
            self.pre_load_stream = stream
            with torch.cuda.stream(stream):
                if not target_param.is_cuda:
                    # target_param.data = target_param.to(origin_device, non_blocking=non_blocking)
                    param_info = self._offload_param_map[param_index]
                    pin_tensor = param_info.get('cpu_pin_tensor', None)
                    assert pin_tensor is not None
                    target_param.data = pin_tensor.to(origin_device, non_blocking=non_blocking)

    def next_param_load(self, cur_param, non_blocking: bool = True) -> None:
        cur_param_index = cur_param.__dict__['param_index']
        cur_param_offload_index = self._offload_param_map[cur_param_index]['offload_index']
        if cur_param_offload_index < len(self._offload_param_index_sort_list) - 1:
            next_param_index = self._offload_param_index_sort_list[cur_param_offload_index + 1]
            next_param = self._offload_param_map[next_param_index]['param']
            self.param_load(next_param, non_blocking=non_blocking)

    def pre_param_load(self, cur_param, non_blocking: bool = True) -> None:
        cur_param_index = cur_param.__dict__['param_index']
        cur_param_offload_index = self._offload_param_map[cur_param_index]['offload_index']
        if cur_param_offload_index > 0:
            pre_param_index = self._offload_param_index_sort_list[cur_param_offload_index - 1]
            pre_param = self._offload_param_map[pre_param_index]['param']
            self.param_load(pre_param, non_blocking=non_blocking)

    def param_offload(self, target_param, non_blocking: bool = True, init=False, forward=False, backward=False) -> None:

        param_index = target_param.__dict__.get('param_index', -1)
        origin_device_index = target_param.__dict__.get('origin_device_index', -1)

        assert param_index > -1
        assert origin_device_index > -1

        # stream_key = 'cuda:%s_to_cpu' % origin_device_index
        # stream = self.streams[stream_key]

        new_stream_key = 'cuda:%s' % origin_device_index
        stream = torch.cuda.Stream(new_stream_key)

        with torch.cuda.stream(stream):

            if target_param.is_cuda:
                param_info = self._offload_param_map[param_index]
                pin_tensor = param_info.get('cpu_pin_tensor', None)
                assert pin_tensor is not None
                pin_tensor.copy_(target_param, non_blocking=non_blocking)
                target_param.data = torch.empty(0, device="cpu", dtype=target_param.dtype)

                # target_param.data = target_param.to(self.offload_device, non_blocking=non_blocking)

        self.cur_step += 1

        need_synchronize_index_list = [self._offload_param_index_sort_list[0], self._offload_param_index_sort_list[-1]]
        if param_index in need_synchronize_index_list:
            self.clear_cache()
        elif backward:  # 反向传播已经经过内存拐点，可不处理，仅在回到第一节点回收一次
            pass
        elif self.cur_step % self._clear_cache_step == 0:
            self.clear_cache()
        

        #
        #
        #     torch.cuda.synchronize()
        #     torch.cuda.empty_cache()

    @staticmethod
    def clear_cache():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def tensor_load(self, tensor, cuda_index=0, non_blocking: bool = True) -> None:
        target_device = torch.device("cuda:%s" % cuda_index)
        stream_key = 'cpu_to_cuda:%s' % cuda_index
        with torch.cuda.stream(self.streams[stream_key]):
            if not tensor.is_cuda:
                tensor.data = tensor.to(target_device, non_blocking=non_blocking)
        # self.streams[stream_key].synchronize()

    def tensor_offload(self, tensor, cuda_index=0, non_blocking: bool = True) -> None:
        stream_key = 'cuda:%s_to_cpu' % cuda_index
        with torch.cuda.stream(self.streams[stream_key]):
            if tensor.is_cuda:
                tensor.data = tensor.to(self.offload_device, non_blocking=non_blocking)
        self.streams[stream_key].synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        a = 1


if __name__ == '__main__':
    dtype = torch.bfloat16
    model = transformers.AutoModelForCausalLM.from_pretrained(
        '/data2/llm3-8',
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="cuda:0"
    )

    offload_mgr = OffloadManager(model, 1)
    time_start = time.time()

    # s = torch.cuda.Stream(device="cuda:0")
    # with torch.cuda.stream(s):
    for name, param in model.named_parameters():
        offload_mgr.register_and_offload_param(param)
    init_time = time.time()

    print(f"init use time: {init_time - time_start:.5f} seconds")

    for name, param in model.named_parameters():
        offload_mgr.param_load(param, non_blocking=False)
    load_time = time.time()
    print(f"load use time: {load_time - init_time:.5f} seconds")

    for name, param in model.named_parameters():
        offload_mgr.param_offload(param, non_blocking=False)
    offload_time = time.time()
    print(f"offload use time: {offload_time - load_time:.5f} seconds")
