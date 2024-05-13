import gc

import torch
import transformers
import time


class OffloadManager:
    def __init__(self, model, cuda_num, offload_device='cpu'):
        self.model = model
        self.mark_model_params()

        self.streams = {}
        self._init_streams(cuda_num)

        self.offload_device = offload_device
        self._offload_param_map = {}  # "param_index": {"offload_index": xx, "param": xx}
        self._offload_param_index_sort_list = []  # 参与offload参数下标排序,代替双向链表

    def _update_offload_sort(self):
        self._offload_param_index_sort_list = sorted(self._offload_param_index_sort_list)
        for i in range(len(self._offload_param_index_sort_list)):
            param_index = self._offload_param_index_sort_list[i]
            self._offload_param_map[param_index]['offload_index'] = i

    def _init_streams(self, cuda_num):
        for i in range(cuda_num):
            self.streams['cpu_to_cuda:%s' % str(i)] = torch.cuda.Stream(device="cuda:%s" % i)
            self.streams['cuda:%s_to_cpu' % str(i)] = torch.cuda.Stream(device="cuda:%s" % i)

    def add_offload_param_tensor(self, weight_tensor):
        assert weight_tensor.is_cuda
        device_index = weight_tensor.device.index
        param_index = weight_tensor.__dict__['param_index']

        self._offload_param_map[param_index] = {'offload_index': len(self._offload_param_index_sort_list),
                                                'param': weight_tensor}
        self._offload_param_index_sort_list.append(param_index)
        self._update_offload_sort()

        # Move weight_tensor to CPU
        self.param_offload(weight_tensor)

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
        # target_param = self._offload_param_map[param_index]['param']
        origin_device_index = target_param.__dict__['origin_device_index']

        origin_device = torch.device("cuda:%s" % origin_device_index)
        stream_key = 'cpu_to_cuda:%s' % origin_device_index
        with torch.cuda.stream(self.streams[stream_key]):
            if not target_param.is_cuda:
                target_param.data = target_param.to(origin_device, non_blocking=non_blocking)

        self.streams[stream_key].synchronize()

    def param_offload(self, target_param, non_blocking: bool = True) -> None:
        # target_param = self._offload_param_map[param_index]['param']
        origin_device_index = target_param.__dict__['origin_device_index']

        assert origin_device_index > -1
        stream_key = 'cuda:%s_to_cpu' % origin_device_index
        with torch.cuda.stream(self.streams[stream_key]):
            if target_param.is_cuda:
                target_param.data = target_param.to(self.offload_device, non_blocking=non_blocking)

        self.streams[stream_key].synchronize()
        torch.cuda.empty_cache()

    def tensor_load(self, tensor, cuda_index=0, non_blocking: bool = True) -> None:
        target_device = torch.device("cuda:%s" % cuda_index)
        stream_key = 'cpu_to_cuda:%s' % cuda_index
        with torch.cuda.stream(self.streams[stream_key]):
            if not tensor.is_cuda:
                tensor.data = tensor.to(target_device, non_blocking=non_blocking)
        self.streams[stream_key].synchronize()

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
    s = torch.cuda.Stream(device="cuda:0")
    with torch.cuda.stream(s):
        pass

