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
        self.param_offload(param_index)

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

    def param_load(self, param_index: int, non_blocking: bool = True) -> None:
        target_param = self._offload_param_map[param_index]['param']
        origin_device_index = target_param.__dict__['origin_device_index']

        origin_device = torch.device("cuda:%s" % origin_device_index)
        stream_key = 'cpu_to_cuda:%s' % origin_device_index
        with torch.cuda.stream(self.streams[stream_key]):
            if not target_param.is_cuda:
                target_param.to(origin_device, non_blocking=non_blocking)

        self.streams[stream_key].synchronize()

    def param_offload(self, param_index: int, non_blocking: bool = True) -> None:
        target_param = self._offload_param_map[param_index]['param']
        origin_device_index = target_param.__dict__['origin_device_index']

        assert origin_device_index > -1
        stream_key = 'cuda:%s_to_cpu' % origin_device_index
        with torch.cuda.stream(self.streams[stream_key]):
            if target_param.is_cuda:
                target_param.to(self.offload_device, non_blocking=non_blocking)

        self.streams[stream_key].synchronize()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    from mem_optimize import model_to_recompute_mode

    base_model = "/data2/llm3-8"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    # model_to_recompute_mode(model)
    OM = OffloadManager(model, 3)
    pre_time = time.time()
    torch.cuda.empty_cache()
    for name, p in model.named_parameters():
        time_start = time.time()
        param_index = p.__dict__["param_index"]

        OM.add_offload_param_tensor(p)
        time_offload = time.time()
        print(param_index, "[%s]offload use: " % name, time_offload - time_start)
        #
        #

        OM.param_load(param_index)
        # torch.cuda.synchronize()
        print(param_index, "[%s]load use: " % name, time.time() - time_offload)
        #
        # a.param_offload(index)
        # torch.cuda.synchronize()
        #
    print('total time: ', time.time() - pre_time)
