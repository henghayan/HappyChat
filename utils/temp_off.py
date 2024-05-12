import torch
import torch.nn as nn
import transformers
import time


class OffloadManager:
    def __init__(self, model, cuda_num, offload_device='cpu'):
        self.model = model
        self.offload_device = offload_device
        self.streams = {}
        self.init_streams(cuda_num)

        # Pin and offload the specified layers to CPU at initialization
        self.offload_param_list = []
        self.origin_param_device_list = []
        self.offload_initial_layers()

    def init_streams(self, cuda_num):
        for i in range(cuda_num):
            self.streams['cpu_to_cuda:%s' % str(i)] = torch.cuda.Stream(device="cuda:%s" % i)
            self.streams['cuda:%s_to_cpu' % str(i)] = torch.cuda.Stream(device="cuda:%s" % i)

    def offload_initial_layers(self):
        offload_index = 0
        for name, param in self.model.named_parameters():
            if "layer" in name:
                param.__dict__["offload_index"] = offload_index
                offload_index += 1

                assert param.is_cuda

                param_device = param.device

                with torch.cuda.stream(self.streams['cuda:%s_to_cpu' % param_device.index]):
                    param.data = param.to(self.offload_device, non_blocking=True)
                param.data = param.data.pin_memory()
                self.offload_param_list.append(param)
                self.origin_param_device_list.append(param_device.index)
            else:
                param.__dict__["offload_index"] = None

    def param_load(self, offload_index: int, non_blocking: bool = True) -> None:
        origin_device_index = self.origin_param_device_list[offload_index]
        origin_device = torch.device("cuda:%s" % origin_device_index)
        target_param = self.offload_param_list[offload_index]
        stream_key = 'cpu_to_cuda:%s' % origin_device_index
        with torch.cuda.stream(self.streams[stream_key]):
            if not target_param.is_cuda:
                target_param.to(origin_device, non_blocking=non_blocking)

        self.streams[stream_key].synchronize()

    def param_offload(self, offload_index: int, non_blocking: bool = True) -> None:
        origin_device_index = self.origin_param_device_list[offload_index]

        target_param = self.offload_param_list[offload_index]
        stream_key = 'cuda:%s_to_cpu' % origin_device_index
        with torch.cuda.stream(self.streams[stream_key]):
            if target_param.is_cuda:
                self.offload_param_list[offload_index].to(self.offload_device, non_blocking=non_blocking)

        self.streams[stream_key].synchronize()
        torch.cuda.empty_cache()

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer_a = nn.Linear(2**11, 2**11, bias=False)
        self.layer_b = nn.Linear(2**11, 2**11, bias=False)
        self.layer_c = nn.Linear(2**11, 2**11, bias=False)

    def forward(self, x):
        x = self.layer_a(x)
        x = self.layer_b(x)
        x = self.layer_c(x)
        return x

class NLiner(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight.detach() *2
        self.weight.__dict__ = weight.__dict__


if __name__ == '__main__':
    # path = "/data2/llm3-8"
    # data_path = "/data/HappyChat/train_data/vir.json"
    #
    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     path,
    #     torch_dtype=torch.bfloat16,
    #     trust_remote_code=True,
    #     device_map="cuda:0"
    # )
    #
    # offload_mgr = OffloadManager(model, 1)
    # b = list(model.named_parameters())
    # torch.cuda.synchronize()
    # torch.cuda.empty_cache()
    #
    # index = 0
    # for name, p in model.named_parameters():
    #     print(name, p)
    #
    #     offload_mgr.param_load(index)
    #
    #     offload_mgr.param_offload(index)
    #
    #     index += 1




    # 初始化模型
    model = MyModel()
    a = list(model.named_parameters())
    # 将layer_b detach并替换
    model = model.to("cuda:0")
    for name, module in model.named_children():
        if module is model.layer_b:
            detached_b = NLiner(module.weight)
            setattr(model, name, detached_b)
            del module

    b = 1