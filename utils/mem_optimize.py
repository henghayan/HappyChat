import gc
import time

import torch
from offload_manager import OffloadManager


def model_to_recompute_mode(module, offload_manager=None, lr=0.003):
    if type(module) == torch.nn.ModuleList:
        children = list(module.named_children())
        for i in range(len(children)):
            item = children[i]
            if type(item[1]) == torch.nn.Linear:
                module[i] = RecomputeLinear(item[1].weight, item[1].bias, offload_manager=offload_manager, lr=lr)

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Linear:
            setattr(
                module,
                attr_str,
                RecomputeLinear(target_attr.weight, target_attr.bias, offload_manager=offload_manager, lr=lr),
            )

            gc.collect()
            torch.cuda.empty_cache()
    # a = list(module.named_children())
    for name, child in module.named_children():
        model_to_recompute_mode(child, offload_manager)


class RecomputeLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, module):
        ctx.module = module

        weight_index = weight.__dict__['param_index']
        if ctx.module.offload_manager:
            ctx.module.offload_manager.param_load(weight_index)

        # if not input.is_cuda:
        #     input.data = input.to(weight.device)
        output = torch.nn.functional.linear(input, weight, bias)
        # input.data = input.to('cpu')
        

        ctx.save_for_backward(input, weight, bias)
        # gc.collect()
        torch.cuda.empty_cache()
        if ctx.module.offload_manager:
            ctx.module.offload_manager.param_offload(weight_index)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        weight_index = weight.__dict__['param_index']
        if ctx.module.offload_manager:
            ctx.module.offload_manager.param_load(weight_index)

        # input.data = input.to(weight.device)
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            # grad_input = torch.bmm(grad_output, weight.expand(grad_output.size(0), *weight.size()))
            # 'ijk,kl->ijl' grad_output（ijk）和 weight（kl）相乘，然后对 k 维度进行求和，结果是 ijl
            # 即 (batch_size, seq_length, in_features)
            grad_input = torch.einsum('ijk,kl->ijl', grad_output, weight)
        if ctx.needs_input_grad[0]:
            # grad_weight = torch.bmm(input.transpose(1, 2), grad_output).sum(0).transpose(0, 1)
            # grad_weight = torch.einsum('bij,bij->ji', input, grad_output)
            intermediate = torch.einsum('ijk,ijl->kl', input, grad_output)
            grad_weight = intermediate.transpose(0, 1)
            ctx.module.weight -= grad_weight * ctx.module.lr
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
            ctx.module.bias -= grad_bias * ctx.module.lr
        # del input, weight, bias
        # torch.cuda.empty_cache()
        if ctx.module.offload_manager:
            ctx.module.offload_manager.param_offload(weight_index)
        return grad_input, grad_weight, grad_bias, None


class RecomputeLinear(torch.nn.Module):
    def __init__(self, weight, bias, offload_manager: OffloadManager = None, lr=0.003):
        super().__init__()
        self.weight = weight.detach()
        self.weight.__dict__ = weight.__dict__
        del weight
        self.offload_manager = offload_manager
        if offload_manager is not None:
            self.offload_manager.add_offload_param_tensor(self.weight)

        # self.weight.requires_grad = True
        self.bias = bias.detach() if bias is not None else None
        self.lr = lr

    def forward(self, input_tensor):
        return RecomputeLinearFunction.apply(input_tensor, self.weight, self.bias, self)


def recover_from_recompute_mode(module, same_device=False):
    if type(module) == torch.nn.ModuleList:
        children = list(module.named_children())
        for i in range(len(children)):
            item = children[i]

            if type(item[1]) == RecomputeLinear:
                if item[1].bias is None:
                    module[i] = torch.nn.Linear(item[1].weight.shape[1], item[1].weight.shape[0], bias=False,
                                                dtype=item[1].weight.dtype, device=item[1].weight.device)
                else:
                    module[i] = torch.nn.Linear(item[1].weight.shape[1], item[1].weight.shape[0],
                                                dtype=item[1].weight.dtype, device=item[1].weight.device)
                    module[i].bias.data.copy_(item[1].bias.data)
                module[i].weight.data.copy_(item[1].weight.data)
                # module[i] = RecomputeLinear(item[1].weight, item[1].bias, lr=lr)

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == RecomputeLinear:
            if target_attr.bias is None:
                setattr(module, attr_str,
                        torch.nn.Linear(target_attr.weight.shape[1], target_attr.weight.shape[0], bias=False,
                                        dtype=target_attr.weight.dtype, device=target_attr.weight.device))
            else:
                setattr(module, attr_str,
                        torch.nn.Linear(target_attr.weight.shape[1], target_attr.weight.shape[0],
                                        dtype=target_attr.weight.dtype, device=target_attr.weight.device))
                getattr(module, attr_str).bias.data.copy_(target_attr.bias.data)
            getattr(module, attr_str).weight.data.copy_(target_attr.weight.data)
    for name, child in module.named_children():
        recover_from_recompute_mode(child)


###########################以下为测试代码#####################################

import torch.nn as nn


class TimeLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, module):
        ctx.module = module
        ctx.save_for_backward(input, weight, bias)

        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        # start_event.record()

        output = torch.nn.functional.linear(input, weight, bias)

        # end_event.record()

        # torch.cuda.synchronize()
        # elapsed_time = start_event.elapsed_time(end_event)
        # print(f"[Forward] Time: {time.time()}, use_time:{elapsed_time}, layer:{module.i}, GPU:{module.c}")
        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # print("input", input.size())
        # print("weight", weight.size())
        # print("bias", bias.size())
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        # start_event.record()
        print("ctx.needs_input_grad", ctx.needs_input_grad)
        if ctx.needs_input_grad[0]:
            # grad_input = torch.bmm(grad_output, weight.expand(grad_output.size(0), *weight.size()))
            # 'ijk,kl->ijl' grad_output（ijk）和 weight（kl）相乘，然后对 k 维度进行求和，结果是 ijl
            # 即 (batch_size, seq_length, in_features)
            grad_input = torch.einsum('ijk,kl->ijl', grad_output, weight)
        if ctx.needs_input_grad[1]:
            # grad_weight = torch.bmm(input.transpose(1, 2), grad_output).sum(0).transpose(0, 1)
            grad_weight = torch.einsum('ijk,ikl->jkl', input, grad_output).sum(dim=0)
            ctx.module.weight -= grad_weight * ctx.module.lr
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
            # ctx.module.bias -= grad_bias * ctx.module.lr

        # end_event.record()

        # torch.cuda.synchronize()
        # elapsed_time = start_event.elapsed_time(end_event)
        # print(f"[Backward] Time: {time.time()}, use_time:{elapsed_time}, layer:{ctx.module.i}, GPU:{ctx.module.c}")
        return grad_input, grad_weight, grad_bias, None


class TimeLinear(torch.nn.Module):
    def __init__(self, in_, out, i, c):
        super().__init__()
        temp_linear = nn.Linear(in_, out)
        self.weight = temp_linear.weight.detach().to(f"cuda:{c}")
        self.bias = temp_linear.bias
        self.lr = 0.1
        self.i = i
        self.c = c
        self.device = f"cuda:{c}"

    def forward(self, input_tensor):
        return TimeLinearFunction.apply(input_tensor, self.weight, self.bias, self)


class SimpleLinearModel(nn.Module):

    def __init__(self):
        super(SimpleLinearModel, self).__init__()

        self.layers = nn.ModuleList([TempLinear(4096, 4096) for _ in range(2)])

        self.fc = nn.Linear(4096, 512)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.fc(x)
        return x


class TempLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super(TempLinear, self).__init__()
        self.fc = torch.nn.Linear(in_f, out_f)

    def forward(self, x):
        return self.fc(x)


if __name__ == "__main__":
    model = SimpleLinearModel()
    model.to("cuda:0")
    # model = TransformerTest(d_model, num_heads, num_layers, 512, dtype=dtype)
    criterion = nn.MSELoss()  # Loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    model_to_recompute_mode(model)

    input_seq_tensor = torch.rand(8192, 2, 4096, dtype=torch.float32).to("cuda:0")
    input_seq_tensor.requires_grad = True
    target_seq_tensor = torch.randn(8192, 2, 512, dtype=torch.float32).to("cuda:0")

    model.train()

    model_to_recompute_mode(model)

    output = model(input_seq_tensor)
    loss = criterion(output, target_seq_tensor)
    loss.backward()
    optimizer.step()
