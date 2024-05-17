import gc
import time

import torch
# from offload_manager import OffloadManager


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


forward_total = 0
backward_total = 0

forward_load_total = 0
backward_load_total = 0

forward_liner_total = 0
backward_liner_total = 0

forward_offload_total = 0
backward_offload_total = 0

forward_save_total = 0


class RecomputeLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, module):
        global forward_total, forward_liner_total, forward_load_total, forward_offload_total, forward_save_total

        ctx.module = module
        start_time = time.time()
        load_time = start_time
        if ctx.module.offload_manager:
            ctx.module.offload_manager.param_load(weight, non_blocking=False)
            ctx.module.offload_manager.next_param_load(weight, non_blocking=True)
            load_time = time.time()
            forward_load_total += time.time() - start_time
            # print(f"[Forward] load from cpu, use: {time.time() - start_time:.5f} seconds ")
            # if not input.is_cuda:
            #     ctx.module.offload_manager.tensor_load(input, non_blocking=False)

        output = torch.nn.functional.linear(input, weight, bias)
        linear_time = time.time()
        forward_liner_total += linear_time - load_time
        # print(f"[Forward] linear use: {linear_time - load_time:.5f} seconds")

        if ctx.module.offload_manager:
            ctx.module.offload_manager.param_offload(weight, forward=True)
            # ctx.module.offload_manager.tensor_offload(input)
        offload_time = time.time()
        forward_offload_total += offload_time - linear_time

        ctx.save_for_backward(input, weight, bias)
        forward_save_total += time.time() - offload_time
        # print(f"[Forward] save use time: {time.time() - offload_time:.5f} seconds")

        # print(f"[Forward] item use time: {time.time() - start_time:.5f} seconds ")
        forward_total += time.time() - start_time
        # print(f"[Forward] total use time: {forward_total:.5f} seconds; forward_load_total: {forward_load_total:.5f}\n")
        return output

    @staticmethod
    def backward(ctx, grad_output):
        global backward_total, backward_liner_total, backward_offload_total
        global backward_load_total
        input, weight, bias = ctx.saved_tensors

        start_time = time.time()
        load_time = start_time
        if ctx.module.offload_manager:
            ctx.module.offload_manager.param_load(weight, non_blocking=False)
            ctx.module.offload_manager.pre_param_load(weight, non_blocking=True)
            # torch.cuda.synchronize()
            # if not input.is_cuda:
            #     ctx.module.offload_manager.tensor_load(input)
            load_time = time.time()
            backward_load_total += load_time - start_time
            # print(f"[Backward] load from cpu use: {time.time() - start_time:.5f} seconds")
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
        torch.cuda.synchronize()
        linear_time = time.time()
        backward_liner_total += linear_time - load_time
        # print(f"[Backward] back time: {linear_time - load_time:.5f} seconds\n")

        if ctx.module.offload_manager:
            ctx.module.offload_manager.param_offload(weight, backward=True)
            # ctx.module.offload_manager.tensor_offload(input)
        backward_offload_total += time.time() - linear_time
        backward_total += (time.time() - start_time)

        print(f"[forward] total use time: {forward_total:.5f};\n "
              f"forward_load_total: {forward_load_total:.5f};\n "
              f"forward_liner_total:{forward_liner_total:.5f};\n "
              f"forward_offload_total:{forward_offload_total:.5f};\n "
              f"[Backward] total use time: {backward_total:.5f} seconds;\n "
              f"backward_load_total:{backward_load_total:.5f};\n "
              f"backward_liner_total:{backward_liner_total:.5f};\n "
              f"backward_offload_total:{backward_offload_total:.5f};\n\n "
              f"forward_save_total:{forward_save_total}")
        return grad_input, grad_weight, grad_bias, None


class RecomputeLinear(torch.nn.Module):
    def __init__(self, weight, bias, offload_manager=None, lr=0.003):
        super().__init__()
        self.weight = weight.detach()
        self.weight.__dict__ = weight.__dict__
        del weight
        self.offload_manager = offload_manager
        if offload_manager is not None:
            self.offload_manager.register_and_offload_param(self.weight)

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


if __name__ == "__main__":
    pass
