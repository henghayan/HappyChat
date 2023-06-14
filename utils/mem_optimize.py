import gc

import torch


def model_to_recompute_mode(module):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Linear:
            setattr(
                module,
                attr_str,
                RecomputeLinear(target_attr.weight, target_attr.bias),
            )

    for name, child in module.named_children():
        model_to_recompute_mode(child)


class RecomputeLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        output = torch.nn.functional.linear(input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.bmm(grad_output, weight.expand(grad_output.size(0), *weight.size()))
        if ctx.needs_input_grad[1]:
            grad_weight = torch.bmm(input.transpose(1, 2), grad_output).sum(0).transpose(0, 1)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


class RecomputeLinear(torch.nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = weight.detach()
        self.bias = bias

    def forward(self, input_tensor):
        return RecomputeLinearFunction.apply(input_tensor, self.weight, self.bias)
