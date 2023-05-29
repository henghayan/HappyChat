import gc
import dataclasses

import torch
import torch.nn.functional as F
import torch.autograd as autograd
import torch.nn as nn
from torch import Tensor


@dataclasses.dataclass
class CompressionConfig:
    num_bits: int
    group_size: int
    group_dim: int
    symmetric: bool
    enabled: bool = True


default_compression_config = CompressionConfig(
    num_bits=8, group_size=256, group_dim=1, symmetric=True, enabled=True)


class CompressLinearFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, bias, c_obj):
        weight = decompress(c_obj.compress_obj, default_compression_config)
        ctx.save_for_backward(input, bias)
        ctx.c_obj = c_obj
        output = F.linear(input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        grad_input = grad_bias = None

        weight = decompress(ctx.c_obj.compress_obj, default_compression_config)
        if ctx.needs_input_grad[0]:
            grad_input = torch.bmm(grad_output, weight.expand(grad_output.size(0), *weight.size()))
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).sum(0)
        weight_grad = torch.bmm(input.transpose(1, 2), grad_output).sum(0).transpose(0, 1)
        ctx.c_obj.compress_obj = compress((weight - weight_grad * 0.03).data, default_compression_config)
        return grad_input, grad_bias, None


class CLinear(nn.Module):

    def __init__(self, weight, bias, device):
        super().__init__()
        self.compress_obj = compress(weight.data.to(device), default_compression_config)
        self.bias = bias
        self.weight = weight
        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, input: Tensor) -> Tensor:
        return CompressLinearFunction.apply(input, self.bias, self)


def compress_module(module, target_device):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Linear:
            setattr(
                module,
                attr_str,
                CLinear(target_attr.weight, target_attr.bias, target_device),
            )
    for name, child in module.named_children():
        compress_module(child, target_device)


def compress(tensor, config):
    if not config.enabled:
        return tensor

    group_size, num_bits, group_dim, symmetric = (
        config.group_size, config.num_bits, config.group_dim, config.symmetric)
    assert num_bits <= 8

    original_shape = tensor.shape
    num_groups = (original_shape[group_dim] + group_size - 1) // group_size
    new_shape = (original_shape[:group_dim] + (num_groups, group_size) +
                 original_shape[group_dim + 1:])

    # Pad
    pad_len = (group_size - original_shape[group_dim] % group_size) % group_size
    if pad_len != 0:
        pad_shape = original_shape[:group_dim] + (pad_len,) + original_shape[group_dim + 1:]
        tensor = torch.cat([
            tensor,
            torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)],
            dim=group_dim)
    data = tensor.view(new_shape)

    # Quantize
    if symmetric:
        B = 2 ** (num_bits - 1) - 1
        scale = B / torch.max(data.abs(), dim=group_dim + 1, keepdim=True)[0]
        data = data * scale
        data = data.clamp_(-B, B).round_().to(torch.int8)
        return data, scale, original_shape
    else:
        B = 2 ** num_bits - 1
        mn = torch.min(data, dim=group_dim + 1, keepdim=True)[0]
        mx = torch.max(data, dim=group_dim + 1, keepdim=True)[0]

        scale = B / (mx - mn)
        data = data - mn
        data.mul_(scale)

        data = data.clamp_(0, B).round_().to(torch.uint8)
        return data, mn, scale, original_shape


def decompress(packed_data, config):
    if not config.enabled:
        return packed_data

    group_size, num_bits, group_dim, symmetric = (
        config.group_size, config.num_bits, config.group_dim, config.symmetric)

    # Dequantize
    if symmetric:
        data, scale, original_shape = packed_data
        data = data / scale
    else:
        data, mn, scale, original_shape = packed_data
        data = data / scale
        data.add_(mn)

    # Unpad
    pad_len = (group_size - original_shape[group_dim] % group_size) % group_size
    if pad_len:
        padded_original_shape = (
                original_shape[:group_dim] +
                (original_shape[group_dim] + pad_len,) +
                original_shape[group_dim + 1:])
        data = data.reshape(padded_original_shape)
        indices = [slice(0, x) for x in original_shape]
        return data[indices].contiguous()
    else:
        return data.view(original_shape)


def decompress_module(module, dtype):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if isinstance(target_attr, CLinear):
            decompressed_weight = decompress(target_attr.weight, default_compression_config)
            if target_attr.bias is None:
                setattr(module, attr_str,
                        torch.nn.Linear(decompressed_weight.shape[1], decompressed_weight.shape[0], bias=False,
                                        dtype=dtype))
            else:
                setattr(module, attr_str,
                        torch.nn.Linear(decompressed_weight.shape[1], decompressed_weight.shape[0], dtype=dtype))
                getattr(module, attr_str).bias.data.copy_(target_attr.bias.data)
            getattr(module, attr_str).weight.data.copy_(decompressed_weight)

    for name, child in module.named_children():
        decompress_module(child, dtype)
