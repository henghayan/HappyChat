import gc

import torch


def model_to_recompute_mode(module, lr=0.003):
    if type(module) == torch.nn.ModuleList:
        children = list(module.named_children())
        for i in range(len(children)):
            item = children[i]

            if type(item[1]) == torch.nn.Linear:
                module[i] = RecomputeLinear(item[1].weight, item[1].bias, lr=lr)

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Linear:
            setattr(
                module,
                attr_str,
                RecomputeLinear(target_attr.weight, target_attr.bias, lr=lr),
            )
            gc.collect()
            torch.cuda.empty_cache()
    a = list(module.named_children())
    for name, child in module.named_children():
        model_to_recompute_mode(child)


class RecomputeLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, module):
        ctx.module = module
        ctx.save_for_backward(input, weight, bias)
        output = torch.nn.functional.linear(input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, weight, bias = ctx.saved_tensors

        print("input", input.size())
        print("weight", weight.size())
        print("output", output.size())
        grad_input = grad_weight = grad_bias = None

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
            ctx.module.bias -= grad_bias * ctx.module.lr
        return grad_input, grad_weight, grad_bias, None


class RecomputeLinear(torch.nn.Module):
    def __init__(self, weight, bias, lr=0.003):
        super().__init__()
        self.weight = weight.detach()
        self.bias = bias.detach() if bias is not None else None
        self.lr = lr

    def forward(self, input_tensor):
        return RecomputeLinearFunction.apply(input_tensor, self.weight, self.bias, self)


import torch.nn as nn


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
