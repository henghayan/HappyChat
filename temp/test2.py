import torch
import torch.autograd as autograd


class LinearFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        # 计算前向传播的结果
        output = torch.matmul(input, weight.t())

        # 保存需要在反向传播中使用的变量
        ctx.save_for_backward(input, weight)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 获取前向传播保存的变量
        input, weight = ctx.saved_tensors

        # 计算输入和权重的梯度
        grad_input = torch.matmul(grad_output, weight)
        grad_weight = torch.matmul(grad_output.t(), input)

        return grad_input, grad_weight.t()


batch_size = 2
sequence_length = 3
input_features = 4
output_features = 2

# 创建一个随机输入张量和随机权重矩阵
input_tensor = torch.randn(batch_size, sequence_length, input_features)
weights = torch.randn(output_features, input_features)

# 将输入张量重新排列成二维形状 (batch_size * sequence_length, input_features)
reshaped_input = input_tensor.view(batch_size * sequence_length, input_features)

# 创建 LinearFunction 对象
linear_function = LinearFunction.apply

# 执行前向传播：调用 LinearFunction 的 forward 方法
output = linear_function(reshaped_input, weights)

# 将输出张量重新排列回三维形状 (batch_size, sequence_length, output_features)
output = output.view(batch_size, sequence_length, output_features)

# 创建一个与输出形状相同的全1张量作为梯度
grad_output = torch.ones_like(output)

# 执行反向传播：由PyTorch框架自动处理
output.backward(grad_output)

print("输入张量的梯度:")
print(input_tensor.grad)

print("权重矩阵的梯度:")
print(weights.grad)
