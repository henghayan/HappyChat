import torch

batch_size = 2
sequence_length = 3
input_features = 4
output_features = 2

# 创建一个随机输入张量和随机权重矩阵
input_tensor = torch.randn(batch_size, sequence_length, input_features)
weights = torch.randn(output_features, input_features)

# 将输入张量重新排列成二维形状 (batch_size * sequence_length, input_features)
reshaped_input = input_tensor.view(batch_size * sequence_length, input_features)

# 执行前向传播：矩阵乘法
output = torch.matmul(reshaped_input, weights.t())

# 将输出张量重新排列回三维形状 (batch_size, sequence_length, output_features)
output = output.view(batch_size, sequence_length, output_features)

# 创建一个与输出形状相同的全1张量作为梯度
grad_output = torch.ones_like(output)

grad_reshape = grad_output.view(batch_size * sequence_length, output_features)
# 执行反向传播：矩阵乘法的梯度计算
grad_input_reshaped = torch.matmul(grad_reshape, weights)
grad_input = grad_input_reshaped.view(batch_size, sequence_length, input_features)

print("输出:")
print(output)

print("输入处理的梯度:")
print(grad_reshape)

print("权重:")
print(weights)

print("grad_input_reshaped:")
print(grad_input_reshaped)

#
# print("grad_input:")
# print(grad_input)


gradT = grad_output.view(batch_size * sequence_length, output_features).t()
grad_weights = torch.matmul(gradT, reshaped_input)
grad_weights = grad_weights.view(output_features, input_features)

print("gradT:")
print(gradT)

print("grad_input_reshaped:")
print(grad_input_reshaped)

print("权重矩阵的梯度:")
print(grad_weights)
