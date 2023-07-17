import torch.nn as nn
import deepspeed
import argparse

net = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(inplace=True),
    nn.Linear(10, 10)
)

# 创建一个简单的命令行参数对象，只包含我们所需的最小深度速度配置
args = argparse.Namespace()
args.deepspeed = True
args.local_rank = -1

# 初始化DeepSpeed
model_engine, _, _, _ = deepspeed.initialize(args=args, model=net)

# 现在可以尝试创建PipelineModule了
from deepspeed.pipe import PipelineModule
net = PipelineModule(layers=net, num_stages=2)


# layer1 = Layer(0)
# layer2 = Layer(multi_gpu)
# layers = nn.Sequential(layer1, layer2)
# # model = model.to("cuda")
# # model = PipelineModule(layers=layers, num_stages=2)
# model_engine, _, _, _ = deepspeed.initialize(
#     model=layers,
#     config_params=deepspeed_config
# )
#
# #
#

from deepspeed.pipe import PipelineModule

# model = PipelineModule(layers=layers, num_stages=2)
#
# if __name__ == "__main__":
#     loss_fn = nn.MSELoss()
#
#     # data = torch.randn(4096, 4096, dtype=torch.float16).to(model.local_rank)
#     # target = torch.randn(4096, 4096, dtype=torch.float16).to(model.local_rank)
#     data = torch.randn(16, 512, 4096 * 2).to(model.local_rank)
#     target = torch.randn(16, 512, 4096 * 2).to(model.local_rank)
#     start = time.time()
#     print("##############start time:", time.time())
#     for epoch in range(10):
#         outputs = model_engine(data)
#         # output = model(data)
#         # loss = loss_fn(output, target)
#         loss = loss_fn(outputs, target)
#         model_engine.backward(loss)
#         model_engine.step()
#         # model.step()
#         print("epoch", epoch)
#     print("use time:", time.time() - start)