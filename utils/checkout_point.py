import torch.nn
from torch.utils.checkpoint import checkpoint
from types import MethodType


def make_checkpointed(model):
    # 遍历模型的所有子模块
    modules = list(model.named_modules())
    for name, module in modules:
        if not isinstance(module, torch.nn.Linear):
            continue
        # 保存原始的 forward 方法
        old_forward = module.forward

        # 定义新的 forward 方法，该方法使用 checkpoint
        def new_forward(self, *args, **kwargs):
            return checkpoint(old_forward, *args, **kwargs)

        # 将新的 forward 方法设置为模块的 forward 方法
        module.forward = MethodType(new_forward, module)

    return model
