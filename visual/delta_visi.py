import torch
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from transformers import AutoModelForCausalLM

# 加载训练前的模型
model1 = AutoModelForCausalLM.from_pretrained('/data/Llama-2-13b-chat-hf',
                                              torch_dtype=torch.float16,
                                              trust_remote_code=True,
                                              device_map="auto",
                                              )

# 选择要关注的层
all_layer_names = [name for name, _ in model1.named_parameters()]
print("all_layer_names", len(all_layer_names), all_layer_names)


c = 59
layers_to_watch = all_layer_names[c:100]

# 获取训练前的权重并移动到CPU内存
pre_train_weights = {name: param.clone().cpu() for name, param in model1.named_parameters() if
                     any(layer in name for layer in layers_to_watch)}

# 删除模型并进行垃圾回收
del model1
gc.collect()
torch.cuda.empty_cache()


# 加载训练后的模型
model2 = AutoModelForCausalLM.from_pretrained('/data/llm2',
                                              torch_dtype=torch.float16,
                                              trust_remote_code=True,
                                              device_map="auto",
                                              )

# 获取训练后的权重并移动到CPU内存
post_train_weights = {name: param.clone().cpu() for name, param in model2.named_parameters() if
                      any(layer in name for layer in layers_to_watch)}

# 删除模型并进行垃圾回收
del model2
gc.collect()
torch.cuda.empty_cache()

# 分批次比较权重

for key in pre_train_weights.keys():
    # 获取训练前的权重
    pre_weight = pre_train_weights[key]

    # 获取训练后的权重
    post_weight = post_train_weights[key]

    # 计算权重的差异
    diff = torch.sub(post_weight, pre_weight)

    # 检查训练前的权重是否包含零，并将它们替换为一个非常小的值
    pre_weight = torch.where(pre_weight == 0, torch.full_like(pre_weight, 1e-7), pre_weight)

    # 检查权重差异是否包含非常大的值，并将它们替换为一个较小的值
    diff = torch.where(diff > 1e4, torch.full_like(diff, 1e4), diff)

    # 计算权重差异比例
    ratio = torch.div(diff, pre_weight)
    print("check value:", key, ratio.min(), ratio.max(), ratio.mean())

    # 可视化权重的比例
    plt.figure(figsize=(20, 16))
    # 将数据转换为二维的
    if len(ratio.shape) == 1:
        ratio = ratio.reshape(-1, 1)

    sns.heatmap(ratio.detach().numpy(), cmap='coolwarm', center=0, vmin=-1, vmax=1)

    plt.title(f'Weight ratio for {key}')
    print(f"start saving {key}_ratio.png...", c)
    # 保存图形为文件
    plt.savefig(f'/data/visual/llm2/{key}_ratio.png')

    # 清理内存
    del pre_weight
    del post_weight
    del ratio
    del diff
    gc.collect()

    # 关闭图形
    plt.close()
    c += 1
# 展示所有的图形
# plt.show()
