import torch
import transformers


def resize_embedding_weights(model_path, new_vocab_size, out_dir):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    # ######## embeddings ###############
    # 找到第一个参数
    first_param_name, first_param_value = next(iter(model.named_parameters()))
    print(f"The first parameter is '{first_param_name}'.")
    old_embeddings = first_param_value.data
    old_vocab_size, embedding_dim = old_embeddings.shape
    new_embeddings = torch.zeros(new_vocab_size, embedding_dim)
    new_embeddings[:old_vocab_size] = old_embeddings
    new_embeddings[old_vocab_size:] = torch.randn((new_vocab_size - old_vocab_size, embedding_dim))* 0.01

    new_emd_layer = torch.nn.Embedding.from_pretrained(new_embeddings)
    model_layer = getattr(model, first_param_name.split(".")[0])
    setattr(model_layer, first_param_name.split(".")[1], new_emd_layer)
    model.config.vocab_size = new_vocab_size

    # ######### full_connect ##########
    old_linear_weight = model.lm_head.weight
    old_vocab_size, embedding_size = old_linear_weight.shape
    new_linear_weight = torch.randn(new_vocab_size, embedding_size) * 0.01  # 应该使用正态分布初始化
    # new_linear_weight = torch.zeros(new_vocab_size, embedding_size)  # 这里只是验证
    new_linear_weight[:old_vocab_size, :] = old_linear_weight
    new_lm_head = torch.nn.Linear(embedding_size, new_vocab_size, bias=False)
    new_lm_head.weight.data = new_linear_weight
    model.lm_head = new_lm_head

    # 使用新的权重张量来更新嵌入层的权重

    model.save_pretrained(out_dir)
    return


if __name__ == "__main__":
    resize_embedding_weights("/data/Xwin-LM-13B-V0.1", 40706, "/data/x_win_40706/")
