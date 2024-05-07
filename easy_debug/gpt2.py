import gc

import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, GPT2Config
import torch
from data_loader import load_train_data

from utils.mem_optimize import model_to_recompute_mode


# torch.cuda.is_available = lambda: False


def train(model, tokenizer, data_path, cutoff_len=128):
    train_data = load_train_data(data_path, tokenizer, int(cutoff_len))

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./gpt2",  # 输出目录
            overwrite_output_dir=True,
            num_train_epochs=10,  # 训练轮数
            per_device_train_batch_size=1,  # 训练批次大小
            save_steps=10000,
            save_total_limit=2,
        ),
        train_dataset=train_data,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True),
    )

    trainer.train()


def generate(model, tokenizer, input_text):
    model.eval()
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    input_ids = input_ids.to("cuda:0")
    # 使用模型生成文本
    output = model.generate(input_ids, max_length=100, temperature=0.9)
    print(tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True))


def get_diy_model():
    config = GPT2Config(n_embd=2048,
                        n_layer=24,
                        n_head=16,
                        n_positions=1024,
                        n_ctx=1024,
                        vocab_size=50257,
                        # use_bfloat16=True,
                        # torch_dtype=torch.float16,
                        qwer="qwer")
    # config.json = GPT2Config()
    return GPT2LMHeadModel(config)




if __name__ == "__main__":
    torch.cuda.init()
    # 初始化模型和分词器
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    # print(tokenizer)
    # model = GPT2LMHeadModel.from_pretrained('/data/gpt2',
    #                                         device_map="auto",
    #                                         # torch_dtype=torch.float32,
    #                                         trust_remote_code=True)
    # # tokenizer.pad_token = tokenizer.eos_token
    model = get_diy_model()
    model = model.to("cuda")
    model_to_recompute_mode(model)

    gc.collect()
    torch.cuda.empty_cache()
    train(model, tokenizer, "/data/HappyChat/train_data/test.json")
    # print(model)

    input_text = "Explain how to solve the system of equations: x + y = 5, x - y = 1"
    generate(model, tokenizer, input_text)
