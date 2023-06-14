from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, GPT2LMHeadModel, AdamW
import torch

# 配置模型和标记器
config = GPT2Config(n_embd=hidden_size,
                    n_layer=num_layers,
                    n_head=num_attention_heads,
                    n_positions=max_seq_len,
                    n_ctx=max_seq_len,
                    vocab_size=vocab_size)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 加载预训练模型
model = GPT2LMHeadModel(config)
model.train()  # 切换到训练模式

# 加载数据和优化器
dataset = YourDataset()  # 替换为您自己的数据集
optimizer = AdamW(model.parameters(), lr=1e-5)

# 训练模型
for epoch in range(num_epochs):
    for batch in dataset:
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# 切换到推理模式
model.eval()

# 保存模型
model.save_pretrained("path/to/save/model")

# 加载模型
model = GPT2LMHeadModel.from_pretrained("path/to/saved/model")

# 输入文本编码
input_text = "您的输入文本"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 生成文本
output = model.generate(input_ids, max_length=100, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
