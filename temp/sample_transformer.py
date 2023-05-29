import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from compression import *


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)


# 定义多头自注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dtype=torch.float32):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model, dtype=dtype)
        self.key = nn.Linear(d_model, d_model, dtype=dtype)
        self.value = nn.Linear(d_model, d_model, dtype=dtype)

        self.fc = nn.Linear(d_model, d_model, dtype=dtype)

    def forward(self, query, key, value, mask=None):
        N = query.shape[0]
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        Qview = Q.view(N, -1, self.num_heads, self.head_dim)

        Q = Qview.permute(0, 2, 1, 3)
        K = K.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.head_dim ** 0.5
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(attention, dim=-1)
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(N, -1, self.d_model)
        x = self.fc(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dtype=torch.float32):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward1 = nn.Linear(d_model, d_model * 4, dtype=dtype)
        self.feed_forward2 = nn.Linear(d_model * 4, d_model, dtype=dtype)
        self.relu = nn.ReLU()

    def forward(self, value, key, query, mask=None):
        attention = self.attention(query, key, value, mask)
        x = self.norm1(attention + query)
        ff = self.relu(self.feed_forward1(x))
        ff = self.feed_forward2(ff)
        x = self.norm2(ff + x)
        return x


# 定义Transformer模型
class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, vocab_size, dtype=torch.float32):
        super(Transformer, self).__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size, dtype=dtype)

    def forward(self, x, mask=None):
        N, seq_length = x.shape
        embding = self.embed(x)
        pos = self.pos_enc.pe[:, :seq_length, :]
        x = embding + pos

        for layer in self.layers:
            x = layer(x, x, x, mask)

        x = self.fc(x)
        return x




#################################################################################################################
#################################################################################################################

text = "这是一个很长的样本，这是一个很长的样本。这是一个很长的样本！哈"

chars = list(set(text))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}

sequence_length = 20
input_seqs = []
target_seqs = []

for i in range(len(text) - sequence_length):
    input_seq = text[i:i + sequence_length]
    target_seq = text[i + 1:i + sequence_length + 1]
    input_seqs.append([char_to_idx[char] for char in input_seq])
    target_seqs.append([char_to_idx[char] for char in target_seq])

# 参数设置
d_model = 8
num_heads = 8
num_layers = 1
vocab_size = len(chars)
n_epochs = 1
print_interval = 10


# 实例化模型
def get_init_model():
    model = Transformer(d_model, num_heads, num_layers, vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    return model, criterion, optimizer


# 训练模型
def train(model, criterion, optimizer):
    for epoch in range(n_epochs):
        step = 0
        for input_seq, target_seq in zip(input_seqs, target_seqs):
            input_seq_tensor = torch.tensor(input_seq).unsqueeze(0)
            target_seq_tensor = torch.tensor(target_seq).unsqueeze(0)

            optimizer.zero_grad()

            output = model(input_seq_tensor)
            loss = criterion(output.squeeze(0), target_seq_tensor.squeeze(0))

            loss.backward()
            optimizer.step()

            step += 1
            if (step + 1) % print_interval == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Step {step + 1}, Loss: {loss.item()}")


# 生成新的文本
def generate_text(model, seed_text, max_length=100):
    model.eval()

    # with torch.no_grad():
    input_seq = [char_to_idx[char] for char in seed_text]
    input_seq_tensor = torch.tensor(input_seq).unsqueeze(0)

    res_text = seed_text

    for _ in range(max_length):
        output = model(input_seq_tensor)
        _, predicted_idx = torch.max(output[:, -1, :], dim=-1)

        predicted_char = idx_to_char[predicted_idx.item()]
        res_text += predicted_char

        input_seq_tensor = torch.cat([input_seq_tensor, predicted_idx.unsqueeze(0)], dim=1)
        input_seq_tensor = input_seq_tensor[:, 1:]

    return res_text


def save(model, path="D:\\HappyChat\\test.bin"):
    torch.save(model.state_dict(), path)


def load_model(init_model, path="D:\\HappyChat\\test.bin"):
    init_model.load_state_dict(torch.load(path))


if __name__ == "__main__":
    model, criterion, optimizer = get_init_model()
    # load_model(model)
    print("premodel")
    # temp_data = getattr(model, "fc").weight
    # print("============model\n", model.state_dict())
    compress_module(model, "cpu")
    # decompress_module(model, dtype=torch.float32)
    train(model, criterion, optimizer)
    res = generate_text(model, "这是")
    print("res_text", res)
    # save(model)
    # print("===============model\n", model.state_dict())`