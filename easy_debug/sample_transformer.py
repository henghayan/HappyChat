import time
import gc

import torch.optim as optim
from torch.utils.checkpoint import checkpoint

from utils.mem_optimize import model_to_recompute_mode, RecomputeLinear
from utils.compression import *


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dtype=torch.float16):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        pe = pe.to(dtype)
        self.register_buffer('pe', pe)


# 定义多头自注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dtype=torch.float16, device="cuda"):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model, dtype=dtype).to(device)
        self.key = nn.Linear(d_model, d_model, dtype=dtype).to(device)
        self.value = nn.Linear(d_model, d_model, dtype=dtype).to(device)

        self.fc = nn.Linear(d_model, d_model, dtype=dtype).to(device)
        self.device = device

    def forward(self, query, key, value, mask=None):
        query = query.to(self.device)
        key = key.to(self.device)
        value = value.to(self.device)
        mask = mask.to(self.device) if mask else None

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
    def __init__(self, d_model, num_heads, device="cuda", dtype=torch.float16, i=0):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dtype=dtype, device=device)
        self.norm1 = nn.LayerNorm(d_model, dtype=dtype)
        self.norm2 = nn.LayerNorm(d_model, dtype=dtype)

        self.feed_forward1 = nn.Linear(d_model, d_model * 4, dtype=dtype)
        self.feed_forward2 = nn.Linear(d_model * 4, d_model, dtype=dtype)
        self.relu = nn.ReLU()
        self.device = device
        self.i = i

    def forward(self, x, mask=None):
        x = x.to(self.device)
        mask = mask.to(self.device) if mask is not None else None
        # attention = self.attention(query, key, value, mask)
        value = key = query = x
        attention = self.attention(query, key, value, mask)

        x = self.norm1(attention + query)
        ff1 = self.feed_forward1(x)
        # ff1 = checkpoint(self.feed_forward1, x)
        ff1_relu = self.relu(ff1)
        ff2 = self.feed_forward2(ff1_relu)
        # ff2 = checkpoint(self.feed_forward2, ff1_relu)
        x = self.norm2(ff2 + x)

        return x


import torch.distributed.pipeline.sync as pipe_sync


# class TransformerTest(nn.Module):
#     def __init__(self, d_model, num_heads, num_layers, vocab_size, dtype=torch.float16):
#         super(TransformerTest, self).__init__()
#
#         self.embed = nn.Embedding(vocab_size, d_model, dtype=dtype).to('cuda:0')
#         self.pos_enc = PositionalEncoding(d_model, dtype=dtype).to('cuda:0')
#         self.fc = nn.Linear(d_model, vocab_size, dtype=dtype).to('cuda:1')
#         self.dtype = dtype
#         for layer_i in range(num_layers):
#             print("layer_i", layer_i)
#
#         self.layers = nn.Sequential(*[
#             TransformerBlock(d_model, num_heads, device=f'cuda:{int(layer_i // (num_layers / 2))}', dtype=dtype,
#                              i=layer_i).to(
#                 f'cuda:{int(layer_i // (num_layers / 2))}')
#             for layer_i in range(num_layers)
#         ])
#         a = list(self.layers.named_modules())
#         self.layers = pipe_sync.Pipe(self.layers, chunks=8)
#         b = list(self.layers.named_modules())
#         print("1")
#
#     def forward(self, x, mask=None):
#         N, seq_length = x.shape
#         embding = self.embed(x)
#         pos = self.pos_enc.pe[:, :seq_length, :]
#         x = embding + pos
#
#         # first_partition_device = next(self.layers.parameters()).device
#         x = x.to("cuda:0")
#         x = self.layers(x, mask)
#         x = x.to_here()
#         x = self.fc(x)
#         return x


# 定义Transformer模型
class TransformerTest(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, vocab_size, dtype=torch.float16):
        super(TransformerTest, self).__init__()

        self.embed = nn.Embedding(vocab_size, d_model, dtype=dtype)
        self.pos_enc = PositionalEncoding(d_model, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size, dtype=dtype)
        self.dtype = dtype

    def forward(self, x, mask=None):
        N, seq_length = x.shape
        embding = self.embed(x)
        pos = self.pos_enc.pe[:, :seq_length, :]
        x = embding + pos

        for layer in self.layers:
            x = layer(x, mask)

        x = self.fc(x)
        return x


#################################################################################################################
#################################################################################################################

text = '''这是一个很长的样本，这是一个很长的样本。这是一个很长的样本！哈,这是一个很长的样本，这是一个很长的样本。
这是一个很长的样本，这是一个很长的样本。这是一个很长的样本！哈,这是一个很长的样本，这是一个很长的样本。
这是一个很长的样本，这是一个很长的样本。这是一个很长的样本！哈,这是一个很长的样本，这是一个很长的样本。
'''

chars = list(set(text))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}

sequence_length = 32
input_seqs = []
target_seqs = []

for i in range(len(text) - sequence_length):
    input_seq = text[i:i + sequence_length]
    target_seq = text[i + 1:i + sequence_length + 1]
    input_seqs.append([char_to_idx[char] for char in input_seq])
    target_seqs.append([char_to_idx[char] for char in target_seq])

# 参数设置
d_model = 2048
num_heads = 2
num_layers = 2
vocab_size = len(chars)
n_epochs = 4
print_interval = 10


# 实例化模型
def train(model, criterion, optimizer, batch_size=4, device="cuda"):
    model.train()
    n_samples = len(input_seqs)
    n_batches = (n_samples + batch_size - 1) // batch_size

    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, n_samples)

            batch_input_seqs = input_seqs[start_idx:end_idx]
            batch_target_seqs = target_seqs[start_idx:end_idx]

            input_seq_tensor = torch.tensor(batch_input_seqs)
            target_seq_tensor = torch.tensor(batch_target_seqs)

            optimizer.zero_grad()
            input_seq_tensor = input_seq_tensor.to(device)
            target_seq_tensor = target_seq_tensor.to(device)
            output = model(input_seq_tensor)
            res = output.view(-1, output.size(-1))  # Reshape to [64, 15]
            target = target_seq_tensor.view(-1)
            loss = criterion(res, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / n_batches
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {average_loss}")


def get_init_model(dtype=torch.float16):
    model = TransformerTest(d_model, num_heads, num_layers, vocab_size, dtype=dtype)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    return model, criterion, optimizer


# 生成新的文本
def generate_text(model, seed_text, max_length=100, device="cuda"):
    model.eval()

    # with torch.no_grad():
    input_seq = [char_to_idx[char] for char in seed_text]
    input_seq_tensor = torch.tensor(input_seq).unsqueeze(0)
    input_seq_tensor = input_seq_tensor.to(device)

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
    # print("remain", remain)
    import os

    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '29500'
    # torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)
    model, criterion, optimizer = get_init_model(dtype=torch.bfloat16)
    # load_model(model)
    print("premodel")

    model = model.to("cuda")
    pre_name = list(model.named_modules())
    model_to_recompute_mode(model)
    # make_checkpointed(model)
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()
    # a = list(model.named_parameters())
    # now_name = list(model.named_modules())
    train(model, criterion, optimizer)
    end_time = time.time()
    print("use_time", end_time - start_time)
    res = generate_text(model, "这是")
    print("res_text", res)
    # save(model)
    # print("===============model\n", model.state_dict())`
