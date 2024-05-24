import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import time

model = AutoModel.from_pretrained('/data3/llm3-v', trust_remote_code=True, torch_dtype=torch.float16)
model = model.to(device='cuda')

tokenizer = AutoTokenizer.from_pretrained('/data3/llm3-v', trust_remote_code=True)
model.eval()

image = Image.open('test1.jpg').convert('RGB')
question = '这是一个pcb的参数需求表，你能告诉我客户需要的板厚是多少吗?'
msgs = [{'role': 'user', 'content': question}]

time_start = time.time()

res = model.generate_batch(
    image_batch=[image for i in range(2)],
    msgs_batch=[msgs for i in range(2)],
    tokenizer=tokenizer,
    sampling=True,
    temperature=1
)
print(res)
print("use time", time.time() - time_start)

# res = model.chat(
#     image=[image, image],
#     msgs=[msgs, msgs],
#     tokenizer=tokenizer,
#     sampling=True,
#     temperature=0.7
# )
