from transformers.models.llama.tokenization_llama import LlamaTokenizer
# from transformers.models.llama.modeling_llama import
import transformers


def get_input_ids(base_model):
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    a = tokenizer("你好吗啊哈哈哈哈")
    q = tokenizer("q")
    w = tokenizer("w")
    e = tokenizer("e")
    r = tokenizer("r")
    print(f"a:{a}/n")
    print(f"a:{q}/n")
    print(f"a:{w}/n")
    print(f"a:{e}/n")
    print(f"a:{r}/n")


if __name__ == "__main__":
    get_input_ids("/data/llm2/")
