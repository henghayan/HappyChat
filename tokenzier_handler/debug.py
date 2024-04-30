from transformers.models.llama.tokenization_llama import LlamaTokenizer
# from transformers.models.llama.modeling_llama import
import transformers
import sentencepiece as spm


def get_input_ids(base_model, test_text):
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    a = tokenizer(test_text)
    q = tokenizer("q")
    w = tokenizer("w")
    e = tokenizer("e")
    r = tokenizer("r")
    print(f"a:{a}/n")
    print(f"a:{q}/n")
    print(f"a:{w}/n")
    print(f"a:{e}/n")
    print(f"a:{r}/n")


def sp_debug(sp_model_path, test_text):
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(sp_model_path)
    print(sp_model.vocab_size())
    print(sp_model.EncodeAsPieces(test_text))


if __name__ == "__main__":
    test_text = "你好啊哈哈哈哈"
    sp_debug("/data/llama_token_cn/tokenizer.model", test_text)

    get_input_ids("/data/llama_token_cn/", test_text)
