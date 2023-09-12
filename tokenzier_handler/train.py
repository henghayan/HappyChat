import sentencepiece as spm
import time


def train_chinese_sp_model(input_file=None, input_text=None, model_prefix='chinese_sp', vocab_size=32000):
    if input_file:
        spm.SentencePieceTrainer.Train(
            f'--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size}'
        )
    elif input_text:
        with open("temp_chinese_data.txt", "w", encoding="utf-8") as f:
            f.write(input_text)

        spm.SentencePieceTrainer.Train(
            f'--input=temp_chinese_data.txt --model_prefix={model_prefix} --vocab_size={vocab_size}'
        )


def print_token(file_path):
    vocab = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            token, score = line.strip().split('\t')
            vocab[token] = float(score)

    # 打印词汇表
    for token, score in vocab.items():
        print(f"Token: {token}, Score: {score}")


if __name__ == "__main__":
    # data_path = "/data/HappyChat/train_data/token.txt"
    # res_path = "/data/tokenizer/test_sp"
    # start_time = time.time()
    # train_chinese_sp_model(input_file=data_path, model_prefix=res_path, vocab_size=10000)
    # end_time = time.time()
    # print("use time:", end_time-start_time)

    print_token("/data/tokenizer/test_sp.vocab")
