import os
import argparse
from transformers import AutoTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm


def load_tokenizer_and_sp_model(tokenizer_dir, sp_model_file):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(sp_model_file)
    return tokenizer, sp_model


def load_sentencepiece_proto(sp_model):
    spm_proto = sp_pb2_model.ModelProto()
    spm_proto.ParseFromString(sp_model.serialized_model_proto())
    return spm_proto


def merge_sentencepiece_models(base_spm, new_spm):
    base_tokens_set = set(p.piece for p in base_spm.pieces)
    for piece in new_spm.pieces:
        if piece.piece not in base_tokens_set:
            new_piece = sp_pb2_model.ModelProto.SentencePiece()
            new_piece.piece = piece.piece
            new_piece.score = piece.score
            base_spm.pieces.append(new_piece)


def save_merged_model(merged_spm, output_sp_dir, output_hf_dir):
    os.makedirs(output_sp_dir, exist_ok=True)
    with open(f"{output_sp_dir}/merged_sp.model", 'wb') as f:
        f.write(merged_spm.SerializeToString())

    new_tokenizer = AutoTokenizer(vocab_file=f"{output_sp_dir}/merged_sp.model")
    new_tokenizer.save_pretrained(output_hf_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_tokenizer_dir', required=True, type=str)
    parser.add_argument('--chinese_sp_model_file', default='./chinese_sp.model', type=str)
    args = parser.parse_args()

    base_tokenizer, chinese_sp_model = load_tokenizer_and_sp_model(args.base_tokenizer_dir, args.chinese_sp_model_file)
    base_spm = load_sentencepiece_proto(base_tokenizer.sp_model)
    chinese_spm = load_sentencepiece_proto(chinese_sp_model)

    merge_sentencepiece_models(base_spm, chinese_spm)

    output_sp_dir = 'merged_tokenizer_sp'
    output_hf_dir = 'merged_tokenizer_hf'
    save_merged_model(base_spm, output_sp_dir, output_hf_dir)

    print(f"Merged tokenizer has been saved to {output_hf_dir}")


if __name__ == "__main__":
    main()
