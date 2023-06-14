from utils.prompter import Prompter
from datasets import load_dataset


def data_format_func(tokenizer, cutoff_len=256, add_eos_token=True):
    prompter = Prompter()

    def data_format(data_point):
        full_prompt = prompter.generate_prompt(data_point["input"], data_point["output"])
        result = tokenizer(full_prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None)
        if result["input_ids"][-1] != tokenizer.eos_token_id and len(
                result["input_ids"]) < cutoff_len and add_eos_token:
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    return data_format


def load_train_data(data_path, tokenizer, cutoff_len=256):
    data = load_dataset('json', data_files=data_path)
    train_data = data["train"].map(data_format_func(tokenizer, cutoff_len))
    return train_data
