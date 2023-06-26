from utils.prompter import Prompter
from datasets import load_dataset
from torch.utils.data import DataLoader


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


from torch.nn.utils.rnn import pad_sequence
from torch import Tensor


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    input_ids = pad_sequence([Tensor(item) for item in input_ids], batch_first=True).long()
    attention_mask = pad_sequence([Tensor(item) for item in attention_mask], batch_first=True).long()
    labels = pad_sequence([Tensor(item) for item in labels], batch_first=True).long()

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# def get_all_batch_data():
#     DataLoader
