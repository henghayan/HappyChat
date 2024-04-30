import pandas as pd
import json


def parse_parquets_data_to_json(parquet_path, json_path=None):
    df = pd.read_parquet(parquet_path)
    json_str = df.to_json(orient='records', force_ascii=False)

    print(df.columns)
    print(df.head(2))
    # with open(json_path, 'w', encoding="utf-8") as f:
    #     f.write(json_str)


def read_json_and_concatenate_values(json_file_path, save_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    concatenated_values = []

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, str):
                        concatenated_values.append(value)
                    elif isinstance(value, list):
                        concatenated_values.append(" ".join(str(sub_item) for sub_item in value))
                    elif isinstance(value, dict):
                        concatenated_values.append(" ".join(str(v) for v in value.values()))
                    else:
                        concatenated_values.append(str(value))
            else:
                concatenated_values.append(str(item))
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                concatenated_values.append(value)
            elif isinstance(value, list):
                concatenated_values.append(" ".join(str(item) for item in value))
            elif isinstance(value, dict):
                concatenated_values.append(" ".join(str(v) for v in value.values()))
            else:
                concatenated_values.append(str(value))

    long_text = "".join(concatenated_values)
    # print(long_text[:10000])
    with open(save_path, 'w', encoding="utf-8") as f:
        f.write(long_text)
    return long_text


if __name__ == "__main__":
    # parse_parquets_data_to_json("/data/train_data/chinese.parquet", "/data/train_data/chinese.json")

    read_json_and_concatenate_values("/data/train_data/chinese.json", "/data/train_data/chinese.txt")
