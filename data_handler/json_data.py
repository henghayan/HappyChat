import os
import json


def read_data(data_path):
    with open(data_path, "r") as f:
        res = json.load(f)

    return res


def deliver_data(data_list, start_index, end_index, file_name=None):
    res = data_list[start_index: end_index]
    if file_name is not None:
        with open(file_name, "a+", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False))

    return res


if __name__ == "__main__":
    data = read_data("/data/train_data/guanaco_vicuna.json")
    res = deliver_data(data, 0, 1000, '/data/train_data/g_v_1000.json')
