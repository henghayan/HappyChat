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


def merge_data(data_list1, data_list2, file_name=None):
    if len(data_list1) > len(data_list2):
        long_data = data_list1
        short_data = data_list2
    else:
        long_data = data_list2
        short_data = data_list1

    per_gap = len(long_data) // len(short_data)
    i = 0
    res_data = []
    while i * per_gap < len(long_data):
        res_data.extend(long_data[i * per_gap:(i + 1) * per_gap])
        res_data.extend(short_data[i:i + 1])
        i += 1

    if file_name is not None:
        with open(file_name, "a+", encoding="utf-8") as f:
            f.write(json.dumps(res_data, ensure_ascii=False))
    return res_data

if __name__ == "__main__":
    # end_index = 3000
    # data = read_data("/data/train_data/guanaco_vicuna.json")
    #
    # res = deliver_data(data, 0, end_index, '/data/train_data/g_v_%s.json' % end_index)

    # #
    data1 = read_data("/data/train_data/g_v_3000.json")
    data2 = read_data("/data/HappyChat/train_data/t2.json")
    res = merge_data(data1, data2, '/data/train_data/g_v_3000_m.json')