from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from transformers import AwqConfig, AutoConfig
import json


def get_json_data(path):
    with open(path, 'r') as file:
        data = json.load(file)

    output = []

    # 处理每个条目
    for item in data:
        prompt = item['prompt']
        answers = item['answer']
        # 将prompt与每个answer拼接
        for answer in answers:
            item_str = f"{prompt} {answer}"
            output.append(item_str[:512])
    return output


def make_awq_model(model_path, quant_path, quant_config):
    print("load data ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)



    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    json_data = get_json_data("/data/HappyChat/train_data/dpo_zh.json")

    print("load model to gpu ...")
    # Load model
    max_memory_mapping = {0: "38GB", 1: "44GB", 2: "80GB"}
    model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto",
                                               max_memory=max_memory_mapping)
    print("start quantize ...")
    # Quantize
    model.quantize(tokenizer, quant_config=quant_config, calib_data=json_data[:100])

    # modify the config.json file so that it is compatible with transformers integration
    quantization_config = AwqConfig(
        bits=quant_config["w_bit"],
        group_size=quant_config["q_group_size"],
        zero_point=quant_config["zero_point"],
        version=quant_config["version"].lower(),
    ).to_dict()

    model.model.config.quantization_config = quantization_config
    print("start save ...")
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)


if __name__ == "__main__":
    model_path = "/data2/llm3-70"
    quant_path = "/data/awq_llm3_70"
    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

    make_awq_model(model_path, quant_path, quant_config)

