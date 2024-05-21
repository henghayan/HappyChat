from flask import Flask, request, jsonify
import transformers
import torch

from vllm import LLM, SamplingParams

app = Flask(__name__)


def create_app_with_llm(llm_client, eos_token_ids):
    @app.route('/generate', methods=['POST'])
    def generate():
        # 解析输入数据
        req_data = request.json
        prompts = req_data['prompts']
        template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>%s<|eot_id|><|start_header_id|>context<|end_header_id|>%s<|eot_id|><|start_header_id|>user<|end_header_id|>%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

        input_data = []
        for p in prompts:
            sys_prompt = p.get("system", "你是一个查询机器人，总是从user给出的资料中获取答案")
            context = p.get("context", "")
            user_prompt = p.get("user", None)
            input_data.append(template % (sys_prompt, context, user_prompt))

        max_new_tokens = req_data.get('max_new_tokens', 256)
        do_sample = req_data.get('do_sample', False)
        temperature = req_data.get('temperature', 1)
        top_p = req_data.get('top_p', 0.9)
        sampling_params = SamplingParams(temperature=temperature,
                                         top_p=top_p,
                                         max_tokens=max_new_tokens,
                                         stop_token_ids=eos_token_ids)
        # for eos_token_id in eos_token_ids:
        #     sampling_params.all_stop_token_ids.add(eos_token_id)
        # 创建prompt
        outputs = llm_client.generate(input_data, sampling_params)
        print("input_data", input_data)
        print("outputs", outputs)
        res = []
        for output in outputs:
            res.append(output.outputs[0].text)
        # 返回生成的文本
        return jsonify({
            "data": res
        })

    return app


def main():
    model_path = "/data/awq_llm3_70"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    llm = LLM(model=model_path, quantization="AWQ", tensor_parallel_size=2)
    eos_token_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    app_with_llm = create_app_with_llm(llm, eos_token_ids)
    app_with_llm.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == '__main__':
    main()
