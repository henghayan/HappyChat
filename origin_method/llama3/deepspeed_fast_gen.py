import time

from flask import Flask, request, jsonify
import transformers
import torch
import mii

app = Flask(__name__)

# 加载模型
model_path = "/data2/llm3-8-half"
# pipe = mii.pipeline(model_path, quantization_mode='wf6af16')


# client = mii.serve(model_path)
client = mii.serve(model_path, tensor_parallel=2)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
#
@app.route('/generate', methods=['POST'])
def generate():
    # 解析输入数据
    input_data = request.json
    messages = input_data['messages']
    max_new_tokens = input_data.get('max_new_tokens', 500)
    do_sample = input_data.get('do_sample', False)
    temperature = input_data.get('temperature', 0.6)
    top_p = input_data.get('top_p', 0.9)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    time_start = time.time()
    print("----load_success------\n")
    response = client.generate(prompt, max_new_tokens=500)
    print("use_time:%s" % (time.time() - time_start))
    print("res\n\n\n", response)
    # 返回生成的文本
    return jsonify({
        "generated_text": response
    })

#
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)




