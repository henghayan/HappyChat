from flask import Flask, request, jsonify
import transformers
import torch

app = Flask(__name__)

# 加载模型
model_path = "/data/llm3-70b"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)


@app.route('/generate', methods=['POST'])
def generate():
    # 解析输入数据
    input_data = request.json
    messages = input_data['messages']
    max_new_tokens = input_data.get('max_new_tokens', 128)
    do_sample = input_data.get('do_sample', False)
    temperature = input_data.get('temperature', 0.6)
    top_p = input_data.get('top_p', 0.9)

    # 创建prompt
    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 生成文本
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )

    # 返回生成的文本
    return jsonify({
        "generated_text": outputs[0]["generated_text"][len(prompt):]
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)


