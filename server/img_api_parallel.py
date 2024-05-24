import io
from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torch
import json

app = Flask(__name__)

# Load the model and tokenizer
model = AutoModel.from_pretrained('/data3/llm3-v', trust_remote_code=True, torch_dtype=torch.float16)
model = model.to(device='cuda')

tokenizer = AutoTokenizer.from_pretrained('/data3/llm3-v', trust_remote_code=True)
model.eval()


@app.route('/analyze_p', methods=['POST'])
def analyze_image():

    image_files = []
    for i in range(len(request.files)):
        image_key = "image" + str(i)
        image_handle = request.files[image_key]
        image = Image.open(io.BytesIO(image_handle.read())).convert('RGB')
        image_files.append(image)

    msgs = request.form['msgs']

    # Convert msgs from JSON string to Python list
    try:
        msgs = json.loads(msgs)
    except Exception as e:
        print("parse image api msgs json fail,", e)
        return jsonify({"error": "Invalid JSON for msgs"}), 400

    res = model.generate_batch(
        image_batch=image_files,
        msgs_batch=msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.1
    )

    return jsonify({"data": res})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
