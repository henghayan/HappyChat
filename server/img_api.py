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


@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files or 'msgs' not in request.form:
        return jsonify({"error": "Please provide an image and msgs"}), 400

    image_file = request.files['image']
    msgs = request.form['msgs']

    # Convert msgs from JSON string to Python list
    try:
        msgs = json.loads(msgs)
    except ValueError:
        return jsonify({"error": "Invalid JSON for msgs"}), 400

    # Read the image directly from the file object
    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')

    # Get the model response
    res = model.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.7
    )

    return jsonify(res)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
