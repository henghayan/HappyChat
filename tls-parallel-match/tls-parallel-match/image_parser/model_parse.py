import requests


class ImageParser:
    def __init__(self):
        self.system_prompt = "You are an expert in image recognition, and you can output the text in pictures as it is, especially physical symbols and details such as line breaks."

    def parse_images(self, images):
        images_data = {}
        msgs = []
        for i in range(len(images)):
            image_key = "image" + str(i)
            images_data[image_key] = images[i]
            msgs.append({"role": "system", "content": self.system_prompt})

        url = "http://192.168.8.242:5000/analyze_p"  # 替换为实际的API URL
        response = requests.post(url, data={"msgs": msgs}, files=images_data)
        if response.status_code == 200:
            outputs = response.json().get("data", [])
            return outputs
        else:
            raise Exception(f"Model API call failed with status code {response.status_code}")


