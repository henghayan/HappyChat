import requests
import json


class ModelAPI:
    def __init__(self):
        self.system_prompt = '''You are a query robot, looking for answers to relevant questions from the context, and your responses always follow the following principles:
1. json format
2, when there is a corresponding answer, use numbers + units
3, When there is no answer, reply ""
4. You don't focus on things that aren't relevant to the problem.
For example: "context" : "On April 20, 2024, there is a new 5km world record of 12 minutes.Li has 5 bags of apples, 4 in each bag",
"user": ["What is the 5km world record so far?", "How many apples does Li have?", "How many people live in the world today?"] ,
"assistant": {"What is the 5km world record so far?" : "12 minutes", "How many apples does Li have?" : "20", "":"How many people live in the world today?" : ""}'''

    def format_prompt(self, questions, contexts):
        prompts = []
        questions_str = str(questions)
        for context in contexts:
            prompt = {
                "system": self.system_prompt,
                "context": context,
                "user": questions_str
            }
            prompts.append(prompt)

        return prompts

    def call_model_api(self, prompts):
        url = "http://192.168.8.242:5000/generate"  # 替换为实际的API URL
        response = requests.post(url, json={"prompts": prompts})
        if response.status_code == 200:
            outputs = response.json().get("data", [])
            return outputs
        else:
            raise Exception(f"Model API call failed with status code {response.status_code}")


# 初始化一个ModelAPI实例
model_api = ModelAPI()
