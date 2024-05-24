import pandas as pd
import os

EXCEL_DIRECTORY = "question_template/"


def load_excel(app_key):
    # 加载Excel文件
    file_path = os.path.join(EXCEL_DIRECTORY, f"{app_key}.xls")
    df = pd.read_excel(file_path)
    return df


def get_questions_from_excel(app_key):
    df = load_excel(app_key)
    questions = df['item'].tolist()
    return questions


def generate_excel(json_data):
    # 接收一个json列表，生成Excel文件
    data = []
    for item in json_data:
        data.append({"item": item["item"], "value": item["value"]})

    df = pd.DataFrame(data)
    output_path = os.path.join(EXCEL_DIRECTORY, "generated_output.xlsx")
    df.to_excel(output_path, index=False)

    return {"message": "Excel file generated", "output_path": output_path}
