from flask import Blueprint, request, jsonify

import io

import sys

sys.path.append("/data/HappyChat/tls-parallel-match")

from image_parser.model_parse import ImageParser
from excel_operator.excel_utils import get_questions_from_excel, generate_excel
from app.parallel_handler import ParallelHandler

bp = Blueprint('api', __name__)


@bp.route('/data_parallel_match', methods=['POST'])
def query():
    data = request.get_json()
    questions = data.get('questions', [])
    contexts = data.get('contexts', [])

    if len(questions) == 0 or len(contexts) == 0:
        return 403, 'No questions or contexts provided'

    ph = ParallelHandler()
    response_data = ph.get_answers(questions, contexts)

    return jsonify(response_data)


@bp.route('/image_parallel_match', methods=['POST'])
def image_query():
    app_key = request.form['app_key']
    images = []
    files = request.files
    for i in range(len(files)):
        image_key = "image" + str(i)
        image = files[image_key].read()
        images.append(image)

    IP = ImageParser()
    contexts = IP.parse_images(images)

    questions = get_questions_from_excel(app_key)

    # ph = ParallelHandler()
    # response_data = ph.get_answers(questions, contexts)

    response_data = {"contexts": contexts}

    return jsonify(response_data)


@bp.route('/excel_match', methods=['POST'])
def excel_query():
    data = request.get_json()

    data = request.get_json()
    app_key = data.get('app_key', "")
    contexts = data.get('contexts', [])

    if app_key == "" or len(contexts) == 0:
        return 403, 'No app_key or contexts provided'
    # 获取预置问题
    questions = get_questions_from_excel(app_key)

    ph = ParallelHandler()
    response_data = ph.get_answers(questions, contexts)

    # # 生成结果Excel文件
    # excel_response = generate_excel(response_data)

    return jsonify(response_data)
