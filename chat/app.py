import os
import openai
from flask import Flask, request, jsonify, render_template, redirect, url_for, json, Response
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 替换为自己的密钥
SECRET_KEY = 'yahaha'
# openai.api_key = "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"

sk = os.environ.get("sk", None)
openai.api_key = sk

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# 使用字典存储用户数据和聊天记录
users = {
    'user1': {'password': 'password1', 'messages': []},
    'dwb': {'password': 'dwb', 'messages': []},
    'wc': {'password': 'wc', 'messages': []},
    'zcw': {'password': 'zcw', 'messages': []},
    'dy': {'password': 'dy', 'messages': []},
    'ww': {'password': 'ww', 'messages': []},
    'pj': {'password': 'pj', 'messages': []},
    'lj': {'password': 'lj', 'messages': []},
    'yb': {'password': 'yb', 'messages': []},
    'lyz': {'password': 'lyz', 'messages': []},
    'ljl': {'password': 'ljl', 'messages': []}
}


class User(UserMixin):
    def __init__(self, username, messages):
        self.id = username
        self.messages = messages


@login_manager.user_loader
def load_user(username):
    if username not in users:
        return None
    return User(username, users[username]['messages'])


@app.route('/login', methods=['GET'])
def login():
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/mychat')
@login_required
def index():
    return render_template('index.html')


@app.route('/api/login', methods=['POST'])
def api_login():
    response = {'status': 'error', 'message': '未知错误'}

    data = json.loads(request.data)
    username = data['username']
    password = data['password']

    if username not in users:
        response['message'] = '用户不存在，请联系管理员注册'
    elif users[username]['password'] != password:
        response['message'] = '密码不正确'
    else:
        user = User(username, users[username]['messages'])
        login_user(user)
        response['status'] = 'success'
        response.pop('message', None)

    return jsonify(response)


# @app.route('/api/ask', methods=['GET'])
# @login_required
# def ask():
#     user_message = request.args.get('message')
#
#     messages = current_user.messages + [
#         {"role": "system", "content": "You are a helpful AI assistant."},
#         {"role": "user", "content": user_message},
#     ]
#
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=messages,
#         max_tokens=500,
#         n=1,
#         stop=None,
#         temperature=0.5,
#         stream=True
#     )
#     res = ""
#     for i in response:
#         data = i["choices"][0]
#         if data['finish_reason'] is None:
#             res_token = data['delta'].get('content', None)
#             res += res_token if res_token else ""
#     current_user.messages.extend([{"role": "user", "content": user_message}, {"role": "assistant", "content": res}])
#     return jsonify(answer=res)


@app.route('/api/ask', methods=['GET'])
@login_required
def ask():
    user_message = request.args.get('message')

    current_user_messages = current_user.messages

    messages = [{"role": "system", "content": "You are a helpful AI assistant."}] + current_user_messages[-6:] + [{
                   "role": "user", "content": user_message}]

    response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo",
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=1,
        top_p=0.1,
        stream=True,
    )

    def generate():
        res = ""
        for i in response:
            data = i["choices"][0]
            if data['finish_reason'] is None:
                res_token = data['delta'].get('content', None)
                res += res_token if res_token else ""
                yield f"data: {res}\n\n"
        current_user_messages.extend([{"role": "user", "content": user_message}, {"role": "assistant", "content": res}])

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/history', methods=['GET'])
@login_required
def history():
    offset = int(request.args.get('offset', 0))
    limit = int(request.args.get('limit', 10))
    messages = current_user.messages[-(offset + limit):-offset] if offset > 0 else current_user.messages[-limit:]
    return jsonify(messages=messages)


@app.route('/api/add_user', methods=['GET'])
def add_user():
    secret_key = request.args.get('secret_key')
    if secret_key != SECRET_KEY:
        return jsonify({"error": "Access denied."}), 401

    username = request.args.get('username')
    password = request.args.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400

    # 这里添加创建用户的逻辑
    # 例如：create_user(username, password)
    if users.get(username, None) is not None:
        return jsonify({"error": "Username already exist."}), 400

    users[username] = {'password': password, 'messages': []}
    return jsonify({"success": f"User '{username}' added."})


@app.route('/api/list_users', methods=['GET'])
def list_users():
    secret_key = request.args.get('secret_key')
    if secret_key != SECRET_KEY:
        return jsonify({"error": "Access denied."}), 401

    # 这里添加获取用户列表的逻辑
    # 例如：users = get_users()

    users_data = [{"username": user_name, "password": users[user_name]['password'],
                   "message": get_user_question(users[user_name]['messages'])} for user_name in users.keys()]

    return jsonify({"users": users_data})


def get_user_question(messages):
    res = []
    i = len(messages)
    c = 0
    while i > 0:
        msg = messages[i - 1]
        if msg['role'] == 'user':
            res.append(msg)
            if c > 10:
                break
            c += 1
        i -= 1
    return res


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
