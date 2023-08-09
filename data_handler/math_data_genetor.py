import random
import string

from json_data import deliver_data, merge_data


def generate_basic_samples(length):
    # 定义主体类
    subjects = ["孩子", "学生", "大人", "工人", "司机"]
    objects = [("糖果", "颗"), ("彩笔", "只"), ("玩具车", "辆"), ("书", "本"), ("铅笔", "支")]

    # 定义行为类
    actions = {
        "add": ["得到了", "加上了", "又找到了", "买到了", "收到了"],
        "sub": ["吃掉了", "丢失了", "送出了", "卖掉了", "损失了"],
        "mul": ["买了", "制造了", "包装了", "收集了", "画了"],
        "div": ["分给了", "送出了", "均分给了", "分摊给了", "平均给了"]
    }

    # 定义结果类
    results = ["总共有", "剩下", "一共是", "结果是", "变成了"]

    # 定义开始和结束的描述
    starts = ["开始有", "原本有", "手头有", "准备了", "刚好有"]
    ends = ["最后有", "剩下", "最终得到", "最后剩余", "最终拥有"]

    # 准备噪声
    noises = [" ", " ", " ", " ", "...", "?", "??", "请问", "能告诉我", "我想知道", "你知道", "我想知道", "能说说"]

    # 生成样本
    samples = []
    for _ in range(length):  # 生成1000个样本
        subject = random.choice(subjects)
        object, quantifier = random.choice(objects)
        num1 = random.randint(1, 200)
        num2 = random.randint(1, 20)  # 为了避免除法结果为小数，将第二个数限制在1到10之间

        noise = random.choice(noises)
        # 加法
        action = random.choice(actions["add"])
        result = random.choice(results)
        start = random.choice(starts)
        end = random.choice(ends)
        question = "{}{}有{}{}{}，{}{}{}{}，{}{}有多少{}{}{}?".format(subject, start, num1, quantifier, object, action,
                                                                    num2, quantifier, object, noise, subject, end,
                                                                    quantifier,
                                                                    object)
        answer = "{}+{}={}，所以{}{}有{}{}{}。".format(num1, num2, num1 + num2, subject, result, num1 + num2, quantifier,
                                                     object)
        samples.append({"input": question, "output": answer})

        # 减法
        action = random.choice(actions["sub"])
        result = random.choice(results)
        start = random.choice(starts)
        end = random.choice(ends)
        question = "{}{}有{}{}{}，{}{}{}{}，{}{}有多少{}{}{}?".format(subject, start, num1, quantifier, object, action,
                                                                    num2, quantifier, object, noise, subject, end,
                                                                    quantifier,
                                                                    object)
        answer = "{}-{}={}，所以{}{}有{}{}{}。".format(num1, num2, num1 - num2, subject, result, num1 - num2, quantifier,
                                                     object)
        samples.append({"input": question, "output": answer})

        # 乘法
        action = random.choice(actions["mul"])
        result = random.choice(results)
        start = random.choice(starts)
        end = random.choice(ends)
        question = "有{}{}，每个{}制造了{}{}{}，{}{}{}{}?".format(num1, subject, subject, num2,
                                                                quantifier, object, noise, end, quantifier, object)
        answer = "{}*{}={}，所以最后有{}{}{}。".format(num1, num2, num1 * num2, num1 * num2, quantifier, object)
        samples.append({"input": question, "output": answer})

        # 除法 (注意，我们要确保除数不为0，并处理结果为小数的情况)
        if num1 % num2 > 0:
            num2 = int((num2 // num1) * num1)
        question = "有{}{}{}，{}被{}个{}均分，{}每个{}可以得到多少{}{}?".format(num1, quantifier, object, object, num2,
                                                                              subject, noise, subject, quantifier,
                                                                              object)
        if num2 != 0:
            answer = "{}/{}={}，所以每个{}可以得到{}{}{}。".format(num1, num2, int(num1 / num2), subject,
                                                                 int(num1 / num2),
                                                                 quantifier, object)
        else:
            answer = "很抱歉，0无法作为除数，无法计算每个{}可以得到多少{}{}".format(subject, quantifier, object)
        samples.append({"input": question, "output": answer})

    return samples


def format_term(coefficient, variable):
    if coefficient == 0:
        return ""
    elif coefficient > 0:
        return f" + {coefficient if abs(coefficient) != 1 else ''}{variable}"
    else:
        return f" - {abs(coefficient) if abs(coefficient) != 1 else ''}{variable}"


def generate_equation_samples(num_problems, range_values=range(-10, 11)):
    problems = []
    for _ in range(num_problems):
        while True:
            val = random.choice(string.ascii_lowercase)
            a, c, e, g = random.choices(range_values, k=4)
            b = random.choice(range_values) if random.random() < 0.5 else 0
            d = random.choice(range_values) if random.random() < 0.5 else 0
            f = random.choice(range_values) if random.random() < 0.5 else 0
            h = random.choice(range_values) if random.random() < 0.5 else 0

            denominator = a + b - e - f
            if denominator == 0:
                continue

            solution = (g + h - c - d) / denominator
            if not solution.is_integer():
                continue

            left_side = get_sub_str(a, False, val) + get_sub_str(b, val=val) + get_sub_str(c) + get_sub_str(d)
            right_side = get_sub_str(e, False, val) + get_sub_str(f, val=val) + get_sub_str(g) + get_sub_str(h)

            problem = f"{left_side} = {right_side}"
            steps = []
            steps.append(f"步骤1: 识别方程，这是一个一元一次方程，未知数为 {val}。")
            simplified_left = get_sub_str(a + b, False, val) + get_sub_str(c + d)
            simplified_right = get_sub_str(e + f, False, val) + get_sub_str(g + h)
            if simplified_left != left_side or simplified_right != right_side:
                steps.append(
                    f"步骤2: 简化等式，简化等号两边，将方程 {problem} 简化为: {simplified_left} = {simplified_right}")
            new_left = get_sub_str(a + b - e - f, False, val)
            new_right = get_sub_str(g + h - c - d, False)
            if new_left != simplified_left or new_right != simplified_right:
                steps.append(
                    f"步骤3: 移项，将同类项移动到等式的同一边，得到{new_left[3:] if new_left.startswith(' + ') else new_left} = {new_right[3:] if new_right.startswith(' + ') else new_right}。")

            steps.extend([
                f"步骤4: 除以系数，将等式两边同时除以未知数的系数{int(a + b - e - f)}，得到 {val} = {int(g + h - c - d)}/{int(a + b - e - f)}，{val} = {int(solution)}",
                f"步骤5：验证，将{val} = {int(solution)}代入原方程式 {problem}，计算等式两边的值是否相等。"
            ])

            # 计算等式两边的值并比较是否相等
            left_value = a * int(solution) + b * int(solution) + c + d
            right_value = e * int(solution) + f * int(solution) + g + h

            left_side_check = get_sub_str(a, False, f"*({int(solution)})") + get_sub_str(b, val=f"*({int(solution)})") + get_sub_str(c) + get_sub_str(d) + f" = {left_value}"
            right_side_check = get_sub_str(e, False, f"*({int(solution)})") + get_sub_str(f, val=f"*({int(solution)})") + get_sub_str(g) + get_sub_str(h) + f" = {right_value}"
            steps.append(f"左边是 {left_side_check}，右边是{right_side_check}。")
            if left_value == right_value:
                steps.append(f"验证成功：{left_value} = {right_value}")
            else:
                steps.append(f"验证失败：{left_value} ≠ {right_value}，所以解是错误的。")

            problems.append({
                "input": f"解决以下一元一次方程：{problem}",
                "output": "\n".join(steps)
            })
            break

    return problems


def generate_equation_basic_samples(num_problems, range_values=range(-10, 11)):
    problems = []
    for _ in range(num_problems):
        while True:
            val = random.choice(string.ascii_lowercase)
            a, c, e, g = random.choices(range_values, k=4)
            b = random.choice(range_values) if random.random() < 0.5 else 0
            d = random.choice(range_values) if random.random() < 0.5 else 0
            f = random.choice(range_values) if random.random() < 0.5 else 0
            h = random.choice(range_values) if random.random() < 0.5 else 0

            denominator = a + b - e - f
            if denominator == 0:
                continue

            solution = (g + h - c - d) / denominator
            if not solution.is_integer():
                continue

            left_side = get_sub_str(a, False, val) + get_sub_str(b, val=val) + get_sub_str(c) + get_sub_str(d)
            right_side = get_sub_str(e, False, val) + get_sub_str(f, val=val) + get_sub_str(g) + get_sub_str(h)

            problem = f"{left_side} = {right_side}"
            steps = []
            simplified_left = get_sub_str(a + b, False, val) + get_sub_str(c + d)
            simplified_right = get_sub_str(e + f, False, val) + get_sub_str(g + h)
            if simplified_left != left_side or simplified_right != right_side:
                steps.append(f"{simplified_left} = {simplified_right}")
            new_left = get_sub_str(a + b - e - f, False, val)
            new_right = get_sub_str(g + h - c - d, False)
            if new_left != simplified_left or new_right != simplified_right:
                steps.append(
                    f"{new_left[3:] if new_left.startswith(' + ') else new_left} = {new_right[3:] if new_right.startswith(' + ') else new_right}")

            steps.extend([
                f"{val} = {int(g + h - c - d)}/{int(a + b - e - f)}，{val} = {int(solution)}",
                f"{val} = {int(solution)}"
            ])

            problems.append({
                "input": f"{problem}",
                "output": "\n".join(steps)
            })
            val2 = random.choice(string.ascii_lowercase)
            val3 = random.choice(string.ascii_lowercase)
            problems.append({
                "input": problems[-1]['input'].replace(val, val2),
                "output": problems[-1]['output'].replace(val, val2)
            })
            problems.append({
                "input": problems[-2]['input'].replace(val, val3),
                "output": problems[-2]['output'].replace(val, val3)
            })
            break

    return problems

def get_sub_str(lab, pre=True, val=""):
    if lab == 0:
        if pre:
            return ""
        else:
            return "0"
    if pre:
        if lab > 0:
            return f"+{lab}{val}"
        else:
            return f"{lab}{val}"
    else:

        return f"{lab}{val}"


if __name__ == "__main__":
    basic_samples = 20
    math_samples = generate_basic_samples(basic_samples)
    # for sample in math_samples:
    #     print("Q: ", sample['input'])
    #     print("A: ", sample['output'])

    # 使用函数生成样本
    equation_samples = generate_equation_basic_samples(basic_samples*100)
    # print(equation_samples)
    #
    merge_data(math_samples, equation_samples, "/data/train_data/basic_math.json")
