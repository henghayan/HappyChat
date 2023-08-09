import random
import string

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

            left_side = format_term(a, val) + format_term(b, val) + format_term(c, "") + format_term(d, "")
            right_side = format_term(e, val) + format_term(f, val) + format_term(g, "") + format_term(h, "")

            problem = f"{left_side[3:] if left_side.startswith(' + ') else left_side} = {right_side[3:] if right_side.startswith(' + ') else right_side}"

            steps = []
            simplified_left = format_term(a + b, val) + format_term(c + d, "")
            simplified_right = format_term(e + f, val) + format_term(g + h, "")
            if simplified_left != left_side or simplified_right != right_side:
                steps.append(
                    f"理解并转化方程：将方程 {problem} 简化为 {simplified_left[3:] if simplified_left.startswith(' + ') else simplified_left} = {simplified_right[3:] if simplified_right.startswith(' + ') else simplified_right}。")

            new_left = format_term(a + b - e - f, val)
            new_right = format_term(g + h - c - d, "")

            problem = f"{left_side[3:] if left_side.startswith(' + ') else left_side} = {right_side[3:] if right_side.startswith(' + ') else right_side}"

            steps = []
            steps.append(f"步骤1：识别方程的类型和未知数：这是一个一元一次方程，未知数为 {val}。")

            simplified_left = format_term(a + b, val) + format_term(c + d, "")
            simplified_right = format_term(e + f, val) + format_term(g + h, "")
            if simplified_left != left_side or simplified_right != right_side:
                steps.append(
                    f"步骤2：整理方程式：将方程 {problem} 简化为 {simplified_left[3:] if simplified_left.startswith(' + ') else simplified_left} = {simplified_right[3:] if simplified_right.startswith(' + ') else simplified_right}。")

            new_left = format_term(a + b - e - f, val)
            new_right = format_term(g + h - c - d, "")
            if new_left != simplified_left or new_right != simplified_right:
                steps.append(
                    f"步骤3：合并类似项：将未知数的系数相加，将常数项相加，得到 {new_left[3:] if new_left.startswith(' + ') else new_left} = {new_right[3:] if new_right.startswith(' + ') else new_right}。")

            steps.extend([
                f"步骤4：化简方程：通过适当的运算，将方程式进一步简化，使未知数的系数变为 1 或 -1。",
                f"步骤5：解得未知数：通过将方程两边同时进行相同的运算，求出未知数 {val} 的具体数值为 {int(solution)}。",
                f"步骤6：验证解的正确性：将求得的未知数 {val} 代入原方程式 {problem}，计算等式两边的值并比较是否相等。"
            ])

            # 计算等式两边的值并比较是否相等
            left_value = a + b * int(solution) + c + d * int(solution)
            right_value = e + f * int(solution) + g + h * int(solution)
            if left_value == right_value:
                steps.append(f"验证成功：{left_value} = {right_value}，所以解是正确的。")
            else:
                steps.append(f"验证失败：{left_value} ≠ {right_value}，所以解是错误的。")

            problems.append({
                "input": f"解决以下一元一次方程：{problem}",
                "output": "\n".join(steps)
            })
            break

    return problems




if __name__ == "__main__":
    basic_samples = 1
    math_samples = generate_equation_samples(basic_samples)
    # for sample in math_samples:
    #     print("Q: ", sample['input'])
    #     print("A: ", sample['output'])
    print(math_samples)
