# 定义加法函数 
def add():
    # 获取用户输入的两个数字并转为浮点型
    num1 = float(input("请输入第一个数字："))
    num2 = float(input("请输入第二个数字："))
    # 计算求和结果
    result = num1 + num2
    # 关键修改：让函数返回计算结果
    return result

# 调用函数并接收返回的结果
sum_result = add()
# 打印结果（现在使用的是外部的sum_result变量，而非函数内的局部变量）
print(f"输出结果：{sum_result}")
