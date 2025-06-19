
import math


def factor_count(n):
    """计算一个正整数最多被拆分成几个数"""
    count = 0
    i = 2  # 从最小的质数开始
    while i <= n:
        while n % i == 0:
            count += 1
            n /= i
        i += 1
    return count

def find_max_factors_in_range(n):
    """在0.9倍到1.1倍范围内找到能被拆分成最多份的正整数"""
    lower_bound = math.ceil(0.9 * n)
    upper_bound = math.floor(1.1 * n)
    
    max_count = 0
    result_number = 0
    
    for num in range(lower_bound, upper_bound + 1):
        count = factor_count(num)
        if count > max_count:
            max_count = count
            result_number = num
    
    return result_number, max_count

# 输入正整数
input_number = int(input("请输入一个正整数："))

# 调用函数
result_number, max_count = find_max_factors_in_range(input_number)

# 输出结果
print(f"在0.9倍到1.1倍范围内，能被拆分成最多份的正整数是 {result_number}，可以被拆分成 {max_count} 份。")
