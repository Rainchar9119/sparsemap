import numpy as np
import random

def prime_factorization(n):
    factors = []
    divisor = 2

    while n > 1:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 1

    return factors

def split_into_three_subsets(input_list):
    # 深拷贝输入列表，以免修改原始列表
    original_list = input_list.copy()

    # 随机生成三个子集的长度
    len_a = random.randint(0, len(original_list))
    len_b = random.randint(0, len(original_list) - len_a)
    len_c = len(original_list) - len_a - len_b

    # 随机分配元素到三个子集中
    subset_a = random.sample(original_list, len_a)
    for elem in subset_a:
        original_list.remove(elem)

    subset_b = random.sample(original_list, len_b)
    for elem in subset_b:
        original_list.remove(elem)

    subset_c = original_list

    return subset_a, subset_b, subset_c


def generate_solutions(dimensions, X, Y, SP, permutation):
    factors = []
    
    while 1:
        solution = []                 #随机产生一个解
        for i in range(7):
            factors.append(prime_factorization(dimensions[permutation[i]]))
            a,b,c = split_into_three_subsets(factors[i])
            split = [1,1,1]
            if a:
                for a_factor in a:
                    split[0] *= a_factor
            if b:
                for b_factor in b:
                    split[1] *= b_factor
            if c:
                for c_factor in c:
                    split[2] *= c_factor
            solution.append(split)

        x_prod = 1
        y_prod = 1
        for i in range(SP):
            x_prod *= solution[i][1]
        for i in range(SP,7):
            y_prod *= solution[i][1]
        if (x_prod <= X) and (y_prod <= Y):
            break
    
    recover_solution = []                   #转换回去CMNPQRS顺序
    for i in range(7):
        recover_solution.append(solution[permutation.index(i)])

    return recover_solution




    
    

# 请替换下面的参数为你实际需要的值
C, M, N, P, Q, R, S = 3, 96, 4, 54, 54, 12, 12
dimensions =[C, M, N, P, Q, R, S]
X, Y = 8, 8
SP = 4
permutation = np.random.permutation(7)

solutions = generate_solutions(dimensions, X, Y, SP, permutation)
print(solutions)