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

def rearrange(input_list):
    # 深拷贝输入列表，以免修改原始列表
    original_list = input_list.copy()

    # 随机生成三个子集的长度
    len_a = random.randint(0, len(original_list))
    len_b = random.randint(0, len(original_list) - len_a)
    len_c = random.randint(0, len(original_list) - len_a - len_b)
    len_d = random.randint(0, len(original_list) - len_a - len_b - len_c)
    len_e = len(original_list) - len_a - len_b - len_c - len_d

    # 随机分配元素到三个子集中
    subset_a = random.sample(original_list, len_a)
    for elem in subset_a:
        original_list.remove(elem)

    subset_b = random.sample(original_list, len_b)
    for elem in subset_b:
        original_list.remove(elem)

    subset_c = random.sample(original_list, len_c)
    for elem in subset_c:
        original_list.remove(elem)

    subset_d = random.sample(original_list, len_d)
    for elem in subset_d:
        original_list.remove(elem)

    subset_e = original_list

    split = [1,1,1,1,1]
    if subset_a:
        for a_factor in subset_a:
            split[0] *= a_factor
    if subset_b:
        for b_factor in subset_b:
            split[1] *= b_factor
    if subset_c:
        for c_factor in subset_c:
            split[2] *= c_factor
    if subset_d:
        for d_factor in subset_d:
            split[0] *= d_factor
    if subset_e:
        for e_factor in subset_e:
            split[0] *= e_factor
    

    return split



def change_mapping(array):
    column_index = random.randint(0,6)                #随机选一个维度
    c = array[:,column_index]
    array[:,column_index] = rearrange(c.tolist())
    return array







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
