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
    l1,l2 = random.sample(range(5), 2)
    size1 = input_list[l1]
    size2 = input_list[l2]
    size = size1*size2
    factors = prime_factorization(size)
    len1 = random.randint(0, len(factors))
    subset1 = random.sample(factors,len1)
    for ele in subset1:
        factors.remove(ele)
    subset2 = factors
    new_l1 = 1
    new_l2 = 1
    if subset1:
        for e in subset1:
            new_l1 *= e
    if subset2:
        for e in subset2:
            new_l2 *= e
    input_list[l1],input_list[l2] = new_l1,new_l2
    return input_list
    





def change_mapping(array):
    column_index = random.randint(0,6)                #随机选一个维度
    c = array[:,column_index]
    array[:,column_index] = rearrange(c.tolist())
    
    return array


a = np.array([[1,1,1,1,3,1,2],[1,12,2,1,3,1,1],[1,8,2,3,1,1,2],[1,1,1,9,6,4,1],[3,1,1,2,1,3,3]])
print(change_mapping(a))