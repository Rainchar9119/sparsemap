import nevergrad as ng
import inspect
from copy import deepcopy
import shutil, argparse, pprint, sys, subprocess
import numpy as np
import matplotlib.pyplot as plt
import math
from parse_timeloop_output import parse_timeloop_stats
from jinja2 import Template
from jinja2 import Environment, FileSystemLoader
import yaml
import os
import random
from datetime import datetime

##此文件用于将state中的mapping矩阵进行一维编解码的测试

import inspect
from copy import deepcopy
import shutil, argparse, pprint, sys, subprocess
import random
from datetime import datetime

OVERWRITE = 1


dimensions = {'C': 3, 'M': 96, 'N': 4, 'P': 54, 'Q': 54, 'R': 12, 'S': 12}

def prime_factorization(n):
    factors = []
    divisor = 2

    while n > 1:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 1

    return factors




cf = prime_factorization(dimensions['C'])
mf = prime_factorization(dimensions['M'])
nf = prime_factorization(dimensions['N'])
pf = prime_factorization(dimensions['P'])
qf = prime_factorization(dimensions['Q'])
rf = prime_factorization(dimensions['R'])
sf = prime_factorization(dimensions['S'])


factor_list = [cf,mf,nf,pf,qf,rf,sf]

lc = len(cf)
lm = len(mf)
lr = len(rf)
ls = len(sf)
ln = len(nf)
lp = len(pf)
lq = len(qf)

len_list = [lc,lm,ln,lp,lq,lr,ls]

mapping_encoding_len = lc+lm+lr+ls+ln+lp+lq


def map_decode(code):
    size = [[1,1,1,1,1,1,1]for _ in range(5)]
    start = 0
    for i in range(7):
        for j in range(len_list[i]):
            size[code[start+j]][i] *= factor_list[i][j]
        start = start + len_list[i]
    return size

'''
def map_encode(size):
    size_code = []
    for i in range(7):
        for j in range(len_list[i]):
            factor = size[i][j] // factor_list[i][j]  # 计算除法的商
            # 根据商的值来确定编码
            for k in range(factor):
                size_code.append(j)
    return size_code
'''
    
def map_encode(size):
    size_code = []
    for dim_index in range(7):
        factors_of_dim = factor_list[dim_index]
        for factor in factors_of_dim:
            for i in range(5):
                if size[i][dim_index]%factor == 0:
                    size[i][dim_index] = size[i][dim_index]//factor
                    size_code.append(i)
    return size_code





code = [4,2,3,2,1,4,1,2,0,0,1,3,4,0,2,1,2,1,2,3,1,2,1]
print('code = ',code)
mapping = map_decode(code)
print('mapping = ', mapping)
new_code = map_encode(mapping)
#print('encode result = ',map_encode(mapping))
print('mapping = ',map_decode(new_code))
