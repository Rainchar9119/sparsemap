
import numpy as np
import matplotlib.pyplot as plt
import math
from parse_timeloop_output import parse_timeloop_stats
from jinja2 import Template
from jinja2 import Environment, FileSystemLoader
import yaml
import os
import inspect
from copy import deepcopy
import shutil, argparse, pprint, sys, subprocess

this_file = os.path.abspath(inspect.getfile(inspect.currentframe()))
directory = os.path.dirname(this_file)

def fill_population_with_best_individuals(population, file_path):
    # 加载保存的最优个体数据
    best_individuals = np.loadtxt(file_path)

    # 确保最优个体数据的维度和种群的最后一维一致
    if best_individuals.shape[1] == population.shape[1]:
        # 将最优个体填充到种群的后100行
        population[-100:] = best_individuals[-100:]
    else:
        print("维度不匹配，无法填充。")

# 示例使用
population = np.random.randint(low=0, high=100, size=(100, 63), dtype=np.int)
file_path = os.path.join(directory,"search_outputs","searching","outputs", "generation_best_individual.txt")
# 调用函数填充 population
fill_population_with_best_individuals(population, file_path)

# 打印结果
print(population)