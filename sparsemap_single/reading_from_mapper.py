import yaml, inspect, os, sys, subprocess, pprint, shutil, argparse
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import csv

from parse_timeloop_output import parse_timeloop_stats
import numpy as np
import random
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
import random
from datetime import datetime

#五层存储的ES  此版本暂时只考虑mapping

OVERWRITE = 1
JOB = "resnet_conv"              #高层次
WORK_LOAD = "workload_resnet_conv4"        #低层次
N = 10                             #要多少个个体
SEARCH_SIZE = 20000
OBJECT = 'delay'
#ALGO = 'random-pruned'
ALGO = 'random'



#这个程序把初始化探索的合法mapping写到csv文件中，供ES初始化种群使用

this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
this_directory = os.path.dirname(this_file_path)
#print("this_directory = ",this_directory)
problem_template_path = os.path.join(this_directory, "yamls", JOB , WORK_LOAD + ".yaml")
#print("problem_template_path = ",problem_template_path)
#arch_path = os.path.join(this_directory,"sparsemap_single", "yamls", "DSTC-RF2x-24-bandwidth.yaml")
arch_path = os.path.join(this_directory, "yamls", "arch_edge.yaml")
#component_path = os.path.join(this_directory, "..", "multiSCNN","fig13_dstc_setup","input_specs",  "compound_components.yaml")
#mapping_path = os.path.join(this_directory, "yamls","mapping_example.yaml")
mapper_path = os.path.join(this_directory, "yamls", "mapper.yaml")
constraints_path = os.path.join(this_directory, "yamls", "constraints.yaml")
sparse_opt_path = os.path.join(this_directory, "yamls", "sparse_opt_example.yaml")
#sparse_opt_path = os.path.join(this_directory, "..", "multiSCNN","single_core_optimization",  "yaml_gen", "sparse_opt_output.yaml")

reading_path = "/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single/outputs/example/outputs/timeloop-mapper.map.yaml"


dimensions = {'C': 3, 'M': 96, 'N': 4, 'P': 54, 'Q': 54, 'R': 12, 'S': 12}

with open(problem_template_path, 'r') as file:
        workload_data = yaml.safe_load(file)
        d = workload_data['problem']['instance']
        dimensions = d
print(dimensions)

##质因数分解
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
factor_list = [cf,mf,nf,pf,qf,rf,sf]                    #质因数汇总

lc = len(cf)
lm = len(mf)
lr = len(rf)
ls = len(sf)
ln = len(nf)
lp = len(pf)
lq = len(qf)
len_list = [lc,lm,ln,lp,lq,lr,ls]                       #每个维度的质因数长度

mapping_encoding_len = lc+lm+lr+ls+ln+lp+lq

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

def map_decode(code):
    size = [[1,1,1,1,1,1,1]for _ in range(5)]
    start = 0
    for i in range(7):
        for j in range(len_list[i]):
            size[code[start+j]][i] *= factor_list[i][j]
        start = start + len_list[i]
    return size

def cantor_encode(permutation):
    n = len(permutation)
    encoded = 0
    factorial = math.factorial(n-1)
    for i in range(n):
        count = 0
        for j in range(i+1, n):
            if permutation[j] < permutation[i]:
                count += 1
        encoded += count * factorial
        if i < n-1:
            factorial //= (n-i-1)
    return encoded + 1  # 加一是因为康托展开从1开始计数

def cantor_decode(encoded, n):
    permutation = []
    factorial = math.factorial(n-1)
    encoded -= 1  # 康托展开从1开始计数，这里要减一
    available = list(range(n))
    for i in range(n):
        index = encoded // factorial
        encoded %= factorial
        permutation.append(available[index])
        available.pop(index)
        if i < n-1:
            factorial //= (n-i-1)
    return permutation

def run_timeloop_(job_name, input_dict, base_dir, ert_path, art_path):
    output_dir = os.path.join(base_dir +"/outputs")
   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        if not OVERWRITE:
            print("Found existing results: ", output_dir)
            return
        else:
            print("Found and overwrite existing results: ", output_dir)

    # reuse generated ERT and ART files
    shutil.copy(ert_path, os.path.join(base_dir, "ERT.yaml"))
    shutil.copy(art_path, os.path.join(base_dir, "ART.yaml"))
    
    input_file_path = os.path.join(base_dir, "aggregated_input.yaml")
    ert_file_path = os.path.join(base_dir, "ERT.yaml")
    art_file_path = os.path.join(base_dir, "ART.yaml")
    logfile_path = os.path.join(output_dir, "timeloop.log")
    
    yaml.dump(input_dict, open(input_file_path, "w"), default_flow_style=False)
    os.chdir(output_dir)
    #subprocess_cmd = ["timeloop-model", input_file_path, ert_path, art_path]
    subprocess_cmd = ["timeloop-mapper", input_file_path, ert_path, art_path]
    print("\tRunning test: ", job_name)

    p = subprocess.Popen(subprocess_cmd)
    p.communicate(timeout=None) 

#将mapper搜索结果的yaml转换成中间形态state
def convert_yaml_to_state(yaml_path):

    state = {}
    with open(yaml_path, 'r') as file:
        mapping_data = yaml.safe_load(file)

    arr = [[0 for _ in range(7)] for _ in range(5)]
    perm = [[] for _ in range(5)]
    sp = [0,0]
    pb = [0,0]

    # 定义列索引顺序
    columns = ['C', 'M', 'N', 'P', 'Q', 'R', 'S']

    # 遍历解析后的数据
    for entry in mapping_data['mapping']:
        target = entry['target']
        type = entry['type']
        factors = entry.get('factors', None)
        permutation_values = entry.get('permutation', None)

        
        # 确定填充位置
        if target == 'DRAM' and type == 'temporal':
            row = 0
        elif target == 'GlobelBuffer' and type == 'temporal':
            row = 1
        elif target == 'GlobelBuffer' and type == 'spatial':
            row = 2
            sp[0] = entry['split']
        elif target == 'PE_buffer' and type == 'temporal':
            row = 3
        elif target == 'PE_buffer' and type == 'spatial':
            row = 4
            sp[1] = entry['split']
        elif target == 'GlobelBuffer' and type == 'datatype':
            iwo = entry['bypass']
            if 'Inputs' in iwo:
                pb[0]+=4
            if 'Weights' in iwo:
                pb[0]+=2
            if 'Outputs' in iwo:
                pb[0]+=1
        elif target == 'PE_buffer' and type == 'datatype':
            iwo = entry['bypass']
            if 'Inputs' in iwo:
                pb[1]+=4
            if 'Weights' in iwo:
                pb[1]+=2
            if 'Outputs' in iwo:
                pb[1]+=1
        else:
            continue  # 跳过其他类型的entry
        
        # 提取factors中的值，并填充到对应位置
        if factors:
            factors_values = factors.split()
            for factor_value in factors_values:
                factor = factor_value[0]  # 提取字母，例如'C'
                number = int(factor_value[1:])  # 提取数字，例如3
                if factor in columns:  # 判断字母是否在列索引顺序中
                    col = columns.index(factor)  # 获取字母在列表中的索引作为列索引
                    arr[row][col] = number

        if permutation_values:
            permutation_values = list(permutation_values)
            for perm_value in permutation_values:
                if perm_value in columns:  # 判断字符是否在列索引顺序中
                    perm[row].append(columns.index(perm_value))

    state['permutations'] = perm
    state['bypass_choice'] = pb
    state['array'] = arr
    state['split'] = sp

    return state

#运行一次mapper搜索，生成一个DNA的74位往后的编码段
def run_mapper_once(ob):
    ert_path = os.path.join(this_directory, "..","..", "evaluation_setups","fig13_dstc_setup","input_specs",  "ERT.yaml")
    art_path = os.path.join(this_directory, "..","..", "evaluation_setups","fig13_dstc_setup","input_specs",  "ART.yaml")
    #print(ert_path)
    
    # load all yaml input specs
    problem_template = yaml.load(open(problem_template_path), Loader = yaml.SafeLoader)
    arch = yaml.load(open(arch_path), Loader = yaml.SafeLoader)
    #components = yaml.load(open(component_path), Loader = yaml.SafeLoader)
    #mapping = yaml.load(open(mapping_path), Loader = yaml.SafeLoader)
    mapper = yaml.load(open(mapper_path), Loader = yaml.SafeLoader)
    constraints = yaml.load(open(constraints_path), Loader = yaml.SafeLoader)
    #sparse_opt = yaml.load(open(sparse_opt_path), Loader = yaml.SafeLoader)
    
    output_base_dir = os.path.join(this_directory, "outputs")

    new_problem = deepcopy(problem_template)

    #new_problem["problem"]["instance"]["densities"]["A"]["density"] = 0.55
    #new_problem["problem"]["instance"]["densities"]["B"]["density"] = 0.35

    aggregated_input = {}
    aggregated_input.update(arch)
    aggregated_input.update(new_problem)
    #aggregated_input.update(components)
    #aggregated_input.update(mapping)
    aggregated_input.update(mapper)
    aggregated_input.update(constraints)
    #aggregated_input.update(sparse_opt)
    job_name  = "example"
    run_timeloop_(job_name, aggregated_input, os.path.join(output_base_dir, job_name), ert_path, art_path)
    xml_path = os.path.join("/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single/outputs/example/outputs/timeloop-mapper.map+stats.xml")
    stat_type = "cycles"
    '''
    'problem': problem,
    'utilization': arithmetic_utilization,
    'cycles': max_cycles,
    'energy_pJ': energy_pJ,
    'energy_per_mac': energy_pJ/macs,
    'macs': macs,
    'energy_breakdown_pJ': energy_breakdown_pJ,
    'bandwidth_and_cycles': bandwidth_and_cycles

    '''
    
    if os.path.exists(xml_path):
        best_performance = parse_timeloop_stats(xml_path)
        perfo = best_performance[ob]
        state = convert_yaml_to_state(reading_path)
        os.remove(reading_path)
        os.remove(xml_path)
        return state,perfo
    else:
        return 0

#运行n次mapper，如果有
def run_mapper_manytimes(n,total_serchsize,object,algo):
    all_states= []
    all_perfo = []
    dict = {'delay':'cycles', 'energy':'energy_pJ'}
    for i in range(n):
        print('============================================================================================== {} st mapper ================================================='.format(i))
        with open(mapper_path, 'r') as file:
            data = yaml.safe_load(file)
            # 修改数据
        data['mapper']['search-size'] = total_serchsize        
        data['mapper']['optimization-metric'] = object
        data['mapper']['algorithm'] = algo
        data['mapper']['num-threads'] = (12*i//n)+1
        # 将修改后的数据写回原文件
        with open(mapper_path, 'w') as file:
            yaml.dump(data, file)
        s,perfo = run_mapper_once(dict[object])
        all_states.append(s)
        all_perfo.append(perfo)
    return all_states,all_perfo


def write_to_csv(all_states,all_perfo,csv_path):    #这里面包含cantor和mapsize code
    valid_mapping_num = len(all_states)
    with open(csv_path, 'w', newline='') as csvfile:
    # 创建 CSV writer 对象，指定分隔符为逗号
        csvwriter = csv.writer(csvfile, delimiter=',')
    # 写入数据
        for i in range(valid_mapping_num):
            state = all_states[i]
            cantor = [cantor_encode(per) for per in state['permutations']]
            mapcode = map_encode(state['array'])
            #print('state = ',state)
            line = [all_perfo[i]]
            line.extend(state['split'])
            line.extend(state['bypass_choice'])
            line.extend(cantor)
            line.extend(mapcode)
            #print('line = ',line)
            csvwriter.writerow(line)  # 写入表头
        

def main():
    
    all_states,all_perfo = run_mapper_manytimes(N,SEARCH_SIZE,OBJECT,ALGO)
    csv_path = os.path.join("/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single", JOB + "_mapping_found", WORK_LOAD + "_initialize.csv" )
    if not os.path.exists(csv_path):
    # 创建路径
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_to_csv(all_states,all_perfo,csv_path)


if __name__ == "__main__":
    main()