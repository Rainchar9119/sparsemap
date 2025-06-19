import yaml, inspect, os, sys, subprocess, pprint, shutil, argparse
from copy import deepcopy
import matplotlib.pyplot as plt
import csv
from parse_timeloop_output import parse_timeloop_stats
import numpy as np
import random
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
from datetime import datetime
import pandas as pd

#五层存储的ES  此版本暂时只考虑mapping

OVERWRITE = 1
JOB = "resnet_conv"              #高层次
WORK_LOAD = "workload_resnet_conv3"        #低层次
N = 11                             #要多少个个体
SEARCH_SIZE = 20000

OBJECT = 'cycles'       #优化目标
OBJECT = 'edp'       #优化目标
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
#ALGO = 'random-pruned'
ALGO = 'random'
#PLATFORM = "edge"
PLATFORM = "mobile"
#PLATFORM = "cloud"
WITH_CONSTRAINTS = 1

N_GENERATIONS = 20
POP_SIZE = 100           # population size
N_KID = 50               # n kids per generation



this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
this_directory = os.path.dirname(this_file_path)
#print("this_directory = ",this_directory)
problem_template_path = os.path.join(this_directory, "yamls", JOB , WORK_LOAD + ".yaml")
constraints_template_path = os.path.join(this_directory, "yamls", "constraints_template.j2")

#print("problem_template_path = ",problem_template_path)
#arch_path = os.path.join(this_directory,"sparsemap_single", "yamls", "DSTC-RF2x-24-bandwidth.yaml")
if PLATFORM == "cloud": 
    arch_path = os.path.join(this_directory, "yamls", "arch_cloud.yaml")
elif PLATFORM == "mobile": 
    arch_path = os.path.join(this_directory, "yamls", "arch_mobile.yaml")
else:
    arch_path = os.path.join(this_directory, "yamls", "arch_edge.yaml")
#component_path = os.path.join(this_directory, "..", "multiSCNN","fig13_dstc_setup","input_specs",  "compound_components.yaml")
#mapping_path = os.path.join(this_directory, "yamls","mapping_example.yaml")
mapper_path = os.path.join(this_directory, "yamls", "mapper.yaml")
constraints_path = os.path.join(this_directory, "yamls", "constraints.yaml")
sparse_opt_path = os.path.join(this_directory, "yamls", "sparse_opt_example_conv.yaml")
#sparse_opt_path = os.path.join(this_directory, "..", "multiSCNN","single_core_optimization",  "yaml_gen", "sparse_opt_output.yaml")

reading_path = "/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single/outputs/example/outputs/timeloop-mapper.map.yaml"

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
    lower_bound = max(1,math.ceil(0.9 * n))
    upper_bound = math.floor(1.1 * n)
    
    max_count = 0
    result_number = 0
    
    for num in range(lower_bound, upper_bound + 1):
        count = factor_count(num)
        if count > max_count:
            max_count = count
            result_number = num
    if result_number == 0:
        return n
    return result_number

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

def correct_dimensions(prob_path):     #Dimension 估算修正
    dimensions = {'C': 0, 'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0}
    with open(prob_path, 'r') as file:
        workload_data = yaml.safe_load(file)
        d = workload_data['problem']['instance']
        dimensions = d
        print(d)
        dimensions['C'] = find_max_factors_in_range(dimensions['C'])      #新的维度值         c_factors是质因数个数
        dimensions['M'] = find_max_factors_in_range(dimensions['M'])
        print("N = ",dimensions['N'])
        dimensions['N'] = find_max_factors_in_range(dimensions['N'])
        print("N = ",dimensions['N'])
        dimensions['P'] = find_max_factors_in_range(dimensions['P'])
        dimensions['Q'] = find_max_factors_in_range(dimensions['Q'])
        dimensions['R'] = find_max_factors_in_range(dimensions['R'])
        dimensions['S'] = find_max_factors_in_range(dimensions['S'])


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

    workload_data['problem']['instance'] = dimensions           #字典更新
    with open(prob_path, 'w') as file:
            yaml.dump(workload_data, file)

    return dimensions, factor_list, len_list, mapping_encoding_len

#辗转相除法求最大公约数
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

def generate_constraints(dimensions,factor_list):
    d_order = ['C','M','N','P','Q','R','S']
    if PLATFORM == "cloud":
        x1 = 32
        y1 = 32
        x2 = 32
        y2 = 2
    elif PLATFORM =="mobile":
        x1 = 16
        y1 = 32
        x2 = 32
        y2 = 2
    else:
        x1 = 16
        y1 = 8
        x2 = 1
        y2 = 2
    '''
    current_max_gcd = 0
    current_max_d = '-'
    #print(dimensions)
    for k in d_order:
        if gcd(dimensions[k],x1) > current_max_gcd:
            current_max_d = k
            current_max_gcd = gcd(dimensions[k],x1)
    dimensions[current_max_d] = dimensions[current_max_d] // current_max_gcd
    #print(dimensions)
    str_GB_spatial = current_max_d + "=" + str(current_max_gcd)

    current_max_gcd = 0
    current_max_d = '-'
    for k in d_order:
        if gcd(dimensions[k],y1) > current_max_gcd:
            current_max_d = k
            current_max_gcd = gcd(dimensions[k],y1)
    dimensions[current_max_d] = dimensions[current_max_d] // current_max_gcd
    #print(dimensions)
    str_GB_spatial = str_GB_spatial + " " + current_max_d + "=" + str(current_max_gcd)
    '''
    current_max_gcd = 0
    current_max_d = '-'
    if PLATFORM == "edge":
        for k in d_order:
            if gcd(dimensions[k],x1) > current_max_gcd:
                current_max_d = k
                current_max_gcd = gcd(dimensions[k],y1)
        dimensions[current_max_d] = dimensions[current_max_d] // current_max_gcd
        #print("GB spatial", current_max_gcd)
        str_GB_spatial = current_max_d + "=" + str(current_max_gcd)
    else:
        for k in d_order:
            if gcd(dimensions[k],x2) > current_max_gcd:
                current_max_d = k
                current_max_gcd = gcd(dimensions[k],x2)
        dimensions[current_max_d] = dimensions[current_max_d] // current_max_gcd
        #print("PEB spatial", current_max_gcd)
        str_PEB_spatial = current_max_d + "=" + str(current_max_gcd)

    keeplist_GB = []
    keeplist_PEB = []
   
    with open(constraints_template_path, 'r') as template_file:
        template_content = template_file.read()
    
    # 创建 Jinja2 模板对象
    template = Template(template_content)
    if PLATFORM == "edge":
        data = {
        'targets': [
            {'target': 'GlobelBuffer', 'type': 'bypass', 'bypass': [], 'keep': keeplist_GB},
            {'target': 'PE_buffer', 'type': 'bypass', 'bypass': [], 'keep': keeplist_PEB},
            {'target': 'GlobelBuffer', 'type': 'spatial', 'factors': str_GB_spatial},
            #{'target': 'PE_buffer', 'type': 'spatial', 'factors': str_PEB_spatial}
        ]
        }
    else:
        data = {
        'targets': [
            {'target': 'GlobelBuffer', 'type': 'bypass', 'bypass': [], 'keep': keeplist_GB},
            {'target': 'PE_buffer', 'type': 'bypass', 'bypass': [], 'keep': keeplist_PEB},
            #{'target': 'GlobelBuffer', 'type': 'spatial', 'factors': str_GB_spatial},
            {'target': 'PE_buffer', 'type': 'spatial', 'factors': str_PEB_spatial}
        ]
        }
    # 渲染模板并生成 YAML 内容
    output = template.render(data)

    # 将渲染后的内容写入文件
    with open(constraints_path, 'w') as file:
        file.write(output)
  
def map_encode(size,factor_list):
    size_code = []
    for dim_index in range(7):
        factors_of_dim = factor_list[dim_index]
        for factor in factors_of_dim:
            for i in range(5):
                if size[i][dim_index]%factor == 0:
                    size[i][dim_index] = size[i][dim_index]//factor
                    size_code.append(i)
    return size_code

def map_decode(code,factor_list,len_list):
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

#mapper
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
    if WITH_CONSTRAINTS:
        aggregated_input.update(constraints)
    #aggregated_input.update(sparse_opt)
    job_name  = "example"
    run_timeloop_(job_name, aggregated_input, os.path.join(output_base_dir, job_name), ert_path, art_path)
    xml_path = os.path.join("/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single/outputs/example/outputs/timeloop-mapper.map+stats.xml")
   
    if os.path.exists(xml_path):
        best_performance = parse_timeloop_stats(xml_path)
        if ob == "edp":
            perfo = best_performance["energy_pJ"]*best_performance["cycles"]
        else:
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
    dict = {'cycles':'delay','energy_pJ':'energy','edp':'edp'}
    for i in range(n):
        print('============================================================================================== {} st mapper ================================================='.format(i))
        with open(mapper_path, 'r') as file:
            data = yaml.safe_load(file)
            # 修改数据
        data['mapper']['search-size'] = total_serchsize        
        data['mapper']['optimization-metric'] = dict[object]
        data['mapper']['algorithm'] = algo
        #data['mapper']['num-threads'] = (12*i//n)+1
        data['mapper']['num-threads'] = i+1
        # 将修改后的数据写回原文件
        with open(mapper_path, 'w') as file:
            yaml.dump(data, file)
        s,perfo = run_mapper_once(object)
        all_states.append(s)
        all_perfo.append(perfo)
    return all_states,all_perfo

def write_to_csv(all_states,all_perfo,csv_path,factor_list):    #这里面包含cantor和mapsize code
    valid_mapping_num = len(all_states)
    with open(csv_path, 'w', newline='') as csvfile:
    # 创建 CSV writer 对象，指定分隔符为逗号
        csvwriter = csv.writer(csvfile, delimiter=',')
    # 写入数据
        for i in range(valid_mapping_num):
            state = all_states[i]
            cantor = [cantor_encode(per) for per in state['permutations']]
            mapcode = map_encode(state['array'],factor_list)
            #print('state = ',state)
            line = [all_perfo[i]]
            line.extend(state['split'])
            line.extend(state['bypass_choice'])
            line.extend(cantor)
            line.extend(mapcode)
            #print('line = ',line)
            csvwriter.writerow(line)  # 写入表头
        
def initialize_csv():
    #print(constraints_template_path)
    dimensions,factor_list, len_list, mapping_encoding_len = correct_dimensions(problem_template_path)
    generate_constraints(dimensions,factor_list)
    all_states,all_perfo = run_mapper_manytimes(N,SEARCH_SIZE,OBJECT,ALGO)
    csv_path = os.path.join("/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single", JOB + "_mapping_found", WORK_LOAD + "_initialize.csv" )
    if not os.path.exists(csv_path):
    # 创建路径
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_to_csv(all_states,all_perfo,csv_path,factor_list)



if __name__ == "__main__":
    initialize_csv()