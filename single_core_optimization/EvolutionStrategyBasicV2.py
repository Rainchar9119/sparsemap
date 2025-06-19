
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


this_file = os.path.abspath(inspect.getfile(inspect.currentframe()))
directory = os.path.dirname(this_file)


#INITIALIZE_METHOD = "random"
#INITIALIZE_METHOD = "appoint_from_txt"
INITIALIZE_METHOD = "valid_sample"
SELECT_METHOD = "select_order"                     #按排名
#SELECT_METHOD = "select"                    #轮盘赌
OVERWRITE = True

C_SIZE = 3
M_SIZE = 96
N_SIZE = 4
P_SIZE = 56
Q_SIZE = 56
R_SIZE = 11
S_SIZE = 11
X = 8                         #PE阵列
Y = 8

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

c, c_factors = find_max_factors_in_range(C_SIZE)
m, m_factors = find_max_factors_in_range(M_SIZE)
n, n_factors = find_max_factors_in_range(N_SIZE)
p, p_factors = find_max_factors_in_range(P_SIZE)
q, q_factors = find_max_factors_in_range(Q_SIZE)
r, r_factors = find_max_factors_in_range(R_SIZE)
s, s_factors = find_max_factors_in_range(S_SIZE)


c_bound = (c_factors+1)*(c_factors+1)
m_bound = (m_factors+1)*(m_factors+1)
n_bound = (n_factors+1)*(n_factors+1)
p_bound = (p_factors+1)*(p_factors+1)
q_bound = (q_factors+1)*(q_factors+1)
r_bound = (r_factors+1)*(r_factors+1)
s_bound = (s_factors+1)*(s_factors+1)


DNA_SIZE = 77            # DNA 

'''
[0,6]          GlobleBuffer 中的 temporal 循环顺序
[7,13]         GlobleBuffer 中的  spatial 循环顺序
[14,20]          PE_Buffer  中的 temporal 循环顺序
[21,27]        C、M、N、P、Q、R、S ----------GLB T mapping
[28,34]        C、M、N、P、Q、R、S ----------GLB S mapping
[35,41]        C、M、N、P、Q、R、S ----------PEB S mapping
[42]           split 位置
[43,60]        Input 压缩格式           uop---0    b---1    cp---2    rle---3
[61,72]        Weight 压缩格式          uop---0    b---1    cp---2    rle---3
[73,74]        GlobleBuffer 中的 skip/gate
[75,76]           PE_Buffer 中的 skip/gate

'''

N_GENERATIONS = 20
POP_SIZE = 1000           # population size
N_KID = 500               # n kids per generation

def prime_factorization(n):
    factors = []
    divisor = 2

    while n > 1:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 1

    return factors

c_factor_list = prime_factorization(c)
m_factor_list = prime_factorization(m)
n_factor_list = prime_factorization(n)
p_factor_list = prime_factorization(p)
q_factor_list = prime_factorization(q)
r_factor_list = prime_factorization(r)
s_factor_list = prime_factorization(s)


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
        recover_solution.append(solution[permutation.tolist().index(i)])

    return recover_solution

def calculate_demension_factors(map):
    

    dimention_factors = [
        {
        'C': map[0],
        'M': map[1],
        'N': map[2],
        'P': map[3],
        'Q': map[4],
        'R': map[5],
        'S': map[6]
        },
        {
        'C': map[7],
        'M': map[8],
        'N': map[9],
        'P': map[10],
        'Q': map[11],
        'R': map[12],
        'S': map[13]
        },
        {
        'C': map[14],
        'M': map[15],
        'N': map[16],
        'P': map[17],
        'Q': map[18],
        'R': map[19],
        'S': map[20]
        },
    ]
    return dimention_factors


'''
[0,6]          GlobleBuffer 中的 temporal 循环顺序
[7,13]         GlobleBuffer 中的  spatial 循环顺序
[14,20]          PE_Buffer  中的 temporal 循环顺序
[21,27]        C、M、N、P、Q、R、S ----------GLB T mapping
[28,34]        C、M、N、P、Q、R、S ----------GLB S mapping
[35,41]        C、M、N、P、Q、R、S ----------PEB S mapping
[42]           split 位置
[43,60]        Input 压缩格式           uop---0    b---1    cp---2    rle---3
[61,72]        Weight 压缩格式          uop---0    b---1    cp---2    rle---3
[73,74]        GlobleBuffer 中的 skip/gate
[75,76]           PE_Buffer 中的 skip/gate

'''
def run_timeloop(job_name, input_dict, base_dir, ert_path, art_path):
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
    subprocess_cmd = ["timeloop-model", input_file_path, ert_path, art_path]
    print("\tRunning test: ", job_name)

    try:
        p = subprocess.Popen(subprocess_cmd)
        p.communicate(timeout = 5)
        # 如果子进程超时，TimeoutExpired 异常将被捕获
    except subprocess.TimeoutExpired:
        print(f"Task {job_name} timed out. Terminating the subprocess.")
        p.kill()  # 使用 kill() 终止子进程
        p.wait()  # 等待子进程完成


def population_initialize():
    population = np.random.randint(low=0, high=100, size=(POP_SIZE, DNA_SIZE), dtype=np.int)
    for i in range(POP_SIZE):
        population[i, :7] = np.random.permutation(7)
        population[i, 7:14] = np.random.permutation(7)
        population[i, 14:21] = np.random.permutation(7)


    ####################待修改
    population[:, 21] = np.random.randint(low=0, high=c_bound, size=POP_SIZE)
    population[:, 22] = np.random.randint(low=0, high=m_bound, size=POP_SIZE)
    population[:, 23] = np.random.randint(low=0, high=n_bound, size=POP_SIZE)
    population[:, 24] = np.random.randint(low=0, high=p_bound, size=POP_SIZE)
    population[:, 25] = np.random.randint(low=0, high=q_bound, size=POP_SIZE)
    population[:, 26] = np.random.randint(low=0, high=r_bound, size=POP_SIZE)
    population[:, 27] = np.random.randint(low=0, high=s_bound, size=POP_SIZE)
    
    for i in range(POP_SIZE):
        population[i, 42] = np.random.randint(low=0, high=8)            #split位置
    ######################



        Input_UOP_free = np.random.randint(low=43, high=61)                    #input压缩格式从哪个开始不是UOP
        Weight_UOP_free = np.random.randint(low=61, high=73)                   #weight压缩格式从哪个开始不是UOP
        
        population[i,43:61] = 0
        for j in range(Input_UOP_free,61):
            population[i,j] =  np.random.randint(low=1, high=4)  

        population[i,61:73] = 0
        for j in range(Weight_UOP_free,73):
            population[i,j] =  np.random.randint(low=1, high=4)



        population[i,73] = np.random.randint(low=0, high=8)                   #GLB  I/W  S/G
        population[i,74] = np.random.randint(low=0, high=8)                   #GLB   O   S/G
        population[i,75] = np.random.randint(low=0, high=8)                   #PEB  I/W  S/G
        population[i,76] = np.random.randint(low=0, high=8)                   #PEB   O   S/G

    return population

def sample_initialize():
    population = np.random.randint(low=0, high=100, size=(POP_SIZE, DNA_SIZE), dtype=np.int)
    for i in range(POP_SIZE):
        population[i, :7] = np.random.permutation(7)
        population[i, 7:14] = np.random.permutation(7)
        population[i, 14:21] = np.random.permutation(7)


    
        population[i, 42] = np.random.randint(low=0, high=8)            #split位置
    
        dimensions = [c,m,n,p,q,r,s]
        SP = population[i,42]
        permutation = population[i,7:14]
        recover_solution = generate_solutions(dimensions,X,Y,SP,permutation)
        for k in range(3):
            for l in range(7):
                population[i,21+7*k+l] = recover_solution[l][k]

        Input_UOP_free = np.random.randint(low=43, high=61)                    #input压缩格式从哪个开始不是UOP
        Weight_UOP_free = np.random.randint(low=61, high=73)                   #weight压缩格式从哪个开始不是UOP
        
        population[i,43:61] = 0
        for j in range(Input_UOP_free,61):
            population[i,j] =  np.random.randint(low=1, high=4)  

        population[i,61:73] = 0
        for j in range(Weight_UOP_free,73):
            population[i,j] =  np.random.randint(low=1, high=4)



        population[i,73] = np.random.randint(low=0, high=8)                   #GLB  I/W  S/G
        population[i,74] = np.random.randint(low=0, high=8)                   #GLB   O   S/G
        population[i,75] = np.random.randint(low=0, high=8)                   #PEB  I/W  S/G
        population[i,76] = np.random.randint(low=0, high=8)                   #PEB   O   S/G

    return population


def evaluate_individual(indi):                                    #indi是长度63的np array

    dimension = ['C','M','N','P','Q','R','S']
    
    #----------------------------------------------------  解码mapping  ------------------------------------------------------

    mapping = indi[21:42]
    dimention_factors = calculate_demension_factors(mapping)

    #print(dimention_factors)
    
    split_GB_to_PE = indi[42]



    permutation_GB_T_order = (indi[0:7]).tolist()
    permutation_GB_S_order = (indi[7:14]).tolist()
    permutation_PEB_T_order = (indi[14:21]).tolist()

    permutation_GB_T_str = (dimension[permutation_GB_T_order[0]] + dimension[permutation_GB_T_order[1]] + dimension[permutation_GB_T_order[2]]
                  + dimension[permutation_GB_T_order[3]] + dimension[permutation_GB_T_order[4]] + dimension[permutation_GB_T_order[5]]
                  + dimension[permutation_GB_T_order[6]])

    permutation_GB_S_str = (dimension[permutation_GB_S_order[0]] + dimension[permutation_GB_S_order[1]] + dimension[permutation_GB_S_order[2]]
                  + dimension[permutation_GB_S_order[3]] + dimension[permutation_GB_S_order[4]] + dimension[permutation_GB_S_order[5]]
                  + dimension[permutation_GB_S_order[6]])

    permutation_PEB_T_str = (dimension[permutation_PEB_T_order[0]] + dimension[permutation_PEB_T_order[1]] + dimension[permutation_PEB_T_order[2]]
                  + dimension[permutation_PEB_T_order[3]] + dimension[permutation_PEB_T_order[4]] + dimension[permutation_PEB_T_order[5]]
                  + dimension[permutation_PEB_T_order[6]])

    factors_GB_T_str = ('C='+str(dimention_factors[0]['C'])+' '+
                        'M='+str(dimention_factors[0]['M'])+' '+
                        'N='+str(dimention_factors[0]['N'])+' '+ 
                        'P='+str(dimention_factors[0]['P'])+' '+
                        'Q='+str(dimention_factors[0]['Q'])+' '+
                        'R='+str(dimention_factors[0]['R'])+' '+
                        'S='+str(dimention_factors[0]['S']))

    factors_GB_S_str = ('C='+str(dimention_factors[1]['C'])+' '+
                        'M='+str(dimention_factors[1]['M'])+' '+
                        'N='+str(dimention_factors[1]['N'])+' '+ 
                        'P='+str(dimention_factors[1]['P'])+' '+
                        'Q='+str(dimention_factors[1]['Q'])+' '+
                        'R='+str(dimention_factors[1]['R'])+' '+
                        'S='+str(dimention_factors[1]['S']))

    factors_PEB_T_str = ('C='+str(dimention_factors[2]['C'])+' '+
                        'M='+str(dimention_factors[2]['M'])+' '+
                        'N='+str(dimention_factors[2]['N'])+' '+ 
                        'P='+str(dimention_factors[2]['P'])+' '+
                        'Q='+str(dimention_factors[2]['Q'])+' '+
                        'R='+str(dimention_factors[2]['R'])+' '+
                        'S='+str(dimention_factors[2]['S']))

    
    mapping_variables = {
    'permutation_GB_T': permutation_GB_T_str,
    'factors_GB_T': factors_GB_T_str,
    'permutation_GB_S': permutation_GB_S_str,
    'split_GB_to_PE': split_GB_to_PE,
    'factors_GB_S': factors_GB_S_str,
    'permutation_PEB_T': permutation_PEB_T_str,
    'factors_PEB_T': factors_PEB_T_str,
    }

    # ------------------------------------  mapping  输入文件生成  -------------------------------------------
    # 读取模板文件
    #print(os.getcwd())
    os.chdir(directory)
    print(os.getcwd())
    with open('evaluate_templates/mapping_template.j2', 'r') as template_file_mapping:
        template_content_mapping = template_file_mapping.read()

    # 创建 Jinja2 模板对象
    template_mapping = Template(template_content_mapping)

    # 渲染模板并生成 YAML 内容
    rendered_yaml_mapping = template_mapping.render(mapping_variables)

    # 将生成的 YAML 内容写入文件
    with open('mapping_output.yaml', 'w') as output_file_mapping:
        output_file_mapping.write(rendered_yaml_mapping)


    # ------------------------------------  解码 sparse_opt -------------------------------------------------

    '''
    [0,6]          GlobleBuffer 中的 temporal 循环顺序
    [7,13]         GlobleBuffer 中的  spatial 循环顺序
    [14,20]          PE_Buffer  中的 temporal 循环顺序
    [21,27]        C、M、N、P、Q、R、S ----------GLB T mapping
    [28,34]        C、M、N、P、Q、R、S ----------GLB S mapping
    [35,41]        C、M、N、P、Q、R、S ----------PEB S mapping
    [42]           split 位置
    [43,60]        Input 压缩格式           uop---0    b---1    cp---2    rle---3
    [61,72]        Weight 压缩格式          uop---0    b---1    cp---2    rle---3
    [73,74]        GlobleBuffer 中的 skip/gate
    [75,76]           PE_Buffer 中的 skip/gate

    '''


    compress_format_dict = ['UOP','B','CP','RLE']

    input_compress = indi[43:61]
    for i in range(0,18):                                                  #模4得到0-3的值
        input_compress[i] = input_compress[i]%4
    weight_compress = indi[61:73]
    for i in range(0,12):
        weight_compress[i] = weight_compress[i]%4
    
    #计算rank层数
    PE_input_rank_num = 0
    PE_weight_rank_num = 0

    for dimension in ['N','C','P','Q','R','S']:
        if dimention_factors[2][dimension] > 1:
            PE_input_rank_num += 1
    for dimension in ['C','R','S','M']:
        if dimention_factors[2][dimension] > 1:
            PE_weight_rank_num += 1

    GB_input_rank_num = 0
    GB_weight_rank_num = 0

    for dimension in ['N','C','P','Q','R','S']:
        for dimention_factor in dimention_factors:
            if dimention_factor[dimension] > 1:
                GB_input_rank_num += 1
    for dimension in ['C','R','S','M']:
        for dimention_factor in dimention_factors:
            if dimention_factor[dimension] > 1:
                GB_weight_rank_num += 1

    #print("PE_input_rank_num =")
    #print(PE_input_rank_num)
    #print("GB_input_rank_num =")
    #print(GB_input_rank_num)

    data = [
        [
            {
                'name': 'GlobelBuffer',
                'representation-format': 
                {
                    'data-spaces': 
                    [
                        {
                            'name': 'Inputs',
                            'ranks': 
                            [
                                #{'format': 'B'},

                            ],
                        },
                        {
                            'name': 'Weights',
                            'ranks': 
                            [
                                #{'format': 'UOP'},
                                
                            ],
                        },
                    ],
                },
            },
            {
                'name': 'PE_buffer',
                'representation-format': {
                    'data-spaces': [
                        {
                            'name': 'Inputs',
                            'ranks': [
                                #{'format': 'CP'},
                                
                            ],
                        },
                        {
                            'name': 'Weights',
                            'ranks': [
                                #{'format': 'UOP'},
                                
                            ],
                        },
                    ],
                },
            },
        ],

    ]

    for i in range(GB_input_rank_num):
        data[0][0]['representation-format']['data-spaces'][0]['ranks'].append({'format': compress_format_dict[input_compress[18-GB_input_rank_num+i]]})
    for i in range(GB_weight_rank_num):
        data[0][0]['representation-format']['data-spaces'][1]['ranks'].append({'format': compress_format_dict[weight_compress[12-GB_weight_rank_num+i]]})
    for i in range(PE_input_rank_num):
        data[0][1]['representation-format']['data-spaces'][0]['ranks'].append({'format': compress_format_dict[input_compress[18-PE_input_rank_num+i]]})
    for i in range(PE_weight_rank_num):
        data[0][1]['representation-format']['data-spaces'][1]['ranks'].append({'format': compress_format_dict[weight_compress[12-PE_weight_rank_num+i]]})

    
    '''
        population[i,73] = np.random.randint(low=0, high=8)                   #GLB  I/W  S/G
        population[i,74] = np.random.randint(low=0, high=8)                   #GLB   O   S/G
        population[i,75] = np.random.randint(low=0, high=8)                   #PEB  I/W  S/G
        population[i,76] = np.random.randint(low=0, high=8)                   #PEB   O   S/G
    '''


#--------------------------------GLB----------I/W----------S/G---------------------------------------
    if indi[73] == 0:
        data.append([])

    elif indi[73] == 1:
        data.append([])

    elif indi[73] == 2:
        SG_data = [
            {
                'name': 'GlobelBuffer',
                'type': 'skipping',
                'target': 'Inputs',
                'condition_on': '[ Weights ]'
            }
        ]
        data.append(SG_data)

    elif indi[73] == 3:
        SG_data = [
            {
                'name': 'GlobelBuffer',
                'type': 'gating',
                'target': 'Inputs',
                'condition_on': '[ Weights ]'
            }
        ]
        data.append(SG_data)

    elif indi[73] == 4:
        SG_data = [
            {
                'name': 'GlobelBuffer',
                'type': 'skipping',
                'target': 'Weights',
                'condition_on': '[ Inputs ]'
            }
        ]
        data.append(SG_data)
    
    elif indi[73] == 5:
        SG_data = [
            {
                'name': 'GlobelBuffer',
                'type': 'gating',
                'target': 'Weights',
                'condition_on': '[ Inputs ]'
            }
        ]
        data.append(SG_data)

    elif indi[73] == 6:
        SG_data = [
            {
                'name': 'GlobelBuffer',
                'type': 'skipping',
                'target': 'Inputs',
                'condition_on': '[ Weights ]'
            },
            {
                'name': 'GlobelBuffer',
                'type': 'skipping',
                'target': 'Weights',
                'condition_on': '[ Inputs ]'
            }
        ]
        data.append(SG_data)

    elif indi[73] == 7:
        SG_data = [
            {
                'name': 'GlobelBuffer',
                'type': 'gating',
                'target': 'Inputs',
                'condition_on': '[ Weights ]'
            },
            {
                'name': 'GlobelBuffer',
                'type': 'gating',
                'target': 'Weights',
                'condition_on': '[ Inputs ]'
            }
        ]
        data.append(SG_data)

#--------------------------------GLB----------O----------S/G---------------------------------------

    if indi[74] == 0:
        data.append([])

    elif indi[74] == 1:
        data.append([])

    elif indi[74] == 2:
        SG_data = [
            {
                'name': 'GlobelBuffer',
                'type': 'skipping',
                'target': 'Outputs',
                'condition_on': '[ Inputs ]'
            }
        ]
        data.append(SG_data)

    elif indi[74] == 3:
        SG_data = [
            {
                'name': 'GlobelBuffer',
                'type': 'gating',
                'target': 'Outputs',
                'condition_on': '[ Inputs ]'
            }
        ]
        data.append(SG_data)

    elif indi[74] == 4:
        SG_data = [
            {
                'name': 'GlobelBuffer',
                'type': 'skipping',
                'target': 'Outputs',
                'condition_on': '[ Weights ]'
            }
        ]
        data.append(SG_data)
    
    elif indi[74] == 5:
        SG_data = [
            {
                'name': 'GlobelBuffer',
                'type': 'gating',
                'target': 'Outputs',
                'condition_on': '[ Weights ]'
            }
        ]
        data.append(SG_data)

    elif indi[74] == 6:
        SG_data = [
            {
                'name': 'GlobelBuffer',
                'type': 'skipping',
                'target': 'Outputs',
                'condition_on': '[ Weights, Inputs ]'
            }
        ]
        data.append(SG_data)

    elif indi[74] == 7:
        SG_data = [
            {
                'name': 'GlobelBuffer',
                'type': 'gating',
                'target': 'Outputs',
                'condition_on': '[ Weights, Inputs ]'
            }
        ]
        data.append(SG_data)


#--------------------------------PEB----------I/W----------S/G---------------------------------------

    if indi[75] == 0:
        data.append([])

    elif indi[75] == 1:
        data.append([])

    elif indi[75] == 2:
        SG_data = [
            {
                'name': 'PE_buffer',
                'type': 'skipping',
                'target': 'Inputs',
                'condition_on': '[ Weights ]'
            }
        ]
        data.append(SG_data)

    elif indi[75] == 3:
        SG_data = [
            {
                'name': 'PE_buffer',
                'type': 'gating',
                'target': 'Inputs',
                'condition_on': '[ Weights ]'
            }
        ]
        data.append(SG_data)

    elif indi[75] == 4:
        SG_data = [
            {
                'name': 'PE_buffer',
                'type': 'skipping',
                'target': 'Weights',
                'condition_on': '[ Inputs ]'
            }
        ]
        data.append(SG_data)
    
    elif indi[75] == 5:
        SG_data = [
            {
                'name': 'PE_buffer',
                'type': 'gating',
                'target': 'Weights',
                'condition_on': '[ Inputs ]'
            }
        ]
        data.append(SG_data)

    elif indi[75] == 6:
        SG_data = [
            {
                'name': 'PE_buffer',
                'type': 'skipping',
                'target': 'Inputs',
                'condition_on': '[ Weights ]'
            },
            {
                'name': 'PE_buffer',
                'type': 'skipping',
                'target': 'Weights',
                'condition_on': '[ Inputs ]'
            }
        ]
        data.append(SG_data)

    elif indi[75] == 7:
        SG_data = [
            {
                'name': 'PE_buffer',
                'type': 'gating',
                'target': 'Inputs',
                'condition_on': '[ Weights ]'
            },
            {
                'name': 'PE_buffer',
                'type': 'gating',
                'target': 'Weights',
                'condition_on': '[ Inputs ]'
            }
        ]
        data.append(SG_data)


#--------------------------------PEB----------O----------S/G---------------------------------------

    if indi[76] == 0:
        data.append([])

    elif indi[76] == 1:
        data.append([])

    elif indi[76] == 2:
        SG_data = [
            {
                'name': 'PE_buffer',
                'type': 'skipping',
                'target': 'Outputs',
                'condition_on': '[ Inputs ]'
            }
        ]
        data.append(SG_data)

    elif indi[76] == 3:
        SG_data = [
            {
                'name': 'PE_buffer',
                'type': 'gating',
                'target': 'Outputs',
                'condition_on': '[ Inputs ]'
            }
        ]
        data.append(SG_data)

    elif indi[76] == 4:
        SG_data = [
            {
                'name': 'PE_buffer',
                'type': 'skipping',
                'target': 'Outputs',
                'condition_on': '[ Weights ]'
            }
        ]
        data.append(SG_data)
    
    elif indi[76] == 5:
        SG_data = [
            {
                'name': 'PE_buffer',
                'type': 'gating',
                'target': 'Outputs',
                'condition_on': '[ Weights ]'
            }
        ]
        data.append(SG_data)

    elif indi[76] == 6:
        SG_data = [
            {
                'name': 'PE_buffer',
                'type': 'skipping',
                'target': 'Outputs',
                'condition_on': '[ Weights, Inputs ]'
            }
        ]
        data.append(SG_data)

    elif indi[76] == 7:
        SG_data = [
            {
                'name': 'PE_buffer',
                'type': 'gating',
                'target': 'Outputs',
                'condition_on': '[ Weights, Inputs ]'
            }
        ]
        data.append(SG_data)


#------------------------------------------生成sparse-opt的yaml文件-------------------------------------------------

    env = Environment(loader=FileSystemLoader('.'))  # 使用当前目录作为模板目录

    # 加载模板
    template = env.get_template('evaluate_templates/sparse_opt_template.j2')

    #print('================sparse_opt==================\n')
    #print(data)
    #print('================sparse_opt==================\n')
    # 渲染模板
    output_yaml = template.render(data=data)

    # 将结果写入文件
    with open('sparse_opt_output.yaml', 'w') as file:
        file.write(output_yaml)


#----------------------------------------------调用sparse_loop-------------------------------------------------------


    this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
    this_directory = os.path.dirname(this_file_path)
    problem_template_path = os.path.join(this_directory, "workload_example.yaml")
    arch_path = os.path.join(this_directory, "arch_example.yaml")
    #component_path = os.path.join(this_directory, "..", "multiSCNN","fig13_dstc_setup","input_specs",  "compound_components.yaml")
    mapping_path = os.path.join(this_directory, "mapping_output.yaml")
    mapper_path = os.path.join(this_directory, "mapper.yaml")
    sparse_opt_path = os.path.join(this_directory, "sparse_opt_output.yaml")
    #sparse_opt_path = os.path.join(this_directory, "..", "multiSCNN","single_core_optimization",  "yaml_gen", "sparse_opt_output.yaml")



    ert_path = os.path.join(this_directory, "..","..", "evaluation_setups","fig13_dstc_setup","input_specs",  "ERT.yaml")
    art_path = os.path.join(this_directory, "..","..", "evaluation_setups","fig13_dstc_setup","input_specs",  "ART.yaml")
    #print(ert_path)
    
    # load all yaml input specs
    problem_template = yaml.load(open(problem_template_path), Loader = yaml.SafeLoader)
    arch = yaml.load(open(arch_path), Loader = yaml.SafeLoader)
    #components = yaml.load(open(component_path), Loader = yaml.SafeLoader)
    mapping = yaml.load(open(mapping_path), Loader = yaml.SafeLoader)
    #mapper = yaml.load(open(mapper_path), Loader = yaml.SafeLoader)
    sparse_opt = yaml.load(open(sparse_opt_path), Loader = yaml.SafeLoader)
    
    output_base_dir = os.path.join(this_directory, "search_outputs")

    new_problem = deepcopy(problem_template)

    #new_problem["problem"]["instance"]["densities"]["A"]["density"] = 0.55
    #new_problem["problem"]["instance"]["densities"]["B"]["density"] = 0.35

    aggregated_input = {}
    aggregated_input.update(arch)
    aggregated_input.update(new_problem)
    #aggregated_input.update(components)
    aggregated_input.update(mapping)
    #aggregated_input.update(mapper)
    aggregated_input.update(sparse_opt)
    
    
    job_name  = "searching"
    

#---------------------------------------------------调用,生成output---------------------------------------------------------
    run_timeloop(job_name, aggregated_input, os.path.join(output_base_dir, job_name), ert_path, art_path)


#------------------------------------------------------读取数据--------------------------------------------------------------
#++++++++++++++++++++++++++++++++++++++++++++++++本实验优化目标+++++++++++++++++++++++++++++++++++++++++
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
    
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    base_output_dir = os.path.join(this_directory, "search_outputs")
    output_file_path = os.path.join(base_output_dir, job_name, 'outputs', "timeloop-model.map+stats.xml")
    if os.path.exists(output_file_path):
        job_output_stats = parse_timeloop_stats(output_file_path)
        os.remove(output_file_path)
        fitness = 1/job_output_stats[stat_type]
    else:
        fitness = 0.000000000000000001
    return fitness



def select(population, fitness):
    # 根据适应度值选择个体
    selected_indices = np.random.choice(len(population), size=POP_SIZE - N_KID, p=fitness/fitness.sum())         #轮盘赌选择
    selected_population = population[selected_indices]
    selected_fitness = fitness[selected_indices]
    return selected_population, selected_fitness 

def select_order(population, fitness):
    # 根据适应度对个体进行排序的索引
    sorted_indices = np.argsort(fitness)[::-1]

    # 选择排名前n的个体及其适应度
    selected_population = population[sorted_indices[:POP_SIZE - N_KID]]
    selected_fitness = fitness[sorted_indices[:POP_SIZE - N_KID]]

    return selected_population, selected_fitness

def crossover(parents, n_kid):
    kids = []
    for i in range(n_kid):
        selected_indices = np.random.choice(parents.shape[0], size=2, replace=False)
        parent1 = parents[selected_indices[0]]
        parent2 = parents[selected_indices[1]]
        crossover_point = np.random.randint(0, 6)                               #决定染色体交叉点
        if crossover_point == 0:
            kid = np.concatenate([parent1[:7], parent2[7:]])
        elif  crossover_point == 1:
            kid = np.concatenate([parent1[:14], parent2[14:]])
        elif  crossover_point == 2:
            kid = np.concatenate([parent1[:21], parent2[21:]])
        elif  crossover_point == 3:
            kid = np.concatenate([parent1[:43], parent2[43:]])
        elif  crossover_point == 4:
            kid = np.concatenate([parent1[:61], parent2[61:]])
        else:
            kid = np.concatenate([parent1[:73], parent2[73:]])

        kids.append(kid)
    return np.array(kids)

def mutate(population, mutation_rate=0.5):

    mutated_population = population.copy()

    for indi in mutated_population:
        if np.random.rand() < mutation_rate:
            mutation_choice = np.random.randint(0,9)                                                #决定在哪一段上进行变异
            if mutation_choice == 0:
                random_numbers = np.random.choice(np.arange(0, 6), size=2, replace=False)           #GB_T交换对
                temp = indi[random_numbers[0]]
                indi[random_numbers[0]] = indi[random_numbers[1]]
                indi[random_numbers[1]] = temp                                                      #完成交换
            
            elif mutation_choice == 1:
                random_numbers = np.random.choice(np.arange(7, 14), size=2, replace=False)           #GB_S交换对
                temp = indi[random_numbers[0]]
                indi[random_numbers[0]] = indi[random_numbers[1]]
                indi[random_numbers[1]] = temp                                                      #完成交换
            
            elif mutation_choice == 2:
                random_numbers = np.random.choice(np.arange(14, 21), size=2, replace=False)           #PEB交换对
                temp = indi[random_numbers[0]]
                indi[random_numbers[0]] = indi[random_numbers[1]]
                indi[random_numbers[1]] = temp                                                      #完成交换

            elif mutation_choice == 3:                                                             #mapping中选一个level.改变其中一个维度
                level = np.random.randint(1, 4)
                dimension = np.random.randint(0, 7)
                if level == 1:
                    dimension_factor_list = prime_factorization(indi[21+dimension])
                    if dimension_factor_list:
                        random_factor = random.choice(dimension_factor_list)
                        up_or_down = np.random.randint(0, 2)
                        if up_or_down == 0:       #向level 2
                            indi[21+dimension] //= random_factor
                            indi[28+dimension] *= random_factor
                        elif up_or_down == 1:       #向level 3
                            indi[21+dimension] //= random_factor
                            indi[35+dimension] *= random_factor

                elif level == 2:
                    dimension_factor_list = prime_factorization(indi[28+dimension])
                    if dimension_factor_list:
                        random_factor = random.choice(dimension_factor_list)
                        up_or_down = np.random.randint(0, 2)
                        if up_or_down == 0:       #向level 1
                            indi[28+dimension] //= random_factor
                            indi[21+dimension] *= random_factor
                        elif up_or_down == 1:       #向level 3
                            indi[28+dimension] //= random_factor
                            indi[35+dimension] *= random_factor

                elif level == 3:
                    dimension_factor_list = prime_factorization(indi[35+dimension])
                    if dimension_factor_list:
                        random_factor = random.choice(dimension_factor_list)
                        up_or_down = np.random.randint(0, 2)
                        if up_or_down == 0:       #向level 1
                            indi[35+dimension] //= random_factor
                            indi[21+dimension] *= random_factor
                        elif up_or_down == 1:       #向level 2
                            indi[35+dimension] //= random_factor
                            indi[28+dimension] *= random_factor

            elif mutation_choice == 4:                                                             #split 向旁边移动一位    split是1到6
                if indi[42] == 1:
                    indi[42] = 2
                elif indi[42] == 2:
                    indi[42] = 3
                elif indi[42] == 3:
                    indi[42] = 4
                elif indi[42] == 4:
                    indi[42] = 3
                elif indi[42] == 5:
                    indi[42] = 4
                elif indi[42] == 6:
                    indi[42] = 5

            elif mutation_choice == 5:                                                             #Input压缩格式低四位 选一个+1
                random_number = np.random.randint(43, 47)
                indi[random_number] = indi[random_number] + 1
                if indi[random_number]%4 == 0:                                            #如果选中UOP，把后面所有UOP给变了；如果变成UOP，把前面全变成UOP
                    for i in range(29,random_number):
                        indi[i] = 0
                elif indi[random_number]%4 == 1:
                    for i in range(random_number,47):
                        if indi[i] == 0:
                            indi[i] = np.random.randint(1, 4)

            elif mutation_choice == 6:                                                             #Weight压缩格式低三位 选一个+1
                random_number = np.random.randint(56, 59)
                indi[random_number] = indi[random_number] + 1
                if indi[random_number]%4 == 0:                                            #如果选中UOP，把后面所有UOP给变了；如果变成UOP，把前面全变成UOP
                    for i in range(47,random_number):
                        indi[i] = 0
                elif indi[random_number]%4 == 1:
                    for i in range(random_number,59):
                        if indi[i] == 0:
                            indi[i] = np.random.randint(1, 4)

            elif mutation_choice == 7:                                                             #GLB_SG选一位随机
                random_number = np.random.randint(59, 61)
                indi[random_number] = np.random.randint(low=0, high=8)

            else:                                                             #GLB_SG选一位随机
                random_number = np.random.randint(61, 63)
                indi[random_number] = np.random.randint(low=0, high=8)

    return mutated_population




def envolve(population,fitness):
    if SELECT_METHOD == "select" :
        selected_population, parents_fitness = select_order(population, fitness)                                       #从上一代中挑出50个
    elif SELECT_METHOD == "select_order":
         selected_population, parents_fitness = select_order(population, fitness)  
    kids = crossover(selected_population, N_KID)
    kids = mutate(kids)                 
    kids_fitness = np.array([evaluate_individual(indi) for indi in kids])                                                    #生成子代       
    next_generation_candidates = np.concatenate([selected_population, kids])  
    fitness = np.append(parents_fitness,kids_fitness)
    return next_generation_candidates, fitness


def show_result(fitness):
    
    print(fitness)

def fill_population_with_best_individuals(population, file_path):
    # 加载保存的最优个体数据，指定dtype为int
    best_individuals = np.loadtxt(file_path, dtype=float)

    # 将加载的值四舍五入为整数
    best_individuals = np.round(best_individuals).astype(int)

    # 确保最优个体数据的维度和种群的最后一维一致
    if best_individuals.shape[1] == population.shape[1]:
        # 将最优个体填充到种群的后100行
        population[-100:] = best_individuals[-100:]
    else:
        print("维度不匹配，无法填充。")




plt.rcParams['font.family'] = 'Arial Unicode MS'

if INITIALIZE_METHOD == "radom":
    population = population_initialize()
elif INITIALIZE_METHOD == "appoint_from_txt":
    population = np.random.randint(low=0, high=100, size=(100, 63), dtype=np.int)
    file_path = os.path.join(directory,"search_outputs","searching","outputs", "generation_best_individual.txt")
    # 调用函数填充 population
    fill_population_with_best_individuals(population, file_path)
elif INITIALIZE_METHOD == "valid_sample":
    population = sample_initialize()

fitness = np.array([evaluate_individual(indi) for indi in population])
generation_best_individual = []
generation_best_performance = []

for i in range(N_GENERATIONS):
    population,fitness = envolve(population,fitness)
    best_indi_index = np.argmax(fitness)
    generation_best_individual.append(population[best_indi_index,:].tolist())
    generation_best_performance.append(fitness[best_indi_index])
    print("=====================================================================\n")
    print("GENERATION = ",i)
    print("Best performance is",fitness[best_indi_index])
    print("=====================================================================\n")
now = datetime.now()
formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
np.savetxt(formatted_now +'_best_individual.txt', generation_best_individual)
np.savetxt(formatted_now +'_best_performance.txt', generation_best_performance)
y_values = generation_best_performance
x_values = list(range(1, len(generation_best_performance) + 1))
plt.title('best_performance_of_generation')
plt.xlabel('generation')
plt.ylabel('performance')
plt.savefig('generation_performance_plot.png')
plt.show()
#show_result(fitness)


