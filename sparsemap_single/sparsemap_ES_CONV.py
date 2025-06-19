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
import csv
import pandas as pd
import nevergrad as ng

#五层存储的ES  此版本暂时只考虑mapping

OVERWRITE = 1

dimensions = {'C': 128, 
              'M': 256, 
              'N': 1, 
              'P': 54, 
              'Q': 54, 
              'R': 3, 
              'S': 3}
JOB = "vgg_conv"              #高层次
WORK_LOAD = "workload_vgg_layer5"        #低层次
PLATFORM = "edge"
N_GENERATIONS = 5
POP_SIZE = 100           # population size
N_KID = 50               # n kids per generation

OBJECT = 'cycles'       #优化目标
#OBJECT = 'edp'
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

this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
this_directory = os.path.dirname(this_file_path)

problem_template_path = os.path.join(this_directory, "yamls", JOB , WORK_LOAD + ".yaml")
sparse_opt_temp_path = os.path.join(this_directory, "yamls",  "sparse_opt_template.j2")
sparse_opt_output_path = os.path.join(this_directory, "yamls",  "sparse_opt_output.yaml")

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

##注：MM和CONV的这个函数不一样
def map_decode(code):
    #print("decoding mapping -----------",code)
    #print("code_len = ", len(code))
    #print("mapping_encoding_len = ", mapping_encoding_len)
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

#run_timeloop负责调用sparseloopp
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
    #ert_file_path = os.path.join(base_dir, "ERT.yaml")
    #art_file_path = os.path.join(base_dir, "ART.yaml")
    #logfile_path = os.path.join(output_dir, "timeloop.log")
    
    yaml.dump(input_dict, open(input_file_path, "w"), default_flow_style=False)
    os.chdir(output_dir)
    subprocess_cmd = ["timeloop-model", input_file_path, ert_path, art_path]
    #subprocess_cmd = ["timeloop-mapper", input_file_path, ert_path, art_path]
    print("\tRunning test: ", job_name)

    p = subprocess.Popen(subprocess_cmd)
    try:
        p.communicate(timeout=0.3) 
    except subprocess.TimeoutExpired:
        p.terminate()
        #this_file_path = '/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single/MCTS_initialize.py'
        #this_directory = os.path.dirname(this_file_path)
        #base_output_dir = os.path.join(this_directory, "MCTS_search_outputs")
        #output_file_path = os.path.join(base_output_dir, job_name, 'outputs', "timeloop-model.map+stats.xml")
        path = '/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single/ES_search_outputs/outputs/timeloop-model.map+stats.xml'
        if os.path.exists(path):
            os.remove(path)
        return 


def generate_sparse_opt(inputCF,weightCF,SG_option,mapcode):
    
    compress_format_dict = ['UOP','B','CP','RLE']

    #对mapping解码
    size = map_decode(mapcode)
    #计算需要的ranks数量
    PE_input_rank_num = 0
    PE_weight_rank_num = 0

    for level in range(3,5):
        for dim in [0,2,3,4,5,6]:
            if size[level][dim]>1:
                PE_input_rank_num +=1

    for level in range(3,5):
        for dim in [0,1,5,6]:
            if size[level][dim]>1:
                PE_weight_rank_num +=1
    
    GB_input_rank_num = 0
    GB_weight_rank_num = 0

    for level in range(1,5):
        for dim in [0,2,3,4,5,6]:
            if size[level][dim]>1:
                GB_input_rank_num +=1

    for level in range(3,5):
        for dim in [0,1,5,6]:
            if size[level][dim]>1:
                GB_weight_rank_num +=1

    DRAM_input_rank_num = 0
    DRAM_weight_rank_num = 0

    for level in range(0,5):
        for dim in [0,2,3,4,5,6]:
            if size[level][dim]>1:
                DRAM_input_rank_num +=1

    for level in range(0,5):
        for dim in [0,1,5,6]:
            if size[level][dim]>1:
                DRAM_weight_rank_num +=1



    PE_inputCF = ['UOP' for i in range(PE_input_rank_num)]
    if PE_input_rank_num >= 5:
        PE_inputCF[-5:] = [compress_format_dict[i] for i in inputCF]
    else:
        P = [compress_format_dict[i] for i in inputCF]
        PE_inputCF = P[-PE_input_rank_num:]



    PE_weightCF = ['UOP' for i in range(PE_weight_rank_num)]
    if PE_weight_rank_num >= 5:
        PE_weightCF[-5:] = [compress_format_dict[i] for i in weightCF]
    else:
        P = [compress_format_dict[i] for i in weightCF]
        PE_weightCF = P[-PE_weight_rank_num:]



    GB_inputCF = ['UOP' for i in range(GB_input_rank_num)]
    if GB_input_rank_num >= 5:
        GB_inputCF[-5:] = [compress_format_dict[i] for i in inputCF]
    else:
        P = [compress_format_dict[i] for i in inputCF]
        GB_inputCF = P[-GB_input_rank_num:]



    GB_weightCF = ['UOP' for i in range(GB_weight_rank_num)]
    if GB_weight_rank_num >= 5:
        GB_weightCF[-5:] = [compress_format_dict[i] for i in weightCF]
    else:
        P = [compress_format_dict[i] for i in weightCF]
        GB_weightCF = P[-GB_weight_rank_num:]


    DRAM_inputCF = ['UOP' for i in range(DRAM_input_rank_num)]
    if DRAM_input_rank_num >= 5:
        DRAM_inputCF[-5:] = [compress_format_dict[i] for i in inputCF]
    else:
        P = [compress_format_dict[i] for i in inputCF]
        DRAM_inputCF = P[-DRAM_input_rank_num:]



    DRAM_weightCF = ['UOP' for i in range(DRAM_weight_rank_num)]
    if DRAM_weight_rank_num >= 5:
        DRAM_weightCF[-5:] = [compress_format_dict[i] for i in weightCF]
    else:
        P = [compress_format_dict[i] for i in weightCF]
        DRAM_weightCF = P[-DRAM_weight_rank_num:]


    data = [
        [   
            {
                'name': 'DRAM',
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
    
    for i in range(DRAM_input_rank_num):
        data[0][0]['representation-format']['data-spaces'][0]['ranks'].append({'format': DRAM_inputCF[i]})
    for i in range(DRAM_weight_rank_num):
        data[0][0]['representation-format']['data-spaces'][1]['ranks'].append({'format': DRAM_weightCF[i]})
    for i in range(GB_input_rank_num):
        data[0][1]['representation-format']['data-spaces'][0]['ranks'].append({'format': GB_inputCF[i]})
    for i in range(GB_weight_rank_num):
        data[0][1]['representation-format']['data-spaces'][1]['ranks'].append({'format': GB_weightCF[i]})
    for i in range(PE_input_rank_num):
        data[0][2]['representation-format']['data-spaces'][0]['ranks'].append({'format': PE_inputCF[i]})
    for i in range(PE_weight_rank_num):
        data[0][2]['representation-format']['data-spaces'][1]['ranks'].append({'format': PE_weightCF[i]})


    if SG_option[0] == 0:
        data.append([])

    elif SG_option[0] == 1:
        data.append([])

    elif SG_option[0] == 2:
        SG_data = [
            {
                'name': 'GlobelBuffer',
                'type': 'skipping',
                'target': 'Inputs',
                'condition_on': '[ Weights ]'
            }
        ]
        data.append(SG_data)

    elif SG_option[0] == 3:
        SG_data = [
            {
                'name': 'GlobelBuffer',
                'type': 'gating',
                'target': 'Inputs',
                'condition_on': '[ Weights ]'
            }
        ]
        data.append(SG_data)

    elif SG_option[0] == 4:
        SG_data = [
            {
                'name': 'GlobelBuffer',
                'type': 'skipping',
                'target': 'Weights',
                'condition_on': '[ Inputs ]'
            }
        ]
        data.append(SG_data)
    
    elif SG_option[0] == 5:
        SG_data = [
            {
                'name': 'GlobelBuffer',
                'type': 'gating',
                'target': 'Weights',
                'condition_on': '[ Inputs ]'
            }
        ]
        data.append(SG_data)

    elif SG_option[0] == 6:
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

    elif SG_option[0] == 7:
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

    if SG_option[1] == 0:
        data.append([])

    elif SG_option[1] == 1:
        data.append([])

    elif SG_option[1] == 2:
        SG_data = [
            {
                'name': 'GlobelBuffer',
                'type': 'skipping',
                'target': 'Outputs',
                'condition_on': '[ Inputs ]'
            }
        ]
        data.append(SG_data)

    elif SG_option[1] == 3:
        SG_data = [
            {
                'name': 'GlobelBuffer',
                'type': 'gating',
                'target': 'Outputs',
                'condition_on': '[ Inputs ]'
            }
        ]
        data.append(SG_data)

    elif SG_option[1] == 4:
        SG_data = [
            {
                'name': 'GlobelBuffer',
                'type': 'skipping',
                'target': 'Outputs',
                'condition_on': '[ Weights ]'
            }
        ]
        data.append(SG_data)
    
    elif SG_option[1] == 5:
        SG_data = [
            {
                'name': 'GlobelBuffer',
                'type': 'gating',
                'target': 'Outputs',
                'condition_on': '[ Weights ]'
            }
        ]
        data.append(SG_data)

    elif SG_option[1] == 6:
        SG_data = [
            {
                'name': 'GlobelBuffer',
                'type': 'skipping',
                'target': 'Outputs',
                'condition_on': '[ Weights, Inputs ]'
            }
        ]
        data.append(SG_data)

    elif SG_option[1] == 7:
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

    if SG_option[2] == 0:
        data.append([])

    elif SG_option[2] == 1:
        data.append([])

    elif SG_option[2] == 2:
        SG_data = [
            {
                'name': 'PE_buffer',
                'type': 'skipping',
                'target': 'Inputs',
                'condition_on': '[ Weights ]'
            }
        ]
        data.append(SG_data)

    elif SG_option[2] == 3:
        SG_data = [
            {
                'name': 'PE_buffer',
                'type': 'gating',
                'target': 'Inputs',
                'condition_on': '[ Weights ]'
            }
        ]
        data.append(SG_data)

    elif SG_option[2] == 4:
        SG_data = [
            {
                'name': 'PE_buffer',
                'type': 'skipping',
                'target': 'Weights',
                'condition_on': '[ Inputs ]'
            }
        ]
        data.append(SG_data)
    
    elif SG_option[2] == 5:
        SG_data = [
            {
                'name': 'PE_buffer',
                'type': 'gating',
                'target': 'Weights',
                'condition_on': '[ Inputs ]'
            }
        ]
        data.append(SG_data)

    elif SG_option[2] == 6:
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

    elif SG_option[2] == 7:
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

    if SG_option[3] == 0:
        data.append([])

    elif SG_option[3] == 1:
        data.append([])

    elif SG_option[3] == 2:
        SG_data = [
            {
                'name': 'PE_buffer',
                'type': 'skipping',
                'target': 'Outputs',
                'condition_on': '[ Inputs ]'
            }
        ]
        data.append(SG_data)

    elif SG_option[3] == 3:
        SG_data = [
            {
                'name': 'PE_buffer',
                'type': 'gating',
                'target': 'Outputs',
                'condition_on': '[ Inputs ]'
            }
        ]
        data.append(SG_data)

    elif SG_option[3] == 4:
        SG_data = [
            {
                'name': 'PE_buffer',
                'type': 'skipping',
                'target': 'Outputs',
                'condition_on': '[ Weights ]'
            }
        ]
        data.append(SG_data)
    
    elif SG_option[3] == 5:
        SG_data = [
            {
                'name': 'PE_buffer',
                'type': 'gating',
                'target': 'Outputs',
                'condition_on': '[ Weights ]'
            }
        ]
        data.append(SG_data)

    elif SG_option[3] == 6:
        SG_data = [
            {
                'name': 'PE_buffer',
                'type': 'skipping',
                'target': 'Outputs',
                'condition_on': '[ Weights, Inputs ]'
            }
        ]
        data.append(SG_data)

    elif SG_option[3] == 7:
        SG_data = [
            {
                'name': 'PE_buffer',
                'type': 'gating',
                'target': 'Outputs',
                'condition_on': '[ Weights, Inputs ]'
            }
        ]
        data.append(SG_data)


    SG_data = [
            {
                'name': 'PE_buffer',
                'type': 'skipping',
                'target': 'Outputs',
                'condition_on': '[ Weights, Inputs ]'
            }
        ]


    #====================================  生成  sparse-opt  =====================================

    context = {'data': data}

    with open(sparse_opt_temp_path,'r') as template_file_sparse_opt:
        template_content_sparse_opt = template_file_sparse_opt.read()
    
    template_sparse_opt = Template(template_content_sparse_opt)

    rendered_yaml_sparse_opt = template_sparse_opt.render(context)

    # 将结果写入文件
    with open(sparse_opt_output_path, 'w') as output:
        output.write(rendered_yaml_sparse_opt)


#evaluate  输入一个individual,返回优化目标的评估值  这个版本暂时只考虑maping  individual是numpy       待修改
def evaluate(individal):       

    #print("evaluating individual    ", individal)
                         
    dimension = ['C','M','N','P','Q','R','S']
    
    #----------------------------------------------------  解码mapping  ------------------------------------------------------
    print(individal[28:28+mapping_encoding_len].tolist())

    mapping = map_decode(individal[28:28+mapping_encoding_len].tolist())

    

    #print(dimention_factors)
    split_DRAM_to_GB = individal[19]
    split_GB_to_PEB = individal[20]

    permutation_DRAM_T_order = cantor_decode(individal[23],7)
    #print("per = ",permutation_DRAM_T_order)
    permutation_GB_T_order = cantor_decode(individal[24],7)
    #print("per = ",permutation_GB_T_order)
    permutation_GB_S_order = cantor_decode(individal[25],7)
    permutation_PEB_T_order = cantor_decode(individal[26],7)
    permutation_PEB_S_order = cantor_decode(individal[27],7)

    permutation_DRAM_T_str = (dimension[permutation_DRAM_T_order[0]] + dimension[permutation_DRAM_T_order[1]] + dimension[permutation_DRAM_T_order[2]]
                  + dimension[permutation_DRAM_T_order[3]] + dimension[permutation_DRAM_T_order[4]] + dimension[permutation_DRAM_T_order[5]]
                  + dimension[permutation_DRAM_T_order[6]])


    permutation_GB_T_str = (dimension[permutation_GB_T_order[0]] + dimension[permutation_GB_T_order[1]] + dimension[permutation_GB_T_order[2]]
                  + dimension[permutation_GB_T_order[3]] + dimension[permutation_GB_T_order[4]] + dimension[permutation_GB_T_order[5]]
                  + dimension[permutation_GB_T_order[6]])

    permutation_GB_S_str = (dimension[permutation_GB_S_order[0]] + dimension[permutation_GB_S_order[1]] + dimension[permutation_GB_S_order[2]]
                  + dimension[permutation_GB_S_order[3]] + dimension[permutation_GB_S_order[4]] + dimension[permutation_GB_S_order[5]]
                  + dimension[permutation_GB_S_order[6]])

    permutation_PEB_T_str = (dimension[permutation_PEB_T_order[0]] + dimension[permutation_PEB_T_order[1]] + dimension[permutation_PEB_T_order[2]]
                  + dimension[permutation_PEB_T_order[3]] + dimension[permutation_PEB_T_order[4]] + dimension[permutation_PEB_T_order[5]]
                  + dimension[permutation_PEB_T_order[6]])
    
    permutation_PEB_S_str = (dimension[permutation_PEB_S_order[0]] + dimension[permutation_PEB_S_order[1]] + dimension[permutation_PEB_S_order[2]]
                  + dimension[permutation_PEB_S_order[3]] + dimension[permutation_PEB_S_order[4]] + dimension[permutation_PEB_S_order[5]]
                  + dimension[permutation_PEB_S_order[6]])


    dimention_factors = [
        {
        'C': mapping[0][0],
        'M': mapping[0][1],
        'N': mapping[0][2],
        'P': mapping[0][3],
        'Q': mapping[0][4],
        'R': mapping[0][5],
        'S': mapping[0][6]
        },
        {
        'C': mapping[1][0],
        'M': mapping[1][1],
        'N': mapping[1][2],
        'P': mapping[1][3],
        'Q': mapping[1][4],
        'R': mapping[1][5],
        'S': mapping[1][6]
        },
        {
        'C': mapping[2][0],
        'M': mapping[2][1],
        'N': mapping[2][2],
        'P': mapping[2][3],
        'Q': mapping[2][4],
        'R': mapping[2][5],
        'S': mapping[2][6]
        },
        {
        'C': mapping[3][0],
        'M': mapping[3][1],
        'N': mapping[3][2],
        'P': mapping[3][3],
        'Q': mapping[3][4],
        'R': mapping[3][5],
        'S': mapping[3][6]
        },
        {
        'C': mapping[4][0],
        'M': mapping[4][1],
        'N': mapping[4][2],
        'P': mapping[4][3],
        'Q': mapping[4][4],
        'R': mapping[4][5],
        'S': mapping[4][6]
        }
        
    ]

    factors_DRAM_T_str = ('C='+str(dimention_factors[0]['C'])+' '+
                        'M='+str(dimention_factors[0]['M'])+' '+
                        'N='+str(dimention_factors[0]['N'])+' '+ 
                        'P='+str(dimention_factors[0]['P'])+' '+
                        'Q='+str(dimention_factors[0]['Q'])+' '+
                        'R='+str(dimention_factors[0]['R'])+' '+
                        'S='+str(dimention_factors[0]['S']))

    factors_GB_T_str = ('C='+str(dimention_factors[1]['C'])+' '+
                        'M='+str(dimention_factors[1]['M'])+' '+
                        'N='+str(dimention_factors[1]['N'])+' '+ 
                        'P='+str(dimention_factors[1]['P'])+' '+
                        'Q='+str(dimention_factors[1]['Q'])+' '+
                        'R='+str(dimention_factors[1]['R'])+' '+
                        'S='+str(dimention_factors[1]['S']))

    factors_GB_S_str = ('C='+str(dimention_factors[2]['C'])+' '+
                        'M='+str(dimention_factors[2]['M'])+' '+
                        'N='+str(dimention_factors[2]['N'])+' '+ 
                        'P='+str(dimention_factors[2]['P'])+' '+
                        'Q='+str(dimention_factors[2]['Q'])+' '+
                        'R='+str(dimention_factors[2]['R'])+' '+
                        'S='+str(dimention_factors[2]['S']))
    
    factors_PEB_T_str = ('C='+str(dimention_factors[3]['C'])+' '+
                        'M='+str(dimention_factors[3]['M'])+' '+
                        'N='+str(dimention_factors[3]['N'])+' '+ 
                        'P='+str(dimention_factors[3]['P'])+' '+
                        'Q='+str(dimention_factors[3]['Q'])+' '+
                        'R='+str(dimention_factors[3]['R'])+' '+
                        'S='+str(dimention_factors[3]['S']))
    
    factors_PEB_S_str = ('C='+str(dimention_factors[4]['C'])+' '+
                        'M='+str(dimention_factors[4]['M'])+' '+
                        'N='+str(dimention_factors[4]['N'])+' '+ 
                        'P='+str(dimention_factors[4]['P'])+' '+
                        'Q='+str(dimention_factors[4]['Q'])+' '+
                        'R='+str(dimention_factors[4]['R'])+' '+
                        'S='+str(dimention_factors[4]['S']))

    
    bypass_GB = individal[21]
    bypass_PEB = individal[22]

    #iwo = ['Weights','Inputs','Outputs']
    iwo_GB = []
    a = bypass_GB//4
    if a == 1:
        iwo_GB.append('Inputs')
    bypass_GB = bypass_GB%4
    b = bypass_GB//2
    if b == 1:
        iwo_GB.append('Weights')
    bypass_GB = bypass_GB%2
    if bypass_GB == 1:
        iwo_GB.append('Outputs')

    iwo_PEB = []
    a = bypass_PEB//4
    if a == 1:
        iwo_PEB.append('Inputs')
    bypass_PEB = bypass_PEB%4
    b = bypass_PEB//2
    if b == 1:
        iwo_PEB.append('Weights')
    bypass_PEB = bypass_PEB%2
    if bypass_PEB == 1:
        iwo_PEB.append('Outputs')



    mapping_variables = {
    'permutation_DRAM_T': permutation_DRAM_T_str,
    'factors_DRAM_T': factors_DRAM_T_str,

    'permutation_GB_T': permutation_GB_T_str,
    'factors_GB_T': factors_GB_T_str,

    'permutation_GB_S': permutation_GB_S_str,
    'split_DRAM_to_GB': split_DRAM_to_GB,
    'factors_GB_S': factors_GB_S_str,

    'permutation_PEB_T': permutation_PEB_T_str,
    'factors_PEB_T': factors_PEB_T_str,

    'permutation_PEB_S': permutation_PEB_S_str,
    'split_GB_to_PEB': split_GB_to_PEB,
    'factors_PEB_S': factors_PEB_S_str,

    'iwo_GB':iwo_GB,
    'iwo_PEB':iwo_PEB

    }

    # ------------------------------------  mapping  输入文件生成  -------------------------------------------
    # 读取模板文件
    #print(os.getcwd())
    
    this_directory  = "/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single"
    #print("this_direc = ",this_directory)
    #os.chdir(this_directory)
    #print("current      ",os.getcwd())
    with open(os.path.join(this_directory, "yamls", "mapping_conv.j2"), 'r') as template_file_mapping:
        template_content_mapping = template_file_mapping.read()

    # 创建 Jinja2 模板对象
    template_mapping = Template(template_content_mapping)

    # 渲染模板并生成 YAML 内容
    rendered_yaml_mapping = template_mapping.render(mapping_variables)

    # 将生成的 YAML 内容写入文件
    with open(os.path.join(this_directory,'yamls','mapping_conv_output.yaml'), 'w') as output_file_mapping:
        output_file_mapping.write(rendered_yaml_mapping)

    generate_sparse_opt(individal[0:5],individal[5:10],individal[15:19],individal[28:])

#----------------------------------------------调用sparse_loop-------------------------------------------------------

    
    
    #problem_template_path = os.path.join(this_directory,"yamls", "workload_example.yaml")
    arch_path = os.path.join(this_directory, "yamls","arch_"+PLATFORM+".yaml")
    #component_path = os.path.join(this_directory, "..", "multiSCNN","fig13_dstc_setup","input_specs",  "compound_components.yaml")
    mapping_output_path = os.path.join(this_directory,"yamls", "mapping_conv_output.yaml")
    #mapper_path = os.path.join(this_directory, "mapper.yaml")
    #sparse_opt_path = os.path.join(this_directory, "sparse_opt_output.yaml")
    #sparse_opt_path = os.path.join(this_directory, "..", "multiSCNN","single_core_optimization",  "yaml_gen", "sparse_opt_output.yaml")



    ert_path = os.path.join(this_directory,"..","..", "evaluation_setups","fig13_dstc_setup","input_specs",  "ERT.yaml")
    art_path = os.path.join(this_directory, "..","..", "evaluation_setups","fig13_dstc_setup","input_specs",  "ART.yaml")
    #print(ert_path)
    
    # load all yaml input specs
    problem_template = yaml.load(open(problem_template_path), Loader = yaml.SafeLoader)
    arch = yaml.load(open(arch_path), Loader = yaml.SafeLoader)
    #components = yaml.load(open(component_path), Loader = yaml.SafeLoader)
    mapping = yaml.load(open(mapping_output_path), Loader = yaml.SafeLoader)
    #mapper = yaml.load(open(mapper_path), Loader = yaml.SafeLoader)
    sparse_opt = yaml.load(open(sparse_opt_output_path), Loader = yaml.SafeLoader)
    
    output_base_dir = os.path.join(this_directory, "ES_search_outputs")

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
    
    
    job_name  = "ES_searching"
    

#---------------------------------------------------调用,生成output---------------------------------------------------------
    run_timeloop(job_name, aggregated_input, output_base_dir , ert_path, art_path)


#------------------------------------------------------读取数据--------------------------------------------------------------
#++++++++++++++++++++++++++++++++++++++++++++++++本实验优化目标+++++++++++++++++++++++++++++++++++++++++
    stat_type = OBJECT
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
    base_output_dir = os.path.join(this_directory, "ES_search_outputs")
    output_file_path = os.path.join(base_output_dir,'outputs', "timeloop-model.map+stats.xml")
    if os.path.exists(output_file_path):
        job_output_stats = parse_timeloop_stats(output_file_path)
        os.remove(output_file_path)
        if stat_type == "edp":
            fitness = -job_output_stats["cycles"]*job_output_stats["energy_pJ"]  #######################################
        else:
            fitness = -job_output_stats[stat_type]
        formatted_number = f"{fitness:.2e}"
        print("current result = ",formatted_number)
    else:
        fitness = -10000000000000000000
    return fitness

def evaluate_state(state):                                    #indi是长度63的np array


    dimension = ['C','M','N','P','Q','R','S']
    
    #----------------------------------------------------  解码mapping  ------------------------------------------------------

    mapping = state['array']
    
    

    #print(dimention_factors)
    split_DRAM_to_GB = state['split'][0]
    split_GB_to_PEB = state['split'][1]

    permutation_DRAM_T_order = state['permutations'][0]
    #print("per = ",permutation_DRAM_T_order)
    permutation_GB_T_order = state['permutations'][1]
    #print("per = ",permutation_GB_T_order)
    permutation_GB_S_order = state['permutations'][2]
    permutation_PEB_T_order = state['permutations'][3]
    permutation_PEB_S_order = state['permutations'][4]

    permutation_DRAM_T_str = (dimension[permutation_DRAM_T_order[0]] + dimension[permutation_DRAM_T_order[1]] + dimension[permutation_DRAM_T_order[2]]
                  + dimension[permutation_DRAM_T_order[3]] + dimension[permutation_DRAM_T_order[4]] + dimension[permutation_DRAM_T_order[5]]
                  + dimension[permutation_DRAM_T_order[6]])


    permutation_GB_T_str = (dimension[permutation_GB_T_order[0]] + dimension[permutation_GB_T_order[1]] + dimension[permutation_GB_T_order[2]]
                  + dimension[permutation_GB_T_order[3]] + dimension[permutation_GB_T_order[4]] + dimension[permutation_GB_T_order[5]]
                  + dimension[permutation_GB_T_order[6]])

    permutation_GB_S_str = (dimension[permutation_GB_S_order[0]] + dimension[permutation_GB_S_order[1]] + dimension[permutation_GB_S_order[2]]
                  + dimension[permutation_GB_S_order[3]] + dimension[permutation_GB_S_order[4]] + dimension[permutation_GB_S_order[5]]
                  + dimension[permutation_GB_S_order[6]])

    permutation_PEB_T_str = (dimension[permutation_PEB_T_order[0]] + dimension[permutation_PEB_T_order[1]] + dimension[permutation_PEB_T_order[2]]
                  + dimension[permutation_PEB_T_order[3]] + dimension[permutation_PEB_T_order[4]] + dimension[permutation_PEB_T_order[5]]
                  + dimension[permutation_PEB_T_order[6]])
    
    permutation_PEB_S_str = (dimension[permutation_PEB_S_order[0]] + dimension[permutation_PEB_S_order[1]] + dimension[permutation_PEB_S_order[2]]
                  + dimension[permutation_PEB_S_order[3]] + dimension[permutation_PEB_S_order[4]] + dimension[permutation_PEB_S_order[5]]
                  + dimension[permutation_PEB_S_order[6]])


    dimention_factors = [
        {
        'C': mapping[0][0],
        'M': mapping[0][1],
        'N': mapping[0][2],
        'P': mapping[0][3],
        'Q': mapping[0][4],
        'R': mapping[0][5],
        'S': mapping[0][6]
        },
        {
        'C': mapping[1][0],
        'M': mapping[1][1],
        'N': mapping[1][2],
        'P': mapping[1][3],
        'Q': mapping[1][4],
        'R': mapping[1][5],
        'S': mapping[1][6]
        },
        {
        'C': mapping[2][0],
        'M': mapping[2][1],
        'N': mapping[2][2],
        'P': mapping[2][3],
        'Q': mapping[2][4],
        'R': mapping[2][5],
        'S': mapping[2][6]
        },
        {
        'C': mapping[3][0],
        'M': mapping[3][1],
        'N': mapping[3][2],
        'P': mapping[3][3],
        'Q': mapping[3][4],
        'R': mapping[3][5],
        'S': mapping[3][6]
        },
        {
        'C': mapping[4][0],
        'M': mapping[4][1],
        'N': mapping[4][2],
        'P': mapping[4][3],
        'Q': mapping[4][4],
        'R': mapping[4][5],
        'S': mapping[4][6]
        }
        
    ]

    factors_DRAM_T_str = ('C='+str(dimention_factors[0]['C'])+' '+
                        'M='+str(dimention_factors[0]['M'])+' '+
                        'N='+str(dimention_factors[0]['N'])+' '+ 
                        'P='+str(dimention_factors[0]['P'])+' '+
                        'Q='+str(dimention_factors[0]['Q'])+' '+
                        'R='+str(dimention_factors[0]['R'])+' '+
                        'S='+str(dimention_factors[0]['S']))

    factors_GB_T_str = ('C='+str(dimention_factors[1]['C'])+' '+
                        'M='+str(dimention_factors[1]['M'])+' '+
                        'N='+str(dimention_factors[1]['N'])+' '+ 
                        'P='+str(dimention_factors[1]['P'])+' '+
                        'Q='+str(dimention_factors[1]['Q'])+' '+
                        'R='+str(dimention_factors[1]['R'])+' '+
                        'S='+str(dimention_factors[1]['S']))

    factors_GB_S_str = ('C='+str(dimention_factors[2]['C'])+' '+
                        'M='+str(dimention_factors[2]['M'])+' '+
                        'N='+str(dimention_factors[2]['N'])+' '+ 
                        'P='+str(dimention_factors[2]['P'])+' '+
                        'Q='+str(dimention_factors[2]['Q'])+' '+
                        'R='+str(dimention_factors[2]['R'])+' '+
                        'S='+str(dimention_factors[2]['S']))
    
    factors_PEB_T_str = ('C='+str(dimention_factors[3]['C'])+' '+
                        'M='+str(dimention_factors[3]['M'])+' '+
                        'N='+str(dimention_factors[3]['N'])+' '+ 
                        'P='+str(dimention_factors[3]['P'])+' '+
                        'Q='+str(dimention_factors[3]['Q'])+' '+
                        'R='+str(dimention_factors[3]['R'])+' '+
                        'S='+str(dimention_factors[3]['S']))
    
    factors_PEB_S_str = ('C='+str(dimention_factors[4]['C'])+' '+
                        'M='+str(dimention_factors[4]['M'])+' '+
                        'N='+str(dimention_factors[4]['N'])+' '+ 
                        'P='+str(dimention_factors[4]['P'])+' '+
                        'Q='+str(dimention_factors[4]['Q'])+' '+
                        'R='+str(dimention_factors[4]['R'])+' '+
                        'S='+str(dimention_factors[4]['S']))

    
    bypass_GB = state['bypass_choice'][0]
    bypass_PEB = state['bypass_choice'][1]

    #iwo = ['Weights','Inputs','Outputs']
    iwo_GB = []
    a = bypass_GB//4
    if a == 1:
        iwo_GB.append('Inputs')
    bypass_GB = bypass_GB%4
    b = bypass_GB//2
    if b == 1:
        iwo_GB.append('Weights')
    bypass_GB = bypass_GB%2
    if bypass_GB == 1:
        iwo_GB.append('Outputs')

    iwo_PEB = []
    a = bypass_PEB//4
    if a == 1:
        iwo_PEB.append('Inputs')
    bypass_PEB = bypass_PEB%4
    b = bypass_PEB//2
    if b == 1:
        iwo_PEB.append('Weights')
    bypass_PEB = bypass_PEB%2
    if bypass_PEB == 1:
        iwo_PEB.append('Outputs')



    mapping_variables = {
    'permutation_DRAM_T': permutation_DRAM_T_str,
    'factors_DRAM_T': factors_DRAM_T_str,

    'permutation_GB_T': permutation_GB_T_str,
    'factors_GB_T': factors_GB_T_str,

    'permutation_GB_S': permutation_GB_S_str,
    'split_DRAM_to_GB': split_DRAM_to_GB,
    'factors_GB_S': factors_GB_S_str,

    'permutation_PEB_T': permutation_PEB_T_str,
    'factors_PEB_T': factors_PEB_T_str,

    'permutation_PEB_S': permutation_PEB_S_str,
    'split_GB_to_PEB': split_GB_to_PEB,
    'factors_PEB_S': factors_PEB_S_str,

    'iwo_GB':iwo_GB,
    'iwo_PEB':iwo_PEB

    }

    # ------------------------------------  mapping  输入文件生成  -------------------------------------------
    # 读取模板文件
    #print(os.getcwd())
    
    #this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
    this_directory = '/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single'
    #print("this_direc = ",this_directory)
    os.chdir('/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single')
    #print(os.getcwd())
    with open('/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single/yamls/mapping_conv.j2', 'r') as template_file_mapping:
        template_content_mapping = template_file_mapping.read()

    # 创建 Jinja2 模板对象
    template_mapping = Template(template_content_mapping)

    # 渲染模板并生成 YAML 内容
    rendered_yaml_mapping = template_mapping.render(mapping_variables)

    # 将生成的 YAML 内容写入文件
    with open('/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single/yamls/mapping_conv_output_MCTS.yaml', 'w') as output_file_mapping:
        output_file_mapping.write(rendered_yaml_mapping)

#----------------------------------------------调用sparse_loop-------------------------------------------------------

    
    
    #problem_template_path = os.path.join(this_directory, "yamls", JOB , WORK_LOAD + ".yaml")
    arch_path = os.path.join(this_directory, "yamls","arch_"+PLATFORM+".yaml")
    #component_path = os.path.join(this_directory, "..", "multiSCNN","fig13_dstc_setup","input_specs",  "compound_components.yaml")
    mapping_path = os.path.join(this_directory,"yamls", "mapping_conv_output_MCTS.yaml")
    #mapper_path = os.path.join(this_directory, "mapper.yaml")
    #sparse_opt_path = os.path.join(this_directory, "sparse_opt_output.yaml")
    #sparse_opt_path = os.path.join(this_directory, "..", "multiSCNN","single_core_optimization",  "yaml_gen", "sparse_opt_output.yaml")



    ert_path = os.path.join(this_directory,"..","..", "evaluation_setups","fig13_dstc_setup","input_specs",  "ERT.yaml")
    art_path = os.path.join(this_directory, "..","..", "evaluation_setups","fig13_dstc_setup","input_specs",  "ART.yaml")
    #print(ert_path)
    
    # load all yaml input specs
    problem_template = yaml.load(open(problem_template_path), Loader = yaml.SafeLoader)
    arch = yaml.load(open(arch_path), Loader = yaml.SafeLoader)
    #components = yaml.load(open(component_path), Loader = yaml.SafeLoader)
    mapping = yaml.load(open(mapping_path), Loader = yaml.SafeLoader)
    #mapper = yaml.load(open(mapper_path), Loader = yaml.SafeLoader)

    generate_sparse_opt(state['inputCF'],state['weightCF'],state['SG_option'],map_encode(state['array']))

    sparse_opt = yaml.load(open(sparse_opt_output_path), Loader = yaml.SafeLoader)
    
    output_base_dir = os.path.join(this_directory, "MCTS_search_outputs")

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
    
    
    job_name  = "MCTS_searching"
    

#---------------------------------------------------调用,生成output---------------------------------------------------------
    run_timeloop(job_name, aggregated_input, os.path.join(output_base_dir, job_name), ert_path, art_path)


#------------------------------------------------------读取数据--------------------------------------------------------------
#++++++++++++++++++++++++++++++++++++++++++++++++本实验优化目标+++++++++++++++++++++++++++++++++++++++++
    stat_type = OBJECT
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
    base_output_dir = os.path.join(this_directory, "MCTS_search_outputs")
    output_file_path = os.path.join(base_output_dir, job_name, 'outputs', "timeloop-model.map+stats.xml")
    if os.path.exists(output_file_path):
        job_output_stats = parse_timeloop_stats(output_file_path)
        os.remove(output_file_path)
        if stat_type == "edp":
            fitness = job_output_stats["cycles"]*job_output_stats["energy_pJ"]
        else:
            fitness = job_output_stats[stat_type]
    else:
        fitness = 10000000000000000000
    return fitness

def target(map_code,perm_code,bypass_choice,split,cf,SG):   #这里的code都是Numpy的  map是长度为mapping_encoding_len的一维，perm长度为5          优化目标函数
    
    root_state = {
    'permutations': [[6,4,0,1,5,2,3],[1,2,4,0,5,6,3],[1,6,2,3,0,5,4],[5,3,4,0,1,6,2],[0,5,6,3,1,2,4]],
    'bypass_choice': [7, 7],
    'array': np.array([[1,6,1,1,3,1,2],[1,2,2,1,3,1,1],[1,8,2,3,1,1,2],[1,1,1,9,6,4,1],[3,1,1,2,1,3,3]]),
    'split': [1,2],
    'inputCF': [],
    'weightCF': [],
    'SG_option': []
    }
    state = root_state
    state['permutations'] = [cantor_decode(perm_code[i],7) for i in range(5) ]
    state['array'] = np.array(map_decode(map_code))
    state['bypass_choice'] = bypass_choice
    state['split'] = split
    state['inputCF'] = cf[0:5]
    state['weightCF'] = cf[5:10]
    state['SG_option'] = SG
    a = evaluate_state(state)
    if a != 10000000000000000000:
        print(a)
    return a

DNA_SIZE = 28+mapping_encoding_len            # DNA 

'''
[0,4]          Input 压缩格式           uop---0    b---1    cp---2    rle---3
[5,9]         Weight 压缩格式          uop---0    b---1    cp---2    rle---3
[10,14]         Output 压缩格式          uop---0    b---1    cp---2    rle---3
[15,16]         GlobleBuffer 中的 skip/gate
[17,18]            PE_Buffer 中的 skip/gate
[19,20]                    split 位置
[21,22]                    bypass_choice
[23,27]                    cantor编码的permutation
[28,28+mapping_encoding_len-1]         map_size  编码

'''
def roulette_wheel_selection(numbers):        #输入适应度表，输出一个随概率选择的索引号
    # 计算总的适应度值（这里假设适应度值就是数字本身）
    total_fitness = sum(numbers)
    # 计算每个数字的选择概率（适应度值越高，被选中的概率越大）
    probabilities = [num / total_fitness for num in numbers]

    # 生成一个随机概率值
    random_prob = random.random()
    cumulative_prob = 0
    for i, prob in enumerate(probabilities):
        cumulative_prob += prob
        if random_prob <= cumulative_prob:
            selected_idx = i
            break     
    return selected_idx

def population_initialize(inin_path):
    
    pop_list = []
    # 创建一个长度为10的空列表来存储每行第一个数
    first_numbers_devided_by1 = []
    with open(inin_path, mode='r') as file:
        reader = csv.reader(file)
        # 逐行读取CSV文件
        for row in reader:
            # 获取每行的第一个数，并将其添加到first_numbers列表中
            first_number = int(float(row[0]))  # 假设第一个数是整数
            first_numbers_devided_by1.append(1/first_number)
        
        for i in range(POP_SIZE):
            indi = [0 for m in range(DNA_SIZE)]
            #print("DNA_SIZE = ",DNA_SIZE)
            #print("lenth of indi = ",len(indi))
                                                                                 #初始化mapping
            row_idx = roulette_wheel_selection(first_numbers_devided_by1)
            #print("ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc   row_idx = ",row_idx)
            df = pd.read_csv(inin_path,header=None)
            #print("DataFrame的行数:", len(df))
            row = df.iloc[row_idx].tolist()
            data_list = row[1:]
            #indi[19:DNA_SIZE] = data_list #########################################################################
            indi[19:DNA_SIZE] = [int(d) for d in data_list]
            #print("lenth of indi = ",len(indi))
                                                                                 #初始化compress_format
            for i in range(15):
                indi[i] = random.randint(1,3)
            for i in range(15,19):
                indi[i] = random.randint(0,7)
            pop_list.append(indi)
    population = np.array(pop_list)

    print("=================================================================  successfully initialized  ====================================================" )

    return population


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
        crossover_point = np.random.randint(79, 83+mapping_encoding_len-1)                               #决定染色体交叉点
        kid = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        kids.append(kid)
    return np.array(kids)

def mutate(parents, n_kid):                    #暂时只用mutate产生子代
    kids = []
    for i in range(n_kid):
        selected_indices = np.random.randint(0,49)
        parent = parents[selected_indices]
        kid = parent.copy()
        mutate_choice = np.random.randint(0,2)
        if mutate_choice == 0:
            cantor_mutate_position = np.random.randint(78,83)        #康托展开变异位置
            p_or_m = np.random.randint(0,2)               #加还是减
            if kid[cantor_mutate_position] == 0:
                kid[cantor_mutate_position] = (kid[cantor_mutate_position]+1)%5040
            else:
                if p_or_m == 0:
                    kid[cantor_mutate_position] = (kid[cantor_mutate_position]+1)%5040
                else:
                    kid[cantor_mutate_position] = (kid[cantor_mutate_position]-1)%5040
        else:
            mapsize_mutate_position = np.random.randint(83,83+mapping_encoding_len)        #mapsize变异位置
            p_or_m = np.random.randint(0,2)               #加还是减
            if kid[mapsize_mutate_position] == 0:
                kid[mapsize_mutate_position] = (kid[mapsize_mutate_position]+1)%5
            else:
                if p_or_m == 0 :
                    kid[mapsize_mutate_position] = (kid[mapsize_mutate_position]+1)%5
                else:
                    kid[mapsize_mutate_position] = (kid[mapsize_mutate_position]-1)%5

        kids.append(kid)
    return np.array(kids)

#变异幅度大的版本
def mutate_big(parents, n_kid):                    #暂时只用mutate产生子代
    kids = []
    for i in range(n_kid):
        selected_indices = np.random.randint(0,49)
        parent = parents[selected_indices]
        kid = parent.copy()
        mutate_choice = np.random.randint(0,2)
        if mutate_choice == 0:
            cantor_mutate_position = np.random.randint(78,83)        #康托展开变异位置
            kid[cantor_mutate_position] = np.random.randint(1,5040)
        else:
            mapsize_mutate_position = np.random.randint(83,83+mapping_encoding_len)        #mapsize变异位置
            kid[mapsize_mutate_position] = np.random.randint(0,5)

        kids.append(kid)
    return np.array(kids)

def crossover(parents, n_kid):
    kids = []
    for i in range(n_kid):
        selected_indices = np.random.choice(parents.shape[0], size=2, replace=False)
        parent1 = parents[selected_indices[0]]
        parent2 = parents[selected_indices[1]]
        crossover_point = np.random.randint(1, DNA_SIZE-1)                               #决定染色体交叉点
        kid = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        kids.append(kid)
    return np.array(kids)

def strenthen(individual):
    initial_indi = {
        "map_code": individual[28:],  
        "perm_code": individual[23:28],
        "bypass_choice" : individual[21:23],
        "split"  :  individual[19:21],
        "cf": individual[0:15],
        "SG": individual[15:19]
    }


    #print(initial_indi)

    #print("==================================================  before_set ======================================================")

    instrum = ng.p.Instrumentation(
        map_code = ng.p.Array(shape=(mapping_encoding_len,)).set_integer_casting().set_bounds(lower=0, upper=4),
        perm_code = ng.p.Array(shape=(5,)).set_integer_casting().set_bounds(lower=0, upper=5040),
        bypass_choice = ng.p.Array(shape=(2,)).set_integer_casting().set_bounds(lower=0, upper=7),
        split = ng.p.Array(shape=(2,)).set_integer_casting().set_bounds(lower=0, upper=7),
        cf = ng.p.TransitionChoice(range(1, 4), repetitions=15),
        SG = ng.p.Array(shape=(4,)).set_integer_casting().set_bounds(lower=0, upper=7),
        )
    #print("==================================================  after_set  ================================================")
    #optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=100)
    #optimizer = ng.optimizers.DifferentialEvolution()
    #optimizer = ng.optimizers.TwoPointsDE(parametrization=instrum, budget=1000, num_workers=10)
    #optimizer = ng.optimizers.TBPSA(parametrization=instrum, budget=1000, num_workers=10)
    #optimizer = ng.optimizers.CMA(parametrization=instrum, budget=1000, num_workers=10)
    #optimizer = ng.optimizers.PortfolioDiscreteOnePlusOne(parametrization=instrum, budget=1000, num_workers=10)
    optimizer = ng.optimizers.OnePlusOne(parametrization=instrum, budget=10, num_workers=10)
    
    optimizer.suggest(map_code = initial_indi['map_code'], 
                    perm_code = initial_indi['perm_code'],
                    bypass_choice = initial_indi['bypass_choice'],
                    split = initial_indi['split'],
                    cf = initial_indi['cf'],
                    SG = initial_indi['SG']
                    )
    #print("==================================================  after_suggest  ================================================")
    #init_popu = [initial_indi for i in range(10) ] 
    #optimizer.internal_population = init_popu

    for _ in range(optimizer.budget):
        #print("==================================================  before_ask ======================================================")
        x = optimizer.ask()
        #print("==================================================  after ask  ================================================")
        if _ == 0 :
            print('==========================================================     OnePlusOne   initialize   ================================================================')
            print(x.kwargs)
        else:
            print('======================================================={} st  OnePlusOne search point============================================================'.format(_))
            print(x.kwargs)
        loss = target(*x.args, **x.kwargs)
        optimizer.tell(x, loss)

    recommendation = optimizer.provide_recommendation()

    new_individual = individual
    new_individual[28:] = recommendation.value[1]['map_code']
    new_individual[23:28] = recommendation.value[1]['perm_code']
    new_individual[21:23] = recommendation.value[1]['bypass_choice']
    new_perform = target(
                         recommendation.value[1]['map_code'],
                         recommendation.value[1]['perm_code'],
                         recommendation.value[1]['bypass_choice'],
                         recommendation.value[1]['split'],
                         recommendation.value[1]['cf'],
                         recommendation.value[1]['SG']
                         )
    if new_perform > sys.maxsize:
            new_perform = sys.maxsize
    return new_individual, - new_perform                                #转换成fitness要带负号

def envolve(population,fitness):                                     
    selected_population, parents_fitness = select_order(population, fitness)   #从上一代中挑出50个
    #kids = mutate(selected_population, N_KID)
    for i in range(10):
        id = random.randint(0,selected_population.shape[0]-1)
        selected_population[id], parents_fitness[id] = strenthen(selected_population[id])
    kids = crossover(selected_population, N_KID)                               #交叉
    kids_fitness = np.array([evaluate(indi) for indi in kids])                                                    #生成子代       
    next_generation_candidates = np.concatenate([selected_population, kids])  
    fitness = np.append(parents_fitness,kids_fitness)
    return next_generation_candidates, fitness


def test_main():
    #initialize_path = "/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single/outputs/example/outputs/initialize_mappings_found.csv"
    initialize_path = os.path.join("/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single", JOB + "_mapping_found", WORK_LOAD + "_initialize.csv" )
    population = population_initialize(initialize_path)
    print("======================================================================================================   population initialized !   ========================================================")
    indi = population[0]
    print( "indi = ",indi )
    print( "lenth of indi = ", len(indi) )
    per = evaluate(indi)
    print(per)

def main():
    initialize_path = os.path.join("/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single", JOB + "_mapping_found", WORK_LOAD + "_initialize.csv" )
    population = population_initialize(initialize_path)
    
    fitness = np.array([evaluate(indi) for indi in population])
    print("======================================================================================================   population initialized !   ========================================================")
    generation_best_individual = []
    generation_best_performance = []

    for i in range(N_GENERATIONS):
        population,fitness = envolve(population,fitness)
        best_indi_index = np.argmax(fitness)
        generation_best_individual.append(population[best_indi_index,:].tolist())
        generation_best_performance.append(-fitness[best_indi_index])
        print("=====================================================================\n")
        print("GENERATION = ",i)
        print("Best performance is",-fitness[best_indi_index])
        print("=====================================================================\n")
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    np.savetxt(this_directory + formatted_now +'_best_individual.txt', generation_best_individual)
    np.savetxt(this_directory + formatted_now +'_best_performance.txt', generation_best_performance)

if __name__ == "__main__":
    main()
    #test_main()


