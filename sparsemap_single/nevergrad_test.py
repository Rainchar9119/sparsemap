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
import inspect
from copy import deepcopy
import shutil, argparse, pprint, sys, subprocess






'''
def square(x):
    return sum((x - 0.5) ** 2+x**6)

# optimization on x as an array of shape (2,)
optimizer = ng.optimizers.NGOpt(parametrization=5, budget=100)
recommendation = optimizer.minimize(square)  # best value
print(recommendation.value)
'''


GLOBAL_MIN = 100000000000000000000000

OVERWRITE = 1
JOB = "vgg_conv"              #高层次
WORK_LOAD = "workload_vgg_layer1"        #低层次

dimensions = {'C': 3, 'M': 64, 'N': 1, 'P': 216, 'Q': 216, 'R': 3, 'S': 3}

problem_template_path = os.path.join("/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single/yamls", JOB , WORK_LOAD + ".yaml")
PLATFORM = "cloud"
OBJECT = 'edp'
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
sparse_opt_output_path = os.path.join(this_directory, "yamls",  "sparse_opt_example_conv.yaml")

with open(problem_template_path, 'r') as file:
        workload_data = yaml.safe_load(file)
        d = workload_data['problem']['instance']
        dimensions = d
print(dimensions)

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


def map_encode(size):
    size_code = []
    for i in range(7):
        for j in range(len_list[i]):
            factor = size[j][i] // factor_list[i][j]
            index = 0
            while factor > 1:
                if factor % factor_list[i][j] == 0:
                    factor //= factor_list[i][j]
                    index += 1
                else:
                    break
            size_code.extend([index] * (j + 1))
    return size_code


#code = [4,2,3,2,1,4,1,2,0,0,1,3,4,0,2,1,2,1,2,3,1,2,1]
#print(decode(code))

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
        p.communicate(timeout=0.2) 
    except subprocess.TimeoutExpired:
        p.terminate()
        #this_file_path = '/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single/MCTS_initialize.py'
        #this_directory = os.path.dirname(this_file_path)
        #base_output_dir = os.path.join(this_directory, "MCTS_search_outputs")
        #output_file_path = os.path.join(base_output_dir, job_name, 'outputs', "timeloop-model.map+stats.xml")
        path = '/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single/MCTS_search_outputs/MCTS_searching/outputs/timeloop-model.map+stats.xml'
        if os.path.exists(path):
            os.remove(path)
        return 



def evaluate(state):                                    #indi是长度63的np array


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

#dimension = ['C','M','N','P','Q','R','S']


root_state = {
    'permutations': [[6,4,0,1,5,2,3],[1,2,4,0,5,6,3],[1,6,2,3,0,5,4],[5,3,4,0,1,6,2],[0,5,6,3,1,2,4]],
    'bypass_choice': [7, 7],
    'array': np.array([[1,6,1,1,3,1,2],[1,2,2,1,3,1,1],[1,8,2,3,1,1,2],[1,1,1,9,6,4,1],[3,1,1,2,1,3,3]]),
    'split': [1,2]
}

def target(map_code,perm_code,bypass_choice,split):   #这里的code都是Numpy的  map是长度为mapping_encoding_len的一维，perm长度为5          优化目标函数
    global GLOBAL_MIN
    state = root_state
    state['permutations'] = [cantor_decode(perm_code[i],7) for i in range(5) ]
    state['array'] = np.array(map_decode(map_code))
    state['bypass_choice'] = bypass_choice
    state['split'] = split
    a = evaluate(state)
    if a != 10000000000000000000:
        print(a)
        if a <  GLOBAL_MIN :
            GLOBAL_MIN = a
    return a


def main():
    initial_indi = {
        "map_code": [2 ,3 ,4 ,4 ,4 ,4 ,4 ,0 ,0 ,0 ,1 ,2 ,2 ,0 ,3 ,0 ,1 ,2 ,1 ,1 ,2],  # 使用相同的初始值示例
        
        #"perm_code": [cantor_encode(root_state['permutations'][0]), 
         #           cantor_encode(root_state['permutations'][1]),
         #           cantor_encode(root_state['permutations'][2]), 
         #           cantor_encode(root_state['permutations'][3]),
         #           cantor_encode(root_state['permutations'][4])]  # 使用特定的初始值示例

        "perm_code":[2524 ,4154 ,2270 ,3017 , 811] ,
        "bypass_choice" : [5,5],
        "split" : [1,1]
    }
   
    instrum = ng.p.Instrumentation(
        map_code = ng.p.Array(shape=(mapping_encoding_len,)).set_integer_casting().set_bounds(lower=0, upper=4),
        perm_code = ng.p.Array(shape=(5,)).set_integer_casting().set_bounds(lower=0, upper=5040),
        bypass_choice = ng.p.Array(shape=(2,)).set_integer_casting().set_bounds(lower=0, upper=7),
        split = ng.p.Array(shape=(2,)).set_integer_casting().set_bounds(lower=0, upper=3)
        )


    #print(child)

    #optimizer = ng.optimizers.PSO(parametrization=instrum, budget=20000)
    #optimizer = ng.optimizers.DifferentialEvolution()
    optimizer = ng.optimizers.RandomSearch(parametrization=instrum, budget=20000, num_workers=10)
    #optimizer = ng.optimizers.TBPSA(parametrization=instrum, budget=20000, num_workers=10)
    #optimizer = ng.optimizers.CMA(parametrization=instrum, budget=20000, num_workers=10)
    #optimizer = ng.optimizers.PortfolioDiscreteOnePlusOne(parametrization=instrum, budget=1000, num_workers=10)
    #optimizer = ng.optimizers.OnePlusOne(parametrization=instrum, budget=20000, num_workers=10)

    
    optimizer.suggest(map_code = initial_indi['map_code'],perm_code = initial_indi['perm_code'],bypass_choice = initial_indi['bypass_choice'],split = initial_indi['split'])
    
    
    for _ in range(optimizer.budget):
        x = optimizer.ask()
        if _ == 0 :
            print('==========================================================initialize================================================================')
            print(x.kwargs)
        else:
            print('======================================================={} st search point============================================================'.format(_))
            print(x.kwargs)
        loss = target(*x.args, **x.kwargs)
        optimizer.tell(x, loss)

    recommendation = optimizer.provide_recommendation()
    
    #recommendation = optimizer.minimize(target)
    #print(recommendation)
    print(recommendation.value)
    #print(target(recommendation.value[1]['map_code'],recommendation.value[1]['perm_code'],recommendation.value[1]['bypass_choice'],recommendation.value[1]['split']))
    #print(optimizer.llambda)
    #population = optimizer.ask()
    print("best performance    =    ",GLOBAL_MIN)
    '''
    for i in range(len(population)):
        indi = population[i]
        print("================================  individual {} ===============================".format(i))
        print(indi.value)
    '''
    #recommendation = optimizer.minimize(target)
    #print(recommendation.value)
    #print(target(recommendation.value[1]['map_code'],recommendation.value[1]['perm_code']))



if __name__ == "__main__":
    main()
    '''
    map_code =  [4,0,1,2,2,2,0,1,2,4,2,3,3,3,3,0,1,3,3,4,0,2,4] 
    perm_code =[cantor_encode(root_state['permutations'][0]), 
                    cantor_encode(root_state['permutations'][1]),
                    cantor_encode(root_state['permutations'][2]), 
                    cantor_encode(root_state['permutations'][3]),
                    cantor_encode(root_state['permutations'][4])] 
    target(map_code,perm_code)
    '''


