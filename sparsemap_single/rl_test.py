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
import random
from datetime import datetime
import torch
import gym
import stable_baselines3 as sb3
from gym import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO


OVERWRITE = 1
JOB = "resnet_conv"              #高层次
WORK_LOAD = "workload_resnet_conv4"        #低层次


SPLIT = [1,2]
MAP_CODE = [0,1,4,1,4,1,1,2,4,2,4,2,4,2,1,2,1,1,1,2,2,2,3,0]
PERM_CODE = [619,2524,3052,3619,91]
BYPASS = [4,2]

dimensions = {'C': 3, 'M': 96, 'N': 4, 'P': 54, 'Q': 54, 'R': 12, 'S': 12}

problem_template_path = os.path.join("/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single/yamls", JOB , WORK_LOAD + ".yaml")
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
        index = int(encoded // factorial)
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


def evaluate(state):                                    #state包含四部分


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

    
    
    problem_template_path = os.path.join(this_directory, "yamls", JOB , WORK_LOAD + ".yaml")
    arch_path = os.path.join(this_directory, "yamls","arch_edge.yaml")
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
    #sparse_opt = yaml.load(open(sparse_opt_path), Loader = yaml.SafeLoader)
    
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
    #aggregated_input.update(sparse_opt)
    
    
    job_name  = "MCTS_searching"
    

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
    base_output_dir = os.path.join(this_directory, "MCTS_search_outputs")
    output_file_path = os.path.join(base_output_dir, job_name, 'outputs', "timeloop-model.map+stats.xml")
    if os.path.exists(output_file_path):
        job_output_stats = parse_timeloop_stats(output_file_path)
        os.remove(output_file_path)
        fitness = job_output_stats[stat_type]
    else:
        fitness = 10000000000000000000
    return fitness

#dimension = ['C','M','N','P','Q','R','S']

def target(state):   #这里的code都是Numpy的  map是长度为mapping_encoding_len的一维，perm长度为5          优化目标函数
    s = {}
    s['split'] = SPLIT
    s['permutations'] = [cantor_decode(state['perm_code'][i],7) for i in range(5) ]
    s['array'] = np.array(map_decode(state['map_code']))
    s['bypass_choice'] = state['bypass_choice']
    a = evaluate(s)
    if a != 10000000000000000000:
        print(a)
    return a

class ParameterOptimizationEnv(gym.Env):
    def __init__(self, mapping_encoding_len, target):
        super(ParameterOptimizationEnv, self).__init__()
        
        # 定义动作空间和状态空间
        #self.split_len = 2
        self.perm_code_len = 5
        self.bypass_choice_len = 2
        self.map_code_len = mapping_encoding_len

        #self.split_space = spaces.MultiDiscrete([8] * self.split_len)
        self.perm_code_space = spaces.MultiDiscrete([5040] * self.perm_code_len)  ####################################
        #self.bypass_choice_space = spaces.MultiDiscrete([8] * self.bypass_choice_len)
        self.map_code_space = spaces.MultiDiscrete([5] * self.map_code_len)

        self.action_space = spaces.MultiDiscrete([
            8,  # map_dim 的取值范围是 0-7，共 8 个取值
            5,  # perm_level 的取值范围是 0-4，共 5 个取值
            2,  # perm_add 的取值范围是 0-1，共 2 个取值
            2,  # map_add 的取值范围是 0-1，共 2 个取值
            2,  # bypass_level 的取值范围是 0-1，共 2 个取值
            2   # bypass_add 的取值范围是 0-1，共 2 个取值
        ])
        
        self.observation_space = spaces.Dict({
            #'split': self.split_space,
            'perm_code': self.perm_code_space,
            #'bypass_choice': self.bypass_choice_space,
            'map_code': self.map_code_space,
        })

        self.target = target
        self.state = {
            #'split': SPLIT,
            'perm_code':    PERM_CODE,
            #'bypass_choice': BYPASS,
            'map_code': MAP_CODE
        }

    def step(self, action):
        map_dim = action[0]  # 获取 map_dim 动作元素的值
        perm_level = action[1]  # 获取 perm_level 动作元素的值
        perm_add = action[2]  # 获取 perm_add 动作元素的值
        map_add = action[3]  # 获取 map_add 动作元素的值
        bypass_level = action[4]  # 获取 bypass_level 动作元素的值
        bypass_add = action[5]  # 获取 bypass_add 动作元素的值

        if perm_add == 1:  # 如果 perm_add 为 1，执行加法操作
            self.state['perm_code'][perm_level] = (self.state['perm_code'][perm_level] + 0.02 * 5040) % 5040
        else:  # 否则执行减法操作
            self.state['perm_code'][perm_level] = (self.state['perm_code'][perm_level] - 0.02 * 5040) % 5040

        if map_add == 1:  # 如果 map_add 为 1，执行加法操作
            self.state['map_code'][map_dim] = (self.state['map_code'][map_dim] + 1) % 5
        else:  # 否则执行减法操作
            self.state['map_code'][map_dim] = (self.state['map_code'][map_dim] - 1) % 5
        '''
        if bypass_add == 1:  # 如果 bypass_add 为 1  执行加法操作
            self.state['bypass_choice'][bypass_level] = (self.state['bypass_choice'][bypass_level] + 1) % 7
        else:  # 否则执行减法操作
            self.state['bypass_choice'][bypass_level] = (self.state['bypass_choice'][bypass_level] - 1) % 7
        '''
        reward = -self.target(self.state)
        done = True  # 在每个动作后结束

        return self.state, reward, done, {}
    

    def reset(self):
        self.state = {
            #'split': SPLIT,
            'perm_code':    PERM_CODE,
            'bypass_choice': BYPASS,
            'map_code': MAP_CODE
        }
        return self.state

    def render(self, mode='human'):
        print(self.state)

    def close(self):
        pass





def main():

    initial_indi = {
        "map_code": [0,1,4,1,4,1,1,2,4,2,4,2,4,2,1,2,1,1,1,2,2,2,3,0],  # 使用相同的初始值示例
        "perm_code":[619,2524,3052,3619,91] ,
        "bypass_choice" : [4,2]
    }

   # 创建环境
    env = ParameterOptimizationEnv(mapping_encoding_len,target)

    # 检查环境
    #check_env(env)

    # 定义并训练模型（使用Dueling DQN）
    model = PPO('MultiInputPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)

    # 保存模型
    model.save("PPO_parameter_optimization")

    # 加载模型
    model = PPO.load("PPO_parameter_optimization")

    # 使用模型进行预测
    obs = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


if __name__ == "__main__":
    main()
