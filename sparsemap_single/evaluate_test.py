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


##这个文件用来测试evaluate函数，输入为字典state,只描述mapping





OVERWRITE = 1

dimensions = {'C': 1, 'M': 2, 'R': 3, 'S': 4, 'N': 5, 'P': 6, 'Q': 7}

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
    p.communicate(timeout=None) 


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
        iwo_GB.append('Weight')
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
        iwo_PEB.append('Weight')
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
    
    this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
    this_directory = os.path.dirname(this_file_path)
    print("this_direc = ",this_directory)
    #os.chdir(this_directory)
    #print(os.getcwd())
    with open('yamls/mapping_conv.j2', 'r') as template_file_mapping:
        template_content_mapping = template_file_mapping.read()

    # 创建 Jinja2 模板对象
    template_mapping = Template(template_content_mapping)

    # 渲染模板并生成 YAML 内容
    rendered_yaml_mapping = template_mapping.render(mapping_variables)

    # 将生成的 YAML 内容写入文件
    with open('yamls/mapping_conv_output_MCTS.yaml', 'w') as output_file_mapping:
        output_file_mapping.write(rendered_yaml_mapping)

#----------------------------------------------调用sparse_loop-------------------------------------------------------

    
    
    problem_template_path = os.path.join(this_directory,"yamls", "workload_example.yaml")
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



root_state = {
    'permutations': [[6,4,0,1,5,2,3],[1,2,4,0,5,6,3],[1,6,2,3,0,5,4],[5,3,4,0,1,6,2],[0,5,6,3,1,2,4]],
    'bypass_choice': [5, 5],
    'array': [[1,6,1,1,3,1,2],[1,2,2,1,3,1,1],[1,8,2,3,1,1,2],[1,1,1,9,6,4,1],[3,1,1,2,1,3,3]],
    'split': [1,3]
}

print(root_state['split'])

print(evaluate(root_state))