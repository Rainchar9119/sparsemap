import yaml, inspect, os, sys, subprocess, pprint, shutil, argparse
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from parse_timeloop_output import parse_timeloop_stats

OVERWRITE = True

PROB = "mm"

print("hello")
# paths to important input specs

this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))

this_directory = os.path.dirname(this_file_path)
print("this_directory = ",this_directory)
#problem_template_path = os.path.join(this_directory, "yamls", "workload_example.yaml")
#problem_template_path = os.path.join(this_directory, "yamls", "deepbrain_mm","workload_deepbrain_model3.yaml")
#problem_template_path = os.path.join(this_directory, "yamls", "resnet_conv","workload_resnet_conv7.yaml")
problem_template_path = os.path.join(this_directory, "yamls", "transformer_mm","workload_spGPT_MLP2.yaml")
#problem_template_path = os.path.join(this_directory, "yamls", "transformer","workload_transformer_mm1.yaml")
#problem_template_path = os.path.join(this_directory, "..", "..","evaluation_setups","fig1","workload", "spmspm.prob.0.4.yaml")
#problem_template_path = os.path.join(this_directory, "..", "..","evaluation_setups","fig13_dstc_setup","input_specs",  "prob.yaml")
#print("problem_template_path = ",problem_template_path)
#arch_path = os.path.join(this_directory, "..", "..","evaluation_setups","fig13_dstc_setup","input_specs",  "architecture.yaml")
#arch_path = os.path.join(this_directory, "yamls", "arch_cloud.yaml")
arch_path = os.path.join(this_directory, "yamls", "arch_mobile.yaml")
#arch_path = os.path.join(this_directory, "..", "..","evaluation_setups","fig1","arch",  "arch.yaml")
#component_path = os.path.join(this_directory, "..", "multiSCNN","fig13_dstc_setup","input_specs",  "compound_components.yaml")
mapping_path = os.path.join(this_directory, "outputs","example","outputs","timeloop-mapper.map.yaml")
#mapping_path = os.path.join(this_directory, "yamls", "timeloop-mapper-map-resconv4.yaml")
#mapping_path = os.path.join(this_directory, "yamls", "mapping_conv_output.yaml")
#mapping_path = "/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single/outputs/example/outputs/timeloop-mapper.map.yaml"
#mapping_path = os.path.join(this_directory, "..", "..","evaluation_setups","fig13_dstc_setup","input_specs",  "Os-mapping.yaml")
#mapping_path = os.path.join(this_directory, "..", "..","evaluation_setups","fig1","outputs", "output_0.4_bitmask", "timeloop-mapper.map.yaml")
#print("mapping_path = ",mapping_path)
#if os.path.exists(mapping_path):
#    print("路径存在")
mapper_path = os.path.join(this_directory, "yamls", "mapper.yaml")
#sparse_opt_path = os.path.join(this_directory, "yamls", "sparse_opt_output.yaml")
if PROB == "mm":
    sparse_opt_path = os.path.join(this_directory, "yamls", "sparse_opt_example_mm.yaml")
else:
    sparse_opt_path = os.path.join(this_directory, "yamls", "sparse_opt_example_conv.yaml")
#sparse_opt_path = os.path.join(this_directory, "..", "..","evaluation_setups","fig13_dstc_setup","input_specs",  "sparse-opt.yaml")
#sparse_opt_path = os.path.join(this_directory, "..", "..","evaluation_setups","fig1",  "sparse-opt","bitmask.skip.yaml")
#sparse_opt_path = os.path.join(this_directory, "..", "..","evaluation_setups","fig1",  "sparse-opt","coordinate_list.yaml")

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
    #subprocess_cmd = ["timeloop-model", input_file_path, ert_path, art_path]
    subprocess_cmd = ["timeloop-model", input_file_path, ert_path, art_path]
    print("\tRunning test: ", job_name)

    p = subprocess.Popen(subprocess_cmd)
    p.communicate(timeout=None) 


def main():
    
    ert_path = os.path.join(this_directory, "..","..", "evaluation_setups","fig13_dstc_setup","input_specs",  "ERT.yaml")
    art_path = os.path.join(this_directory, "..","..", "evaluation_setups","fig13_dstc_setup","input_specs",  "ART.yaml")
    #print(ert_path)
    
    # load all yaml input specs
    problem_template = yaml.load(open(problem_template_path), Loader = yaml.SafeLoader)
    arch = yaml.load(open(arch_path), Loader = yaml.SafeLoader)
    #components = yaml.load(open(component_path), Loader = yaml.SafeLoader)
    mapping = yaml.load(open(mapping_path), Loader = yaml.SafeLoader)
    mapper = yaml.load(open(mapper_path), Loader = yaml.SafeLoader)
    sparse_opt = yaml.load(open(sparse_opt_path), Loader = yaml.SafeLoader)
    
    output_base_dir = os.path.join(this_directory, "outputs")

    new_problem = deepcopy(problem_template)

    #new_problem["problem"]["instance"]["densities"]["A"]["density"] = 0.55
    #new_problem["problem"]["instance"]["densities"]["B"]["density"] = 0.35

    aggregated_input = {}
    aggregated_input.update(arch)
    aggregated_input.update(new_problem)
    #aggregated_input.update(components)
    if mapping is None:
        print("mapping is None")
    aggregated_input.update(mapping)
    #aggregated_input.update(mapper)
    aggregated_input.update(sparse_opt)
    
    job_name  = "example"
    
    run_timeloop(job_name, aggregated_input, os.path.join(output_base_dir, job_name), ert_path, art_path)



    stat_type = "cycles"
    #stat_type = "energy_pJ"
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

    base_output_dir = os.path.join(this_directory, "outputs")
    output_file_path = os.path.join(base_output_dir, job_name, 'outputs', "timeloop-model.map+stats.xml")
    if os.path.exists(output_file_path):
        job_output_stats = parse_timeloop_stats(output_file_path)
        #os.remove(output_file_path)
        #print(job_output_stats[stat_type])
        num = job_output_stats["cycles"]*job_output_stats["energy_pJ"]
        formatted_number = f"{num:.2e}"
        print(formatted_number)
    else:
        print("no_output!")


    
if __name__ == "__main__":
    main()