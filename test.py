import yaml, inspect, os, sys, subprocess, pprint, shutil, argparse
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

#from parse_timeloop_output import parse_timeloop_stats

OVERWRITE = True

print("hello")
# paths to important input specs

this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
this_directory = os.path.dirname(this_file_path)
#print("this_directory = ",this_directory)
problem_template_path = os.path.join(this_directory,"sparsemap_single", "yamls", "workload_example.yaml")
#print("problem_template_path = ",problem_template_path)
#arch_path = os.path.join(this_directory,"sparsemap_single", "yamls", "DSTC-RF2x-24-bandwidth.yaml")
arch_path = os.path.join(this_directory,"sparsemap_single", "yamls", "arch_edge.yaml")
#component_path = os.path.join(this_directory, "..", "multiSCNN","fig13_dstc_setup","input_specs",  "compound_components.yaml")
mapping_path = os.path.join(this_directory,"sparsemap_single", "yamls","mapping_example.yaml")
mapper_path = os.path.join(this_directory, "sparsemap_single", "yamls","mapper.yaml")
sparse_opt_path = os.path.join(this_directory, "sparsemap_single", "yamls", "sparse_opt_example.yaml")
#sparse_opt_path = os.path.join(this_directory, "..", "multiSCNN","single_core_optimization",  "yaml_gen", "sparse_opt_output.yaml")

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
    subprocess_cmd = ["timeloop-mapper", input_file_path, ert_path, art_path]
    #subprocess_cmd = ["timeloop-model", input_file_path, ert_path, art_path]
    print("\tRunning test: ", job_name)

    p = subprocess.Popen(subprocess_cmd)
    p.communicate(timeout=None) 


def main():
    
    ert_path = os.path.join(this_directory, "..", "evaluation_setups","fig13_dstc_setup","input_specs",  "ERT.yaml")
    art_path = os.path.join(this_directory, "..", "evaluation_setups","fig13_dstc_setup","input_specs",  "ART.yaml")
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
    #aggregated_input.update(mapping)
    aggregated_input.update(mapper)
    #aggregated_input.update(sparse_opt)
    
    job_name  = "example"
    
    run_timeloop(job_name, aggregated_input, os.path.join(output_base_dir, job_name), ert_path, art_path)
    
    
if __name__ == "__main__":
    main()