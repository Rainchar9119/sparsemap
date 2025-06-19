
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


this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
this_directory = os.path.dirname(this_file_path)


sparse_opt_temp_path = os.path.join(this_directory, "yamls",  "sparse_opt_template.j2")
sparse_opt_output_path = os.path.join(this_directory, "yamls",  "sparse_opt_output.yaml")


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


        [
            {
                'name': 'GlobelBuffer',
                'type': 'skipping',
                'target': 'Inputs',
                'condition_on': '[ Weights ]'
            }
        ]

    ]

context = {'data': data}

with open(sparse_opt_temp_path,'r') as template_file_sparse_opt:
        template_content_sparse_opt = template_file_sparse_opt.read()
    
template_sparse_opt = Template(template_content_sparse_opt)

rendered_yaml_sparse_opt = template_sparse_opt.render(context)

# 将结果写入文件
with open(sparse_opt_output_path, 'w') as output:
    output.write(rendered_yaml_sparse_opt)