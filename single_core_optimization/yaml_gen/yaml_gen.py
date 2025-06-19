from jinja2 import Template


#workload specs
dimension = ['C','M','N','P','Q','R','S']
dimension_size_dict = { 
    'C': 3,
    'M': 96,
    'N': 4,
    'P': 56,
    'Q': 56,
    'R': 11,
    'S': 11
}


#mapping specs
permutation_GB_T_order = [2,5,6,4,3,1,0]
permutation_GB_S_order = [2,5,6,4,3,1,0]
permutation_PEB_T_order = [2,5,6,4,3,1,0]

dimention_factors = [
    {
    'C': 3,
    'M': 1,
    'N': 4,
    'P': 4,
    'Q': 4,
    'R': 1,
    'S': 1
    },
    {
    'C': 1,
    'M': 1,
    'N': 1,
    'P': 7,
    'Q': 1,
    'R': 1,
    'S': 1
    },
    {
    'C': 1,
    'M': 0,
    'N': 1,
    'P': 2,
    'Q': 14,
    'R': 0,
    'S': 0
    },
]

permutation_GB_T_str = (dimension[permutation_GB_T_order[0]] + dimension[permutation_GB_T_order[1]] + dimension[permutation_GB_T_order[2]]
                  + dimension[permutation_GB_T_order[3]] + dimension[permutation_GB_T_order[4]] + dimension[permutation_GB_T_order[5]]
                  + dimension[permutation_GB_T_order[6]])

permutation_GB_S_str = (dimension[permutation_GB_S_order[0]] + dimension[permutation_GB_S_order[1]] + dimension[permutation_GB_S_order[2]]
                  + dimension[permutation_GB_S_order[3]] + dimension[permutation_GB_S_order[4]] + dimension[permutation_GB_S_order[5]]
                  + dimension[permutation_GB_S_order[6]])

permutation_PEB_T_str = (dimension[permutation_PEB_T_order[0]] + dimension[permutation_PEB_T_order[1]] + dimension[permutation_PEB_T_order[2]]
                  + dimension[permutation_PEB_T_order[3]] + dimension[permutation_PEB_T_order[4]] + dimension[permutation_PEB_T_order[5]]
                  + dimension[permutation_PEB_T_order[6]])


split_GB_to_PE = 4

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

#sparse_opt specs

compress_format_dict = ['B','CP','UOP','RLE']

sparse_object_GB = ['Inputs','Weights','Outputs']
sparse_object_mask_GB = [1,1,0]
dimension_of_tiles_GB = [10,9,0]
format_of_dimension_GB = [[2,2,2,2,2,2,2,2,2,2],[2,2,2,2,2,2,2,2,2],[]]

sparse_object_PEB = ['Inputs','Weights','Outputs']
sparse_object_mask_PEB = [1,1,0]
dimension_of_tiles_PEB = [8,7,0]
format_of_dimension_GB = [[2,2,2,2,2,2,2,2],[2,2,2,2,2,2,2],[]]

GateSkip_component_dict = ['GlobleBuffer','PE_buffer','MAC']
GateSkip_component_mask = [1,1,0]

# Gate/Skip A on B/C

GateSkip_choice = []

data = {
    'targets': [
        {
            'name': 'GlobelBuffer',
            'representation-format': {
                'data-spaces': [
                    {'name': 'Inputs', 'ranks': [{'format': 'UOP'}] * 10},
                    {'name': 'Weights', 'ranks': [{'format': 'UOP'}] * 8}
                ]
            }
        },
        {
            'name': 'PE_buffer',
            'representation-format': {
                'data-spaces': [
                    {'name': 'Inputs', 'ranks': [{'format': 'UOP'}] * 8},
                    {'name': 'Weights', 'ranks': [{'format': 'UOP'}] * 7}
                ]
            },
            'action-optimization': [
                {'type': 'gating', 'options': [{'target': 'Weights', 'condition-on': ['Inputs']}]}
            ]
        }
    ]
}

# ------------------------------------  sparse_opt  输入文件生成  -------------------------------------------
# 读取模板文件
with open('templates/sparse_opt_template.j2', 'r') as template_file_sparse_opt:
    template_content_sparse_opt = template_file_sparse_opt.read()

# 创建 Jinja2 模板对象
template_sparse_opt = Template(template_content_sparse_opt)

# 渲染模板并生成 YAML 内容
rendered_yaml_sparse_opt = template_sparse_opt.render(data)

# 将生成的 YAML 内容写入文件
with open('sparse_opt_output.yaml', 'w') as output_file_sparse_opt:
    output_file_sparse_opt.write(rendered_yaml_sparse_opt)

print("sparse_opt YAML file generated successfully.")


# ------------------------------------  mapping  输入文件生成  -------------------------------------------
# 读取模板文件
with open('templates/mapping_template.j2', 'r') as template_file_mapping:
    template_content_mapping = template_file_mapping.read()

# 创建 Jinja2 模板对象
template_mapping = Template(template_content_mapping)

# 渲染模板并生成 YAML 内容
rendered_yaml_mapping = template_mapping.render(mapping_variables)

# 将生成的 YAML 内容写入文件
with open('mapping_output.yaml', 'w') as output_file_mapping:
    output_file_mapping.write(rendered_yaml_mapping)

print("mapping YAML file generated successfully.")