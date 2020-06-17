
import os
import argparse
import subprocess

import deepclean_prod as dc

# Set default parameters 
CLEAN_PARAMS = ('chanslist', 't0', 'fs', 'duration', 'clean_kernel', 'clean_stride', 
                'pad_mode', 'window', 'train_dir', 'checkpoint', 'ppr_file', 
                'out_dir', 'out_file', 'out_channel', 'save_dataset', 'load_dataset')

def create_append(params, keys=None):
    # if no key is given, take all
    if keys is None:
        keys  = params.keys()
    
    # start parsing
    append = ''
    for key, val in params.items():
        if key not in keys:
            continue
        key = key.replace('_', '-')
        append += f'--{key} '
        if isinstance(val, (list, tuple)):
            for v in val:
                append += str(v)
                append += ' '
        else:
            append += str(val)
            append += ' '
    append = append[:-1]  # exclude the trailing white space
    return append

# Parse command line argument
def parse_cmd():
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__), usage='%(prog)s [options]')
    parser.add_argument('config', help='Path to config file', type=str) 
    params = parser.parse_args()
    return params

params = parse_cmd()
config = dc.io.parser.parse_section(params.config, 'config')

# Call training script
clean_cmd = 'dc-prod-clean '
clean_append = create_append(config, CLEAN_PARAMS)
clean_cmd += clean_append
print('Run cmd: ' + clean_cmd)
print('--------------------------')
subprocess.check_call(clean_cmd.split(' '))
