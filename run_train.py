
import os
import argparse
import subprocess

import deepclean_prod as dc

# Set default parameters 
TRAIN_PARAMS = ('chanslist', 'train_t0', 'fs', 'train_duration', 'train_frac', 
                'filt_fl', 'filt_fh', 'filt_order', 'train_kernel', 'train_stride', 
                'pad_mode', 'batch_size', 'max_epochs', 'num_workers', 'lr', 
                'weight_decay', 'fftlength', 'overlap', 'psd_weight', 'mse_weight', 
                'outdir', 'datadir', 'save_dataset')

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
train_cmd = './dc-prod-train.py '
train_append = create_append(config, TRAIN_PARAMS)
train_cmd += train_append
print('Run cmd: ' + train_cmd)
print('--------------------------')
subprocess.check_call(train_cmd.split(' '))
