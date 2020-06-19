
import os
import sys
import argparse
import subprocess

import numpy as np

from pycondor import Job, Dagman

from deepclean_prod import io

# Set default parameters 
TRAIN_PARAMS = ('chanslist', 'train_t0', 'fs', 'train_duration', 'train_frac', 
                'filt_fl', 'filt_fh', 'filt_order', 'train_kernel', 'train_stride', 
                'pad_mode', 'batch_size', 'max_epochs', 'num_workers', 'lr', 
                'weight_decay', 'fftlength', 'overlap', 'psd_weight', 'mse_weight', 
                'train_dir', 'save_dataset', 'load_dataset', 'device')
CLEAN_PARAMS = ('chanslist', 'clean_t0', 'clean_duration', 'fs', 'clean_kernel', 
                'clean_stride', 'pad_mode', 'window', 'train_dir', 'checkpoint', 
                'ppr_file', 'out_dir', 'out_file', 'out_channel', 'save_dataset', 
                'load_dataset', 'device')

def get_keys(data, keys):
    new = {}
    for k, v in data.items():
        if k in keys:
            new[k] = v
    return new

# Parse command line argument
def parse_cmd():
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__), usage='%(prog)s [options]')
    parser.add_argument('config', help='Path to config file', type=str) 
    parser.add_argument('--submit', help='Submit DAGMAN', action='store_true')
    params = parser.parse_args()
    return params

params = parse_cmd()

# Parse config file
config = io.parse_config(params.config, 'config')
out_dir = config['out_dir']
prefix = config['prefix']
job_name = config['job_name']

# Create output directory
os.makedirs(out_dir, exist_ok=True)

# Run segment script
config['segment_file'] = os.path.join(out_dir, 'segment.txt')
segment_cmd = 'dc-prod-segment ' + io.dict2args(config, ('t0', 't1', 'ifo', 'segment_file'))
print('Get segment data')
subprocess.check_call(segment_cmd.split(' '))

# Read in segment data
segment_data = np.genfromtxt(config['segment_file'])

# For each segment, create a condor job for training and cleaning
# default condor
submit = 'condor/submit'
log = 'condor/log'
error = 'condor/error'
output = 'condor/output'
getenv = True
requirements = '(CUDACapability > 3.5)'
universe = config.get('universe', 'vanilla')
accounting_group = config['accounting_group']
notification = config.get('notification')
extra_lines = (
    'accounting_group = {}'.format(accounting_group),
    'request_gpus = 1',
    'stream_output = True',
    'stream_error = True',
)

# create DAGMAN 
dagman = Dagman(name='dag_{}'.format(job_name), submit=submit)
executable_train = os.path.join(os.path.dirname(sys.executable), 'dc-prod-train')
executable_clean = os.path.join(os.path.dirname(sys.executable), 'dc-prod-clean')

for seg in segment_data:
    
    # Get training/cleaning time
    train_t0, train_t1, clean_t0, clean_t1 = seg.astype(int)
    train_duration = train_t1 - train_t0
    clean_duration = clean_t1 - clean_t0
    
    # Get directory for segment        
    segment_subdir = os.path.join(out_dir, '{}-{:d}-{:d}'.format(
        prefix, clean_t0, clean_duration))
    out_file = '{}-{:d}-{:d}.gwf'.format(prefix, clean_t0, clean_duration)
        
    # Training job
    train_config = get_keys(config, TRAIN_PARAMS)
    train_config['train_dir'] = segment_subdir
    train_config['train_t0'] = train_t0
    train_config['train_duration'] = train_duration
    train_args = io.dict2args(train_config)
    
    job_train = Job(
        name='train_{}'.format(job_name),
        executable=executable_train,
        submit=submit,
        log=log,
        error=error,
        output=output,
        dag=dagman,
        requirements=requirements,
        notification=notification,
        getenv=getenv,
        universe=universe,
        request_memory = '3 GB',
        extra_lines=extra_lines,
        arguments=[train_args]
    )

    # Cleaning job
    clean_config = get_keys(config, CLEAN_PARAMS)
    clean_config['train_dir'] = segment_subdir
    clean_config['out_dir'] = segment_subdir
    clean_config['out_file'] = out_file
    clean_config['clean_t0'] = clean_t0
    clean_config['clean_duration'] = clean_duration
    clean_args = io.dict2args(clean_config)
        
    job_clean = Job(
        name='clean_{}'.format(job_name),
        executable=executable_clean,
        submit=submit,
        log=log, 
        error=error,
        output=output,
        dag=dagman,
        requirements=requirements,
        notification=notification,
        getenv=getenv,
        universe=universe,
        request_memory = '8 GB',
        extra_lines=extra_lines,
        arguments=[clean_args]
    )
    
    # Inter-job training -> cleaning
    job_train.add_child(job_clean)

if params.submit:
    dagman.build_submit()
else:
    dagman.build()
    
