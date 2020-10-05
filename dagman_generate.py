
import os
import sys
import argparse
import subprocess

import numpy as np

from pycondor import Job, Dagman

from deepclean_prod import io

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
prefix = config.get('prefix', 'prefix')
job_name = config.get('job_name', 'job')
max_clean_duration = int(config.get('max_clean_duration', 4096))
request_memory_training = config.get('request_memory_training', '2 GB')
request_memory_cleaning = config.get('request_memory_cleaning', '2 GB')

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
submit = os.path.join(out_dir, 'condor/submit')
log = os.path.join(out_dir, 'condor/log')
error = os.path.join(out_dir, 'condor/error')
output = os.path.join(out_dir, 'condor/output')
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
executable_summary = os.path.join(os.path.dirname(sys.executable), 'dc-prod-summary')

for seg in segment_data:
    
    # Get training/cleaning time
    train_t0, train_t1, clean_t0, clean_t1 = seg.astype(int)
    train_duration = train_t1 - train_t0
    clean_duration = clean_t1 - clean_t0
    
    # Get directory for segment        
    segment_subdir = os.path.join(out_dir, '{}-{:d}-{:d}'.format(
        prefix, clean_t0, clean_duration))

    # Training job
    train_config = io.get_keys(config, io.TRAIN_PARAMS)
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
        request_memory = request_memory_training,
        extra_lines=extra_lines,
        arguments=[train_args]
    )

    # Cleaning job
    clean_config = io.get_keys(config, io.CLEAN_PARAMS)
    clean_config['train_dir'] = segment_subdir
    clean_config['out_dir'] = segment_subdir
        
    t0 = np.arange(clean_t0, clean_t1, max_clean_duration)
    for i in range(len(t0)):
        clean_config['clean_t0'] = t0[i]
        if t0[i] + max_clean_duration > clean_t1:
            duration = clean_t1 - t0[i]
        else:
            duration = max_clean_duration
        clean_config['clean_duration'] = duration        
        clean_config['out_file'] = '{}-{:d}-{:d}.gwf'.format(prefix, t0[i], duration)

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
            request_memory = request_memory_cleaning,
            extra_lines=extra_lines,
            arguments=[clean_args]
        )
        job_train.add_child(job_clean)


    # Summary page job
#     summary_config = io.get_keys(config, io.SUMMARY_PARAMS)
#     summary_config['input_clean'] = os.path.join(segment_subdir, out_file)
#     summary_config['input_original'] = os.path.join(segment_subdir, 'original.h5')
#     summary_config['summary_dir'] = os.path.join(segment_subdir, 'html')
#     summary_config['loss_file'] = os.path.join(segment_subdir, 'metrics/loss.dat')
#     summary_args = io.dict2args(summary_config)

#     print(summary_args)
    
#     job_summary = Job(
#         name='summary_{}'.format(job_name),
#         executable=executable_summary,
#         submit=submit,
#         log=log, 
#         error=error,
#         output=output,
#         dag=dagman,
#         notification=notification,
#         getenv=getenv,
#         universe=universe,
#         request_memory='8 GB',
#         extra_lines=(
#             'accounting_group = {}'.format(accounting_group),
#             'stream_output = True',
#             'stream_error = True'),
#         arguments=[summary_args]
#     )
        
    # Inter-job training -> cleaning
#     job_train.add_child(job_clean)
#     job_clean.add_child(job_summary)

if params.submit:
    dagman.build_submit()
else:
    dagman.build()
    
