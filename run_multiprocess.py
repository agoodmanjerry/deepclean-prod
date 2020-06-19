
import os
import argparse
import subprocess

from multiprocess import Process, Lock

import numpy as np

from deepclean_prod import io

# Set default parameters 
TRAIN_PARAMS = ('chanslist', 'train_t0', 'fs', 'train_duration', 'train_frac', 
                'filt_fl', 'filt_fh', 'filt_order', 'train_kernel', 'train_stride', 
                'pad_mode', 'batch_size', 'max_epochs', 'num_workers', 'lr', 
                'weight_decay', 'fftlength', 'overlap', 'psd_weight', 'mse_weight', 
                'train_dir', 'save_dataset', 'load_dataset', 'log', 'device')
CLEAN_PARAMS = ('chanslist', 'clean_t0', 'clean_duration', 'fs', 'clean_kernel', 
                'clean_stride', 'pad_mode', 'window', 'train_dir', 'checkpoint', 
                'ppr_file', 'out_dir', 'out_file', 'out_channel', 'save_dataset', 
                'load_dataset', 'log', 'device')

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
    params = parser.parse_args()
    return params

params = parse_cmd()
config = io.parse_config(params.config, 'config')
out_dir = config['out_dir']
prefix = config['prefix']
nproc = int(config.get('nproc', 1))

# Create output directory
os.makedirs(config['out_dir'], exist_ok=True)

# Call segment script
config['segment_file'] = os.path.join(out_dir, 'segment.txt')
segment_cmd = 'dc-prod-segment ' + io.dict2args(
    config, ('t0', 't1','ifo', 'segment_file'))
print('Run command: ' + segment_cmd)
print('--------------------------')
subprocess.check_call(segment_cmd.split(' '))
print('--------------------------')

# Read in segment file and call training/cleaning scripts
segment_data = np.genfromtxt(config['segment_file'])
segment_data = np.array_split(segment_data, nproc)


def process(lock, seg, i):

    lock.acquire()
    print(f'Process: {i}')
    lock.release()
    
    for s in seg:
        # Get training/cleaning time
        train_t0, train_t1, clean_t0, clean_t1 = s.astype(int)
        train_duration = train_t1 - train_t0
        clean_duration = clean_t1 - clean_t0

        # Get directory for segment
        segment_subdir = os.path.join(out_dir, '{}-{:d}-{:d}'.format(
            prefix, clean_t0, clean_duration))
        out_file = '{}-{:d}-{:d}.gwf'.format(prefix, clean_t0, clean_duration)
        log_file = 'log.log'
        
        # Get training command
        train_config = get_keys(config, TRAIN_PARAMS)
        train_config['train_dir'] = segment_subdir
        train_config['train_t0'] = train_t0
        train_config['train_duration'] = train_duration
        train_config['log'] = log_file
        train_cmd = 'dc-prod-train ' + io.dict2args(train_config)
                
        # Get cleaning command
        clean_config = get_keys(config, CLEAN_PARAMS)
        clean_config['train_dir'] = segment_subdir
        clean_config['out_dir'] = segment_subdir
        clean_config['out_file'] = out_file
        clean_config['clean_t0'] = clean_t0
        clean_config['clean_duration'] = clean_duration
        clean_config['log'] = log_file
        clean_cmd = 'dc-prod-clean ' + io.dict2args(clean_config)
        
        lock.acquire()
        print('Run cmd: ' + train_cmd)
        print('--------------------------')
        print('Run cmd: ' + clean_cmd)
        print('--------------------------')
        lock.release()

        subprocess.check_call(train_cmd.split(' '))
        subprocess.check_call(clean_cmd.split(' '))
        
if __name__ == '__main__':
    lock = Lock()
    for i, seg in enumerate(segment_data):
        p = Process(target=process, args=(lock, seg, i))
        p.start()
        