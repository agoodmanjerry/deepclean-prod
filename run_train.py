
import os
import argparse
import subprocess

from deepclean_prod import io

# Set default parameters 
TRAIN_PARAMS = ('chanslist', 'train_t0', 'fs', 'train_duration', 'train_frac', 
                'filt_fl', 'filt_fh', 'filt_order', 'train_kernel', 'train_stride', 
                'pad_mode', 'batch_size', 'max_epochs', 'num_workers', 'lr', 
                'weight_decay', 'fftlength', 'overlap', 'psd_weight', 'mse_weight', 
                'train_dir', 'save_dataset', 'load_dataset', 'log', 'device')

# Parse command line argument
def parse_cmd():
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__), usage='%(prog)s [options]')
    parser.add_argument('config', help='Path to config file', type=str) 
    params = parser.parse_args()
    return params

params = parse_cmd()
config = io.parse_config(params.config, 'config')

# Call training script
train_cmd = 'dc-prod-train ' + io.dict2args(config, TRAIN_PARAMS)
print('Run cmd: ' + train_cmd)
print('--------------------------')
subprocess.check_call(train_cmd.split(' '))
