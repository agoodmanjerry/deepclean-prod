
import os
import argparse
import subprocess

from deepclean_prod import io

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
train_cmd = 'dc-prod-train ' + io.dict2args(config, io.TRAIN_PARAMS)
print('Run cmd: ' + train_cmd)
print('--------------------------')
subprocess.check_call(train_cmd.split(' '))
