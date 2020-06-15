
import os
import glob
import pickle

def save_setting(fname, timeseries_params, criterion_params, network_params, ts):
    ''' Save setting '''
    aux_groups = list(ts.channels.keys())
    aux_groups.remove('hoft')
    with open(fname, 'wb') as f:
        pickle.dump(dict(
            norm=timeseries_params.get('norm'),
            filt=timeseries_params.get('filt'),
            filt_order=timeseries_params.get('filt_order'),
            filt_which=timeseries_params.get('filt_which'),
            preprocess_fl=timeseries_params.get('filt_fl'),
            preprocess_fh=timeseries_params.get('filt_fh'),
            postprocess_fl=criterion_params.get('fl'),
            postprocess_fh=criterion_params.get('fh'),
            scale=ts.scale_constants,
            shift=ts.shift_constants,
            aux_groups = aux_groups,
            network_params=network_params,
        ), f, protocol=-1)

def get_dataset_filename(indir_hoft, indir_witnesses):
    ''' Get all default dataset filename from input directory '''
    labels = ['training', 'target']
    dataset_fname = {}
    for label in labels:
        hoft_fname = glob.glob(os.path.join(
            indir_hoft, f'{label}_hoft.h5'))
        witnesses_fname = glob.glob(os.path.join(
            indir_witnesses, f'{label}_witnesses.h5'))
        if len(hoft_fname) > 0 and len(witnesses_fname) > 0:
            hoft_fname = hoft_fname[0]
            witnesses_fname = witnesses_fname[0]
#         if os.path.exists(hoft_fname) and os.path.exists(witnesses_fname):
            dataset_fname[label] = {'hoft': hoft_fname, 'witnesses': witnesses_fname}
    return dataset_fname

    
def get_default_checkpoint_file(indir):
    ''' Get the checkpoint file with the highest epoch number from directory '''
    default = glob.glob(os.path.join(indir, 'models/epoch_*'))
    default = sorted(default, key = lambda k: int(k.split('_')[-1]))[-1]
    return default
        
def replace_keys(org_dict, new_dict):
    ''' Replace key from a dictionary with value from a new dictionary
    If key not found, then create key '''
    for key, val in new_dict.items():
        old_val = org_dict.get(key)
        if val is not None:
            org_dict[key] = val
            if old_val is not None:
                print(f'- replace {key} key (value: {old_val}) in config file'\
                      f' with shell cmd value ({val})')
            else:
                print(f'- add {key} key (value: {val}) to config')
                