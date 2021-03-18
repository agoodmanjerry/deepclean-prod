
import logging
import configparser

logger = logging.getLogger(__name__)

# Parameters of scripts 
TRAIN_PARAMS = ('chanslist', 'train_t0', 'fs', 'train_duration', 'train_frac', 
                'filt_fl', 'filt_fh', 'filt_order', 'train_kernel', 'train_stride', 
                'pad_mode', 'batch_size', 'max_epochs', 'num_workers', 'lr', 
                'weight_decay', 'fftlength', 'overlap', 'psd_weight', 'mse_weight', 
                'train_dir', 'save_dataset', 'load_dataset', 'log', 'device')
CLEAN_PARAMS = ('chanslist', 'clean_t0', 'clean_duration', 'fs', 'clean_kernel', 'clean_stride', 
                'pad_mode', 'window', 'train_dir', 'checkpoint', 'ppr_file', 
                'out_dir', 'out_file', 'out_channel', 'save_dataset', 'load_dataset', 'log', 
                'device')
SUMMARY_PARAMS = ('chanslist', 'out_channel', 'plot_fl', 'plot_fh', 'fftlength_spec', 
                  'fftlength_asd', 'overlap_spec', 'overlap_asd', 'asd_min', 'asd_max',
                  'asd_whiten_min', 'asd_whiten_max')

CONDOR_PARAMS = ('job_name', 'universe', 'request_memory_training', 'request_memory_cleaning', 
                 'accounting_group', 'prefix', 't0', 't1', 'ifo', 'max_clean_duration')

ALL_PARAMS_KEYS = set().union(TRAIN_PARAMS, CLEAN_PARAMS, SUMMARY_PARAMS, CONDOR_PARAMS)


def dict2str(d):
    ''' Convert all dictionary values to str '''
    for k, v in d.items():
        d[k] = str(v)
    return d


def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")


def get_keys(data, keys):
    ''' Get all keys from a dictionayr and return a dictionary '''
    new = {}
    for k, v in data.items():
        if k in keys:
            new[k] = v
    return new


def dict2args(params, keys=None):
    """ Convert dictionary to commandline argument string """

    # If no key is given, take all keys
    if keys is None:
        keys  = params.keys()
    
    # Parse 
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
    append = append[:-1]  # remove the trailing white space
    return append


def parse_config(config_fname, section='config'):
    """ Parse a section of a config file into a dictionary """    
    config = {}
    parser = configparser.ConfigParser()
    parser.read(config_fname)
    for key, val in parser.items(section):
        # ignore unexpected key
        if key not in ALL_PARAMS_KEYS: 
            logger.warning('WARNING: Do not recognize key "%s".' % key)
            continue 
        val = val.split(', ')
        if len(val) == 1:
            val = val[0]
        config[key] = val
    return config
