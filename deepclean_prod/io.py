
import logging
import configparser

logger = logging.getLogger(__name__)

ALL_PARAMS_KEYS = (
    'ifo', 'chanslist', 't0', 't1', 'clean_t0', 'clean_duration', 'train_t0', 
    'train_duration', 'fs', 'train_frac', 'filt_fl', 'filt_fh', 'filt_order', 
    'train_kernel', 'train_stride', 'clean_kernel', 'clean_stride', 'pad_mode', 
    'window', 'batch_size', 'max_epochs', 'num_workers', 'lr', 'weight_decay', 
    'fft_length', 'overlap', 'psd_weight', 'mse_weight', 'train_dir', 'checkpoint', 
    'ppr_file','out_dir', 'out_file', 'out_channel', 'prefix', 'save_dataset', 
    'load_dataset', 'nproc', 'log', 'job_name', 'accounting_group', 'notification',
    'universe', 'device'
)


def dict2str(d):
    ''' Convert all dictionary values to str '''
    for k, v in d.items():
        d[k] = str(v)
    return d


def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")


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
