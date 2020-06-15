
import configparser

from .config import DEFAULT_PARAMS_TYPES

def dict2str(d):
    ''' Convert all dictionary values to str '''
    for k, v in d.items():
        d[k] = str(v)
    return d

def str2bool(v):
  return str(v).lower() in ("yes", "true", "t", "1")

def parse_section(config_f, section):
    ''' Parse one section '''
    
    config = {}
    parser = configparser.ConfigParser()
    parser.read(config_f)
    for k, v in parser.items(section):
        # ignore unexpected key
        if k not in DEFAULT_PARAMS_TYPES[section].keys(): 
            print('do not recognize "%s" in section %s. skipping.....' % (k, section))
            continue 
        # convert to default type and add to dictionary
        if isinstance(DEFAULT_PARAMS_TYPES[section][k], tuple):
            # handle list
            v = v.strip('()[]{}').split(', ')
            config[k] = list(map(DEFAULT_PARAMS_TYPES[section][k][0], v))
        else:
            if DEFAULT_PARAMS_TYPES[section][k] == bool:
                config[k] = str2bool(v)
            else:
                config[k] = DEFAULT_PARAMS_TYPES[section][k](v)

    return config
    
def parse_config(config_f, sections):
    ''' Parse config file using configparser.ConfigParser '''

    config = {}
    if isinstance(sections, (list, tuple)):
        for section in sections:
            try:
                config[section] = parse_section(config_f, section)
            except configparser.NoSectionError:
                config[section] = {}
    else:
        raise TypeError('sections must be a list or tuple.')

    return config
