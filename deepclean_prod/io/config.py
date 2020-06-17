

# default paramter types
DEFAULT_PARAMS_TYPES = {
    'config':{
        # input/output config
        'out_dir': str,
        'train_dir': str,
        'out_file': str,
        'checkpoint': str,
        'out_channel': str,
        'ppr_file': str,
        'save_dataset': bool,
        'load_dataset': bool,

        # dataset config
        'chanslist': str,
        't0': int,
        'duration': int,
        'train_t0': int,
        'train_duration': int,
        'fs': int,
        'train_frac': float,
        
        # timeseries config
        'train_kernel': float,
        'train_stride': float,
        'clean_kernel': float,
        'clean_stride': float,
        'pad_mode': str,
        
        # pre/post-processing config
        'window': str,
        'filt_fl': (float, ),
        'filt_fh': (float, ),
        'filt_order': int,
        
        # training config
        'batch_size': int,
        'max_epochs': int,
        'num_workers': int,
        'lr': float,
        'weight_decay': float,
        
        # loss function config
        'fft_length': float,
        'psd_weight': float,
        'mse_weight': float,
    }
}
