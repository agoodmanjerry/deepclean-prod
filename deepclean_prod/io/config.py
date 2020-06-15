

# default paramter types
DEFAULT_PARAMS_TYPES = {
    'config':{
        'outdir': str,
        'datadir': str,
        'save_dataset': bool,
        'chanslist': str,
        't0': int,
        'duration': int,
        'fs': int,
        'train_t0': int,
        'train_duration': int,
        'train_kernel': float,
        'train_stride': float,
        'clean_kernel': float,
        'clean_stride': float,
        'window': str,
        'pad_mode': str,
        'filt_fl': (float, ),
        'filt_fh': (float, ),
        'filt_order': int,
        'train_frac': float,
        'batch_size': int,
        'max_epochs': int,
        'num_workers': int,
        'lr': float,
        'weight_decay': float,
        'scheduler_step': int,
        'scheduler_gamma': float,
        'fft_length': float,
        'psd_weight': float,
        'mse_weight': float,
        'checkpoint': str,
        'outchannel': str,
    }
}
