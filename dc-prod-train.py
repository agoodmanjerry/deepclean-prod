#!/usr/bin/env python

import os
import pickle
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import deepclean_prod as dc
import deepclean_prod.config as config

# Default tensor type
torch.set_default_tensor_type(torch.FloatTensor)

# Use GPU if available
device = dc.nn.utils.get_device()


def parse_cmd():
    
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__), usage='%(prog)s [options]')
        
    # dataset arguments
    parser.add_argument('--train-t0', help='GPS of the first sample', type=int)
    parser.add_argument('--train-duration', help='Duration of train/val frame', type=int)
    parser.add_argument('--chanslist', help='Path to channel list', type=str)
    parser.add_argument('--fs', help='Sampling frequency', 
                        default=config.DEFAULT_SAMPLE_RATE, type=float)
    parser.add_argument('--train-frac', help='Training/validation partition', 
                        default=config.DEFAULT_TRAIN_FRAC, type=float)
    
    # preprocess arguments
    parser.add_argument('--filt-fl', help='Bandpass filter low frequency', 
                        default=config.DEFAULT_FLOW, nargs='+', type=float)
    parser.add_argument('--filt-fh', help='Bandpass filter high frequency', 
                        default=config.DEFAULT_FHIGH, nargs='+', type=float)
    parser.add_argument('--filt-order', help='Bandpass filter order', 
                        default=config.DEFAULT_FORDER, type=int)
    
    # timeseries arguments
    parser.add_argument('--train-kernel', help='Length of each segment in seconds', 
                        default=config.DEFAULT_TRAIN_KERNEL, type=float)
    parser.add_argument('--train-stride', help='Stride between segments in seconds', 
                        default=config.DEFAULT_TRAIN_STRIDE, type=float)
    parser.add_argument('--pad-mode', help='Padding mode', 
                        default=config.DEFAULT_PAD_MODE, type=str)
    
    # training arguments
    parser.add_argument('--batch-size', help='Batch size',
                        default=config.DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument('--max-epochs', help='Maximum number of epochs to train on',
                        default=config.DEFAULT_MAX_EPOCHS, type=int)
    parser.add_argument('--num-workers', help='Number of worker of DataLoader',
                        default=config.DEFAULT_NUM_WORKERS, type=int)
    parser.add_argument('--lr', help='Learning rate of ADAM optimizer', 
                        default=config.DEFAULT_LR, type=float)
    parser.add_argument('--weight-decay', help='Weight decay of ADAM optimizer',
                        default=config.DEFAULT_WEIGHT_DECAY, type=float)
    
    # loss function arguments
    parser.add_argument('--fftlength', help='FFT length of loss PSD',
                        default=config.DEFAULT_FFT_LENGTH, type=float)
    parser.add_argument('--overlap', help='Overlapping of loss PSD',
                        default=config.DEFAULT_OVERLAP, type=float)
    parser.add_argument('--psd-weight', help='PSD weight of composite loss',
                        default=config.DEFAULT_PSD_WEIGHT, type=float)
    parser.add_argument('--mse-weight', help='MSE weight of composite',
                        default=config.DEFAULT_MSE_WEIGHT, type=float)
    
    # input/output arguments
    parser.add_argument('--outdir', help='Path to output directory', type=str)
    parser.add_argument('--datadir', help='Path to data directory', type=str)
    parser.add_argument('--save-dataset', help='Save training dataset', 
                        default=False, type=bool)
    
    params = parser.parse_args()
    return params

params =  parse_cmd()

# Create output directory
print('Create output directory: {}'.format(params.outdir))
os.makedirs(params.outdir, exist_ok=True)

# Get data from NDS server or local
train_data = dc.timeseries.TimeSeriesSegmentDataset(
    params.train_kernel, params.train_stride, params.pad_mode)
val_data = dc.timeseries.TimeSeriesSegmentDataset(
    params.train_kernel, params.train_stride, params.pad_mode)

def fetch_data():
    # Partition into training and validation dataset
    train_t0 = params.train_t0
    train_duration = int(params.train_duration * params.train_frac)
    val_t0 = train_t0 + train_duration
    val_duration = params.train_duration - train_duration
    
    print('Fetching training data from {} .. {}'.format(train_t0, train_t0 + train_duration))
    train_data.fetch(params.chanslist, train_t0, train_duration, params.fs)

    print('Fetching validation data from {} .. {}'.format(val_t0, val_t0 + val_duration))
    val_data.fetch(params.chanslist, val_t0, val_duration, params.fs)

    # Save training/validation data if enabled
    if params.save_dataset:
        train_data.write(os.path.join(params.outdir, 'training.h5'))
        val_data.write(os.path.join(params.outdir, 'validation.h5'))

if params.datadir is not None:
    print('Data directory is given. Fetching from directory: {}'.format(params.datadir))

    train_datafile = os.path.join(params.datadir, 'training.h5')
    val_datafile = os.path.join(params.datadir, 'validation.h5')
    
    if os.path.exists(train_datafile) and os.path.exists(val_datafile):
        train_data.read(train_datafile, params.chanslist)
        val_data.read(val_datafile, params.chanslist)
    else:
        print('Either training or validation data file does not exist. '\
              'Downloading data instead.')
        fetch_data()
else:
    fetch_data()
    
# Preprocess data
print('Preprocessing')
# bandpass filter
train_data = train_data.bandpass(
    params.filt_fl, params.filt_fh, params.filt_order, 'target')
val_data = val_data.bandpass(
    params.filt_fl, params.filt_fh, params.filt_order, 'target')
# normalization
mean = train_data.mean
std = train_data.std
train_data = train_data.normalize()
val_data = val_data.normalize(mean, std)

# Save preprocessing setting
with open(os.path.join(params.outdir, 'ppr.bin'), 'wb') as f:
    pickle.dump({
        'mean': mean,
        'std': std,
    }, f, protocol=-1)

# Read dataset into DataLoader
train_loader = DataLoader(
    train_data, params.batch_size, num_workers=params.num_workers)
val_loader = DataLoader(
    val_data, params.batch_size, num_workers=params.num_workers)

# Creating model, loss function, optimizer and lr scheduler
print('Creating neural network, loss function, optimizer, etc.')
model = dc.nn.net.DeepClean(train_data.n_channels - 1)
print(model)
model = model.to(device)  # transfer model to GPU if available

criterion = dc.criterion.CompositePSDLoss(
    params.fs, params.filt_fl, params.filt_fh,
    fftlength=params.fftlength, overlap=params.overlap, 
    psd_weight=params.psd_weight, mse_weight=params.mse_weight, 
    reduction='sum', device=device)

optimizer = optim.Adam(
    model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.1)

# Start training
logger = dc.logger.Logger(outdir=params.outdir, metrics=['loss'])
dc.nn.utils.train(
    train_loader, model, criterion, optimizer, lr_scheduler,
    val_loader=val_loader, max_epochs=params.max_epochs, logger=logger, 
    device=device)
