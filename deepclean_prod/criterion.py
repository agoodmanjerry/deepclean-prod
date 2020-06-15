

import numpy as np

import torch
import torch.nn as nn


def _torch_welch(data, fs=1.0, nperseg=256, noverlap=None, average='mean', 
                 device='cpu'):
    """ Compute PSD using Welch's method. 
    NOTE: The function is off by a constant factor from scipy.signal.welch 
    Because we will be taking the ratio, this is not important (for now) 
    """
    if len(data.shape) > 2:
        data = data.view(data.shape[0], -1)
    N, nsample = data.shape
    
    # Get parameters
    if noverlap is None:
        noverlap = nperseg//2
    nstride = nperseg - noverlap
    nseg = int(np.ceil((nsample-nperseg)/nstride)) + 1
    nfreq = nperseg // 2 + 1
    T = nsample*fs
   
    # Calculate the PSD
    psd = torch.zeros((nseg, N, nfreq)).to(device)
    window =  torch.hann_window(nperseg).to(device)*2
    
    # calculate the FFT amplitude of each segment
    for i in range(nseg):
        seg_ts = data[:, i*nstride:i*nstride+nperseg]*window
        seg_fd = torch.rfft(seg_ts, 1)
        seg_fd_abs = (seg_fd[:, :, 0]**2 + seg_fd[:, :, 1]**2)
        psd[i] = seg_fd_abs
    
    # taking the average
    if average == 'mean':
        psd = torch.sum(psd, 0)
    elif average == 'median':
        psd = torch.median(psd, 0)[0]*nseg
    else:
        raise ValueError(f'average must be "mean" or "median", got {average} instead')

    # Normalize
    psd /= T
    return psd


class MSELoss(nn.Module):
    """ Mean-squared error loss """
    
    def __init__(self, reduction='mean', eps=1e-8):
        super().__init__()
        
        if reduction not in ('mean', 'sum'):
            raise ValueError(
                '`reduction` not recognized. must be "mean" or "sum"')
        self.reduction = reduction
        self.eps = eps
        
    def forward(self, pred, target):
        loss = (target - pred) ** 2
        loss = torch.mean(loss, 1)
        
        # Averaging over patch
        if self.reduction == 'mean':
            loss = torch.sum(loss) / len(pred)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
            
        return loss

    
class PSDLoss(nn.Module):
    ''' Compute the power spectrum density (PSD) loss, defined 
    as the average over frequency of the PSD ratio '''
    
    
    def __init__(self, fs=1.0, fl=20., fh=500., fftlength=1., overlap=None, 
                 asd=False, average='mean', reduction='mean', device='cpu'):
        super().__init__()
        
        if isinstance(fl, (int, float)):
            fl = (fl, )
        if isinstance(fh, (int, float)):
            fh = (fh, )
        
        # Initialize attributes
        if reduction not in ('mean', 'sum'):
            raise ValueError(
                '`reduction` not recognized. must be "mean" or "sum"')
        self.reduction = reduction
        self.fs = fs
        self.average = average
        self.device = device
        self.asd = asd
        
        nperseg = int(fftlength * self.fs)
        if overlap is not None:
            noverlap = int(overlap * self.fs)
        else:
            noverlap = None
        self.welch = lambda x: _torch_welch(
            x, fs=fs, nperseg=nperseg, noverlap=noverlap, device=device)
        
        # Get scaling and masking
        freq = torch.linspace(0., fs/2., nperseg//2 + 1)
        self.dfreq = freq[1] - freq[0]
        self.mask = torch.zeros(nperseg//2 +1).type(torch.ByteTensor)
        self.scale = 0.
        for l, h in zip(fl, fh):
            self.mask = self.mask | (l < freq) & (freq < h)
            self.scale += (h - l)
        self.mask = self.mask.to(device)
    
    def forward(self, pred, target):
        
        # Calculate the PSD of the residual and the target
        psd_res = self.welch(target - pred)
        psd_target = self.welch(target)
        psd_res[:, ~self.mask] = 0.

        # psd loss is the integration over all frequencies
        psd_ratio = psd_res/psd_target
        asd_ratio = torch.sqrt(psd_ratio)
            
        if self.asd:
            loss = torch.sum(asd_ratio, 1)*self.dfreq/self.scale
        else:
            loss = torch.sum(psd_ratio, 1)*self.dfreq/self.scale
        
        # Averaging over batch
        if self.reduction == 'mean':
            loss = torch.sum(loss)/len(psd_res)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        
        return loss     
    
    
class CompositePSDLoss(nn.Module):
    ''' PSD + MSE Loss with weight '''
    
    def __init__(self, fs=1.0, fl=20., fh=500., fftlength=1., overlap=None, 
                 asd=False, average='mean', reduction='mean', psd_weight=0.5, 
                 mse_weight=0.5, device='cpu'):
        super().__init__()
        if reduction not in ('mean', 'sum'):
            raise ValueError(
                '`reduction` not recognized. must be "mean" or "sum"')
        self.reduction = reduction
        
        self.psd_loss = PSDLoss(
            fs=fs, fl=fl, fh=fh, fftlength=fftlength, overlap=overlap, asd=asd, 
            average=average, reduction=reduction, device=device)
        self.mse_loss = MSELoss(reduction=reduction)
        
        self.psd_weight = psd_weight
        self.mse_weight = mse_weight
                
    def forward(self, pred, target):
        # if weight is 0: only run 1 to save computational time
        if self.psd_weight == 0:
            return self.mse_loss(pred, target)
        if self.mse_weight == 0:
            return self.psd_loss(pred, target)
        
        psd_loss = self.psd_weight * self.psd_loss(pred, target)
        mse_loss = self.mse_weight * self.mse_loss(pred, target)
        
        return (psd_loss + mse_loss)