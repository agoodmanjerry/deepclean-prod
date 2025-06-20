import numpy as np

import torch
import torch.nn as nn


def _torch_welch(data, fs=1.0, nperseg=256, noverlap=None, average='mean', device='cpu'):
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


def _torch_cross_welch(x, y, fs=1.0, nperseg=256, noverlap=None, average='mean', device='cpu'):
    """
    Compute the cross power spectral density (CSD) between two batched signals using Welch's method.

    Args:
        x (torch.Tensor): First input signal, shape (B, T).
        y (torch.Tensor): Second input signal, shape (B, T).
        fs (float): Sampling frequency.
        nperseg (int): Length of each segment.
        noverlap (int or None): Number of points to overlap between segments.
        average (str): Averaging method across segments ('mean' or 'median').
        device (str): Device on which to perform computation.

    Returns: 
        tuple: (real_part, imag_part) of the cross-spectrum Sxy(f), each with shape (B, F).

    Notes:
        B = batch size (number of samples in the batch)
        C = number of witness channels
        T = time series length
    """

    # Reshape inputs to 2D if they have extra dimensions (e.g., B x 1 x T => B x T)
    if len(x.shape) > 2:
        x = x.view(x.shape[0], -1)
    if len(y.shape) > 2:
        y = y.view(y.shape[0], -1)

    # Move tensors to the specified device (CPU or GPU)
    x = x.to(device)
    y = y.to(device)

    # Get batch size and number of time samples
    N, nsample = x.shape

    # Set default overlap if not specified
    if noverlap is None:
        noverlap = nperseg // 2

    # Compute stride and number of segments
    nstride = nperseg - noverlap
    nseg = int(np.ceil((nsample - nperseg) / nstride)) + 1
    T = nsample * fs # Total time span (not segment length)

    # Preallocate tensor for cross spectra (real and imaginary parts)
    # Shape: [segments, batch, freqs, 2]
    Sxy = torch.zeros((nseg, N, nperseg // 2 + 1, 2)).to(device)

    # Define Hann window (scaled by 2 for normalization)
    window = torch.hann_window(nperseg).to(device) * 2

    # Loop over each segment to compute segment-wise FFTs and cross terms
    for i in range(nseg):
        # Extract and window each segment
        seg_x = x[:, i*nstride:i*nstride+nperseg] * window
        seg_y = y[:, i*nstride:i*nstride+nperseg] * window

        # Compute real FFT (returns complex in 2 parts)
        X = torch.rfft(seg_x, 1)
        Y = torch.rfft(seg_y, 1)

        # Compute real and imaginary parts of cross spectrum: X * conj(Y)
        real = X[..., 0]*Y[..., 0] + X[..., 1]*Y[..., 1]
        imag = X[..., 1]*Y[..., 0] - X[..., 0]*Y[..., 1]

        # Store into cross-spectral tensor
        Sxy[i, ..., 0] = real
        Sxy[i, ..., 1] = imag

    # Average across segments
    if average == 'mean':
        Sxy = torch.sum(Sxy, dim=0)
    elif average == 'median':
        # Compute median along segment dimension (and scale to match mean)
        Sxy = torch.stack([
            torch.median(Sxy[..., 0], dim=0)[0] * nseg,
            torch.median(Sxy[..., 1], dim=0)[0] * nseg
        ], dim=-1)
    else:
        raise ValueError(f'average must be "mean" or "median", got {average} instead')

    # Normalize by total time
    Sxy /= T

    # Return the real and imaginary parts separately
    return Sxy[..., 0], Sxy[..., 1]


def compute_coherence(x, y, fs, nperseg, noverlap=None, device='cpu', average='mean'):
    """
    Compute magnitude-squared coherence between two signals.

    Args:
        x (torch.Tensor): First signal, shape (B, T).
        y (torch.Tensor): Second signal, shape (B, T).
        fs (float): Sampling frequency.
        nperseg (int): Segment length.
        noverlap (int or None): Overlap length between segments.
        device (str): Device for computation.
        average (str): Averaging mode ('mean' or 'median').

    Returns (torch.Tensor): Coherence values in frequency domain, shape (B, F).
    """

    x = x.to(device)
    y = y.to(device)

    Sxx = _torch_welch(x, fs, nperseg, noverlap, average, device)
    Syy = _torch_welch(y, fs, nperseg, noverlap, average, device)
    Sxy_real, Sxy_imag = _torch_cross_welch(x, y, fs, nperseg, noverlap, average, device)

    Sxy_mag2 = Sxy_real**2 + Sxy_imag**2
    return Sxy_mag2 / (Sxx * Syy + 1e-12)


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
    
    
    def __init__(self, fs=1.0, fl=20., fh=500., fftlength=1., overlap=None, asd=False, average='mean', reduction='mean', device='cpu'):
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
        # print(f'0 in target?: {torch.isinf(1/target).reshape(-1).sum()}')
        # print(f'0 in psd_target?: {torch.isinf(1/psd_target).reshape(-1).sum()}')
        psd_res[:, ~self.mask] = 0.

        # psd loss is the integration over all frequencies
        # psd_ratio = psd_res/psd_target
        psd_ratio = psd_res/(psd_target + 1e-13)
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
    

class CoherenceLoss(nn.Module):
    """
    Loss function based on coherence between the residual (target - pred)
    and auxiliary (witness or PEM) channels. The goal is to minimize residual coherence
    in specified frequency bands while preserving target coherence patterns.
    """

    def __init__(self, fs, fl, fh, fftlength=1., overlap=None, reduction='mean', device='cpu', average='mean'):
        """
        Initialize parameters for Welch's method and coherence calculation.

        Args:
            fs (float): Sampling frequency.
            fl, fh (float or list): Frequency bands of interest (lower and upper bounds).
            fftlength (float): Segment length in seconds.
            overlap (float or None): Overlap in seconds between segments.
            reduction (str): 'mean' or 'sum' over batch elements.
            device (str): 'cpu' or 'cuda'.
            average (str): 'mean' or 'median' segment averaging.
        """

        super().__init__()
        self.fs = fs
        self.nperseg = int(fftlength * fs) # Segment length in samples
        self.noverlap = int(overlap * fs) if overlap else None
        self.reduction = reduction
        self.device = device
        self.average = average

        # Define frequency bins: from 0 to Nyquist
        freq = torch.linspace(0., fs/2., self.nperseg//2 + 1).to(device)

        # Create a frequency mask to select only frequencies within fl–fh
        self.freq_mask = torch.zeros_like(freq).type(torch.uint8)
        self.dfreq = freq[1] - freq[0]  # Frequency resolution
        self.scale = 0.                 # Used to normalize loss value 

        # Handle single or multiple bands (fl, fh)
        for l, h in zip(fl if isinstance(fl, (list, tuple)) else [fl], fh if isinstance(fh, (list, tuple)) else [fh]):
            self.freq_mask |= (freq >= l) & (freq <= h)
            self.scale += (h - l)   # Total bandwidth

    def forward(self, pred, target, witness):
        """
        Compute the coherence loss.

        Args:
            pred (torch.Tensor): Model prediction, shape (B, T)
            target (torch.Tensor): Ground truth signal, shape (B, T)
            witness (torch.Tensor): Auxiliary input channels, shape (B, C, T)

        Returns:
            loss (torch.Tensor): Scalar loss value (mean or sum over batch)
        """

        residual = target - pred        # Residual: what was not removed
        B, C, T = witness.shape         # Batch size, # witness channels, time length

        ratio_list = []                 # To store coherence ratios per witness

        for i in range(C):
            # Extract the entire time series for the i-th witness channel from all batch elements.
            # The resulting tensor w has shape: (B, T)
            # 1st: means Select all batches (dimension 0)
            # 2nd: means Select the i-th witness channel (dimension 1)
            # 3rd: means Select all time steps (dimension 2)
            # If witness is shaped (4, 3, 1024) i.e., 4 batches, 3 witness channels, each with 1024 time steps
            # 'w' will return a tensor of shape (4, 1024) containing the second witness channel across all batches.
            w = witness[:, i, :] 

            # Compute coherence between residual and witness
            coh_res = compute_coherence(residual, w, self.fs, self.nperseg, self.noverlap, self.device, self.average)

            # Compute coherence between original target and witness
            coh_tgt = compute_coherence(target, w, self.fs, self.nperseg, self.noverlap, self.device, self.average)

            # Zero out values outside the frequency band of interest
            coh_res[:, ~self.freq_mask] = 0.
            coh_tgt[:, ~self.freq_mask] = 0.

            # Ratio of residual coherence to original target coherence
            ratio = coh_res / (coh_tgt + 1e-12)          # avoid divide-by-zero
            ratio_list.append(torch.mean(ratio, dim=1))  # Mean across frequencies

        # Stack all witness channel losses → (B, C) → average over C
        loss = torch.stack(ratio_list, dim=1).mean(dim=1)

        # Return final loss depending on reduction method
        return loss.mean() if self.reduction == 'mean' else loss.sum()


class TransferFunctionLoss(nn.Module):
    """
    Transfer Function Loss inspired by ICA-style transfer function modeling.
    
    This loss penalizes the correlation between the residual signal (target - prediction)
    and the auxiliary (witness or PEM) channels in the frequency domain. The objective is to 
    minimize the learned transfer functions from auxiliary channels to the residual,
    thereby encouraging the model to subtract correlated noise more effectively.

    Args:
        fs (float): Sampling frequency in Hz.
        fl (float or list): Lower frequency bound(s) for evaluation band(s).
        fh (float or list): Upper frequency bound(s) for evaluation band(s).
        fftlength (float): FFT segment length in seconds (controls spectral resolution).
        overlap (float or None): Overlap duration in seconds between FFT segments.
        reduction (str): 'mean' or 'sum' to specify how to reduce batch losses.
        device (str): Device on which to compute ('cpu' or 'cuda').
        average (str): 'mean' or 'median' to specify how to average across FFT segments.
        nonlinearity (str): Transformation applied to TF magnitude squared.
            Options:
                - 'log1p': log(1 + x), default and recommended
                - 'sqrt': sqrt(x)
                - 'none': no transformation
    """

    def __init__(self, fs, fl, fh, fftlength=1.0, overlap=None,
                 reduction='mean', device='cpu', average='mean',
                 nonlinearity='log1p'):
        super().__init__()

        self.fs = fs
        self.device = device
        self.average = average
        self.reduction = reduction
        self.nonlinearity = nonlinearity

        self.nperseg = int(fftlength * fs)
        self.noverlap = int(overlap * fs) if overlap else None

        # Frequency axis and resolution
        freqs = torch.linspace(0., fs/2., self.nperseg // 2 + 1).to(device)
        self.freqs = freqs
        self.dfreq = freqs[1] - freqs[0]

        # Create frequency mask and weight vector
        self.freq_mask = torch.zeros_like(freqs).type(torch.uint8)
        self.freq_weights = torch.ones_like(freqs)

        if isinstance(fl, (float, int)):
            fl = [fl]
        if isinstance(fh, (float, int)):
            fh = [fh]

        self.scale = 0.0
        for l, h in zip(fl, fh):
            mask = (freqs >= l) & (freqs <= h)
            self.freq_mask |= mask
            self.scale += (h - l)

            # Test: Emphasize center frequency using a Gaussian curve
            center = (l + h) / 2
            width = (h - l)
            weight = torch.exp(-0.5 * ((freqs - center) / (width / 4))**2)
            self.freq_weights = torch.max(self.freq_weights, weight)

        # Normalize weights so they sum to 1 (approx)
        self.freq_weights = self.freq_weights / (torch.sum(self.freq_weights * self.dfreq))

    def _torch_welch(self, x):
        """Compute Welch PSD for input signal"""
        return _torch_welch(x, fs=self.fs, nperseg=self.nperseg,
                            noverlap=self.noverlap, average=self.average,
                            device=self.device)

    def _torch_cross_welch(self, x, y):
        """Compute Welch cross-spectral density between x and y"""
        return _torch_cross_welch(x, y, fs=self.fs, nperseg=self.nperseg,
                                  noverlap=self.noverlap, average=self.average,
                                  device=self.device)

    def apply_nonlinearity(self, x):
        """Apply nonlinearity transformation to enhance numerical stability or learning"""
        if self.nonlinearity == 'log1p':
            return torch.log1p(x)
        elif self.nonlinearity == 'sqrt':
            return torch.sqrt(x + 1e-12)
        else:
            return x

    def forward(self, pred, target, witness):
        """
        Compute the Transfer Function Loss.

        Args:
            pred (Tensor): Model output of shape (B, T)
            target (Tensor): Ground truth strain of shape (B, T)
            witness (Tensor): Auxiliary channels of shape (B, C, T)

        Returns:
            loss (Tensor): Scalar loss value (mean or sum over batch)
        """
        residual = target - pred
        B, C, T = witness.shape
        tf_mag_list = []

        for i in range(C):
            w = witness[:, i, :]  # shape: (B, T)

            # Compute cross-spectrum between residual and witness
            Sxy_real, Sxy_imag = self._torch_cross_welch(residual, w)
            Syy = self._torch_welch(w)

            # Estimate squared transfer function magnitude |TF|^2 = |Sxy / Syy|^2
            tf_real = Sxy_real / (Syy + 1e-12)
            tf_imag = Sxy_imag / (Syy + 1e-12)
            tf_mag = tf_real**2 + tf_imag**2  # shape: (B, F)

            # Apply frequency mask and weights
            tf_masked = tf_mag * self.freq_weights * self.freq_mask.float()

            # Apply nonlinearity
            tf_transformed = self.apply_nonlinearity(tf_masked)

            # Sum over frequency axis (normalized by total weight in band)
            loss_i = torch.sum(tf_transformed, dim=1) / torch.sum(self.freq_weights[self.freq_mask] + 1e-12)
            tf_mag_list.append(loss_i)

        # Average over all witness channels and batch
        stacked = torch.stack(tf_mag_list, dim=1)  # (B, C)
        loss = torch.mean(stacked, dim=1)          # (B,)

        return loss.mean() if self.reduction == 'mean' else loss.sum()

    
class CompositePSDLoss(nn.Module):
    """
    Composite loss function combining MSE, PSD, coherence, and transfer function losses.
    """

    def __init__(self, fs, fl, fh, fftlength=1.0, overlap=None, reduction='mean', device='cpu', psd_weight=0.3, mse_weight=0.2, tf_weight=0.3, coh_weight=0.2, average='mean', nonlinearity='log1p'):
        """
        Args:
            fs (float): Sampling frequency in Hz.
            fl, fh (float or list): Lower and upper frequency bounds.
            fftlength (float): FFT segment length in seconds.
            overlap (float or None): Segment overlap in seconds.
            reduction (str): 'mean' or 'sum'.
            device (str): Computation device ('cpu' or 'cuda').
            psd_weight, mse_weight, tf_weight, coh_weight (float): Loss component weights.
            average (str): Welch averaging mode ('mean' or 'median').
            nonlinearity (str): Nonlinearity applied in TF loss ('log1p', 'sqrt', or 'none').
        """
        super().__init__()

        self.reduction = reduction
        self.psd_weight = psd_weight
        self.mse_weight = mse_weight
        self.coh_weight = coh_weight
        self.tf_weight = tf_weight
        self.device = device

        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.psd_loss = PSDLoss(fs, fl, fh, fftlength, overlap, reduction=reduction, device=device, average=average)
        self.coh_loss = CoherenceLoss(fs, fl, fh, fftlength, overlap, reduction=reduction, device=device, average=average)
        self.tf_loss = TransferFunctionLoss(fs, fl, fh, fftlength, overlap, reduction=reduction, device=device, average=average, nonlinearity=nonlinearity)

        self.latest_loss_values = {'mse': 0.0, 'psd': 0.0, 'coh': 0.0, 'tf': 0.0, 'total': 0.0}

    def forward(self, pred, target, witness, return_dict=False):
        loss = 0.0
        loss_components = {}

        if self.mse_weight > 0:
            mse_val = self.mse_loss(pred, target)
            loss += self.mse_weight * mse_val
            loss_components['mse'] = mse_val.item()

        if self.psd_weight > 0:
            psd_val = self.psd_loss(pred, target)
            loss += self.psd_weight * psd_val
            loss_components['psd'] = psd_val.item()

        if self.coh_weight > 0:
            coh_val = self.coh_loss(pred, target, witness)
            loss += self.coh_weight * coh_val
            loss_components['coh'] = coh_val.item()

        if self.tf_weight > 0:
            tf_val = self.tf_loss(pred, target, witness)
            loss += self.tf_weight * tf_val
            loss_components['tf'] = tf_val.item()

        self.latest_loss_values.update(loss_components)
        self.latest_loss_values['total'] = loss.item() if isinstance(loss, torch.Tensor) else float(loss)

        return (loss, self.latest_loss_values) if return_dict else loss


# class CompositePSDLoss(nn.Module):
#     ''' PSD + MSE Loss with weight '''
    
#     def __init__(self, fs=1.0, fl=20., fh=500., fftlength=1., overlap=None, 
#                  asd=False, average='mean', reduction='mean', psd_weight=0.5, 
#                  mse_weight=0.5, device='cpu'):
#         super().__init__()
#         if reduction not in ('mean', 'sum'):
#             raise ValueError(
#                 '`reduction` not recognized. must be "mean" or "sum"')
#         self.reduction = reduction
        
#         self.psd_loss = PSDLoss(
#             fs=fs, fl=fl, fh=fh, fftlength=fftlength, overlap=overlap, asd=asd, 
#             average=average, reduction=reduction, device=device)
#         self.mse_loss = MSELoss(reduction=reduction)
        
#         self.psd_weight = psd_weight
#         self.mse_weight = mse_weight
                
#     def forward(self, pred, target):
#         # if weight is 0: only run 1 to save computational time
#         if self.psd_weight == 0:
#             return self.mse_loss(pred, target)
#         if self.mse_weight == 0:
#             return self.psd_loss(pred, target)
        
#         psd_loss = self.psd_weight * self.psd_loss(pred, target)
#         mse_loss = self.mse_weight * self.mse_loss(pred, target)
        
#         return (psd_loss + mse_loss)