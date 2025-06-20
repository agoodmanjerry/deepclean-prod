
import math
import warnings

import numpy as np
import scipy.signal as sig


def _parse_window(nperseg, noverlap, window='boxcar'):
    """ Get window function """
    if window == 'rect' or window =='rectangular':
        window = 'boxcar'
    window_fn = vars(sig)[window]
    if window != 'boxcar':
        return window_fn(nperseg) * (nperseg - noverlap) / nperseg * 2
    else:
        return window_fn(nperseg) * (nperseg - noverlap) / nperseg


def resample(data, fs, new_fs, window='hamming', n=60):
    """ Resample using FIR filter. Method borrowed from gwpy.timeseries.TimeSeries
    Parameters
    ----------
    data: `numpy.ndarray`
        timeseries data for resampling
    fs: `float`
        sampling rate in Hz
    new_fs: `float`
        new sampling rate in Hz
    window: `str`, `numpy.ndarray`
    n: `int`
        number of taps
    Returns
    -------
    `numpy.ndarray`
        resampled timeseries data
    """
    factor = (fs / new_fs)
    if math.isclose(factor, 1., rel_tol=1e-09, abs_tol=0.):
        warnings.warn(
            "resample() rate matches current sample_rate ({}), returning "
            "input data unmodified; please double-check your "
            "parameters".format(fs),
            UserWarning,
        )
        return data
    
    # if integer down-sampling, use decimate
    if factor.is_integer():
        filt = sig.firwin(n + 1, 1./factor, window=window)
        new = sig.filtfilt(filt, [1.], data)        
        return new[::int(factor)]        
    # otherwise use Fourier filtering
    else:
        nsamp = int(len(data) * new_fs / fs)
        new = sig.resample(data, nsamp, window=window)
        return new

    
def bandpass(data, fs, fl, fh, order=None, axis=-1):
    """ Apply Butterworth bandpass filter using scipy.signal.sosfiltfilt method
    Parameters
    ----------
    data: array
    fs: sampling frequency
    fl, fh: low and high frequency for bandpass
    axis: axis to apply the filter on 
    
    Returns:
    --------
    data_filt: filtered array 
    """
    if order is None:
        order = 8
        
    # Make filter
    nyq = fs/2.
    low, high = fl/nyq, fh/nyq  # normalize frequency
    z, p, k = sig.butter(order, [low, high], btype='bandpass', output='zpk')
    sos = sig.zpk2sos(z, p, k)

    # Apply filter and return output
    data_filt = sig.sosfiltfilt(sos, data, axis=axis)
    return data_filt


def overlap_add(data, noverlap, window, verbose=True):
    """ Concatenate timeseries using the overlap-add method 
    Parameters
    -----------
    data: `numpy.ndarray` of shape (N, nperseg)
        array of timeseries segments to be concatenate
    noverlap: `int`
        number of overlapping samples between each segment in `data`
    window: `str`, `numpy.ndarray`
    
    Returns
    --------
    `numpy.ndarray`
        concatenated timeseries
    """
    # Get dimension
    N, nperseg = data.shape
    stride = nperseg - noverlap
    
    # Get window function
    if isinstance(window, str):
        window = _parse_window(nperseg, noverlap, window)
        
    # Concatenate timeseries
    nsamp = int((N - 1) * stride + nperseg)
    new = np.zeros(nsamp)    
    for i in range(N):
        new[i * stride: i * stride + nperseg] += data[i] * window
    return new


def filter_add(data, fs, flow, fhigh, order=8, axis=-1):
    """ Sub bandpass filter 
    Parameters
    ----------
    data: `numpy.ndarray`
    fs: `float`
        sampling frequency in Hz
    flow: `float`
        list of low frequencies
    fhigh: `float`
        list of high frequencies
    Returns
    -------
    `numpy.ndarray`
        Data with sub bandpass filter
    """
    if isinstance(flow, (int, float)):
        flow = (flow, )
    if isinstance(fhigh, (int, float)):
        fhigh = (fhigh, )
    
    new = np.zeros_like(data)
    for fl, fh in zip(flow, fhigh):
        new += bandpass(data, fs, fl, fh, order=order, axis=axis)
    return new
