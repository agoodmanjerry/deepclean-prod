
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.signal as sig


def plot_psd(data, fftlength=1., overlap=None, average='median', 
             ax=None, log=True, asd=True, **kargs):
    """ Compute PSD and plot PSD
    Parameters
    ----------
    data: gwpy.timeseries.TimeSeries
        Timeseries data to plot spectrogram
    fftlength: int
        Length of each segment in seconds
    overlap: int
        Number of seconds to overlap between segments
    ax: matplotlib.axes.Axes
        Axes objext 
    log: bool (default: True)
        If True, set colorbar to log scale
    fname: str (default: None)
        If given, save plot to file
    
    Returns
    -------
    freq, psd: numpy.ndarray
        Frequencies and PSD    
    """
    if overlap is None:
        overlap = fftlength / 2.
    nperseg = int(fftlength * data.sample_rate.value)
    noverlap = int(overlap * data.sample_rate.value)
    
    # Compute PSD
    freq, x = sig.welch(data.value, fs=data.sample_rate.value, nperseg=nperseg, 
                        noverlap=noverlap, average=average)
    if asd:
        x = np.sqrt(x)
        
    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        
    if log:
        ax.loglog(freq, x, **kargs)
    else:
        ax.plot(freq, x **kargs)
        
    # Return frequencies and PSD
    return freq, x


def plot_spectrogram(data, fftlength=1., overlap=None, window='tukey', ax=None, 
                     log=True, vmin=None, vmax=None, fname=None, **kargs):
    """ 
    Calculate, plot and return spectrogram
    
    Parameters
    ----------
    data: gwpy.timeseries.TimeSeries
        Timeseries data to plot spectrogram
    fftlength: int
        Length of each segment in seconds
    overlap: int
        Number of seconds to overlap between segments
    ax: matplotlib.axes.Axes
        Axes objext 
    log: bool (default: True)
        If True, set colorbar to log scale
    fname: str (default: None)
        If given, save plot to file
    
    Returns:
    --------
    freq, t: numpy.ndarray
        Frequencies and time of spectrogram
    spec: 2-D numpy.ndarray
        Spectrogram 
    """
    if overlap is None:
        overlap = fftlength / 2.
    
    nperseg = int(fftlength * data.sample_rate.value)
    noverlap = int(overlap * data.sample_rate.value)
    
    # calculate spectrogram
    freq, t, spec = sig.spectrogram(
        data.value, fs=data.sample_rate.value, nperseg=nperseg, 
        noverlap=noverlap, window=window)
    spec = np.sqrt(spec)
    if vmin is None:
        vmin = np.min(spec)
    if vmax is None:
        vmax = np.max(spec)

    # set log scale 
    norm = None
    if log:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        
    # create ax if there is None
    if ax is None:
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
    
    # plot spectrogram
    ax.pcolormesh(t, freq, spec, vmin=vmin, vmax=vmax, norm=norm, **kargs)
    ax.set(title=data.name, ylabel='Frequency [Hz]', 
           xlabel='Time [seconds] since {}'.format(data.t0.value))
    ax.colorbar()
    ax.grid(False)
    
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', dpi=300)
        plt.close()
    
    return freq, t, spec