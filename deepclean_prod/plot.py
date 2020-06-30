
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.signal as sig


def plot_spectrogram(data, fftlength=None, overlap=None, window='tukey', ax=None, 
                     log=True, vmin=None, vmax=None, fname=None):
    """ 
    Plot and return spectrogram
    
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
        Frequencies and time of of spectrogram
    spec: 2-D numpy.ndarray
        Spectrogram 
    """
    
    # calculate spectrogram
    freq, t, spec = sig.spectrogram(
        data.value, fs=data.sample_rate.value, nperseg=nperseg, 
        noverlap=noverlap, window=window)
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
    ax.pcolormesh(t, freq, spec, vmin=vmin, vmax=vmax, norm=norm)
    ax.set(title=data.name, ylabel='Frequency [Hz]', 
           xlabel='Time [seconds] since {}'.format(data.t0.value))
    ax.colorbar()
    
    if fname is not None:
        fig.savefig(fname, bbox_inches='tight', dpi=300)
    
    return freq, t, spec
