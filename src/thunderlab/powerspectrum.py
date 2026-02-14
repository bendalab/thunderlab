"""Powerspectra and spectrograms for a given frequency resolution

## Computation of nfft

- `next_power_of_two()`: round an integer up to the next power of two.
- `nfft()`: compute nfft based on a given frequency resolution.

## Decibel

- `decibel()`: transform power to decibel.
- `power()`: transform decibel to power.

## Power spectra                

- `psd()`: power spectrum for a given frequency resolution.
- `spectrogram()`: spectrogram of a given frequency resolution and overlap fraction.

## Power spectrum analysis

- `peak_freqs()`: peak frequencies computed from power spectra of data snippets.

## Visualization

- `plot_decibel_psd()`: plot power spectrum in decibel.

## Configuration parameter

- `add_spectrum_config()`: add parameters for psd() and spectrogram() to configuration.
- `sepctrum_args()`: retrieve parameters for psd() and spectrogram() from configuration.
"""

import numpy as np

from scipy.signal import get_window
from matplotlib.mlab import psd as mpsd
try:
    from scipy.signal import welch as swelch
    psdscipy  = True
except ImportError:
    psdscipy  = False
from matplotlib.mlab import specgram as mspecgram
try:
    from scipy.signal import spectrogram as sspectrogram
    specgramscipy = True
except ImportError:
    specgramscipy = False

from .eventdetection import detect_peaks


def next_power_of_two(n):
    """The next integer power of two.
    
    Parameters
    ----------
    n: int
        A positive number.

    Returns
    -------
    m: int
        The next integer power of two equal or larger than `n`.
    """
    return int(2 ** np.floor(np.log(n) / np.log(2.0) + 1.0 - 1e-8))


def nfft(rate, freq_resolution, min_nfft=16, max_nfft=None):
    """Required number of samples for an FFT of a given frequency resolution.

    Note that the returned number of FFT samples results
    in frequency intervals that are smaller or equal to `freq_resolution`.

    Parameters
    ----------
    rate: float
        Sampling rate of the data in Hertz.
    freq_resolution: float
        Minimum frequency resolution in Hertz.
    min_nfft: int
        Smallest value of nfft to be used.
    max_nfft: int or None
        If not None, largest value of nfft to be used.

    Returns
    -------
    nfft: int
        Number of FFT points.
    """
    nfft = next_power_of_two(rate / freq_resolution)
    if not max_nfft is None:
        if nfft > max_nfft:
            nfft = next_power_of_two(max_nfft//2 + 1)
    if nfft < min_nfft:
        nfft = min_nfft
    return nfft


def decibel(power, ref_power=1.0, min_power=1e-20):
    """Transform power to decibel relative to ref_power.

    \\[ decibel = 10 \\cdot \\log_{10}(power/ref\\_power) \\]
    Power values smaller than `min_power` are set to `-np.inf`.

    Parameters
    ----------
    power: float or array
        Power values, for example from a power spectrum or spectrogram.
    ref_power: float or None or 'peak'
        Reference power for computing decibel.
        If set to `None` or 'peak', the maximum power is used.
    min_power: float
        Power values smaller than `min_power` are set to `-np.inf`.

    Returns
    -------
    decibel_psd: array
        Power values in decibel relative to `ref_power`.
    """
    if np.isscalar(power):
        tmp_power = np.array([power])
        decibel_psd = np.array([power])
    else:
        tmp_power = power
        decibel_psd = power.copy()
    if ref_power is None or ref_power == 'peak':
        ref_power = np.max(decibel_psd)
    decibel_psd[tmp_power <= min_power] = float('-inf')
    decibel_psd[tmp_power > min_power] = 10.0 * np.log10(decibel_psd[tmp_power > min_power]/ref_power)
    if np.isscalar(power):
        return decibel_psd[0]
    else:
        return decibel_psd


def power(decibel, ref_power=1.0):
    """Transform decibel back to power relative to `ref_power`.

    \\[ power = ref\\_power \\cdot 10^{decibel/10} \\]
    
    Parameters
    ----------
    decibel: array
        Decibel values of the power spectrum or spectrogram.
    ref_power: float
        Reference power for computing power.

    Returns
    -------
    power: array
        Power values of the power spectrum or spectrogram.
    """
    return ref_power * 10.0 ** (0.1 * decibel)


def psd(data, ratetime, freq_resolution=10, overlap_frac=0.5,
        n_fft=None, n_overlap=None, min_nfft=16, max_nfft=None,
        detrend='constant', window='hann'):
    """Power spectrum density of a given frequency resolution.

    If `freq_resolution` is given, the number of samples used for each
    FFT segment is computed from the requested frequency resolution
    and the sampling rate.  Check the returned frequency array for the
    actually used frequency resolution.  The frequency intervals are
    smaller or equal to `freq_resolution`.  The actually used number
    of samples used for each FFT segment can be retrieved by dividing
    the sampling rate by the actual frequency resolution:
    ```
    freq, power = psd(data, samplingrate, 0.1)
    df = np.mean(np.diff(freq))  # the actual frequency resolution
    nfft = int(samplingrate/df)
    ```

    Uses `scipy signal.welch()` if available, otherwise
    `matplotlib.mlab.psd()`.

    Parameters
    ----------
    data: 1-D array 
        Data from which power spectra are computed.
    ratetime: float or array
        If float, sampling rate of the data in Hertz.
        If array, assume `ratetime` to be the time array
        corresponding to the data.
        Compute sampling rate as `1/(ratetime[1]-ratetime[0])`.
    freq_resolution: float or None
        Desired frequency resolution of the power spectrum in Hertz.
        See `nfft()` for details.
        Alternatively, the number of samples used for computing FFTs
        can be specified by the `n_fft` argument.
    overlap_frac: float or None
        Fraction of overlap between subsequent FFT segments
        (0: no overlap, 1: complete overlap).
        Alternatively, the overlap can be specified by the `n_overlap` argument.
    n_fft: int or None
        If `freq_resolution` is None, then this is the number of
        samples used for each FFT segment.
    n_overlap: int or None    
        If `overlap_frac` is None, then this is the number of
        samples subsequent FFT segments overlap.
    min_nfft: int
        Smallest value of nfft to be used.
    max_nfft: int or None
        If not None, largest value of nfft to be used.
    detrend: string
        If 'constant' or 'mean' subtract mean of data.
        If 'linear' subtract line fitted to the data.
        If 'none' do not detrend the data.
    window: string
        Function used for windowing data segements.
        One of hann, blackman, hamming, bartlett, boxcar, triang, parzen,
        bohman, blackmanharris, nuttall, fattop, barthann
        (see scipy.signal window functions).

    Returns
    -------
    freq: 1-D array
        Frequencies corresponding to `power` array.
    power: 1-D array
        Power spectral density in [data]^2/Hz.

    Raises
    ------
    ValueError:
        Both, `freq_resolution`and `n_fft` are not specified, or
        both, `overlap_frac`and `n_overlap` are not specified.
    """
    rate = ratetime if np.isscalar(ratetime) else 1.0/(ratetime[1]-ratetime[0])
    if freq_resolution is not None:
        n_fft = nfft(rate, freq_resolution, min_nfft, max_nfft)
    if n_fft is None:
        raise ValueError('freq_resolution or n_fft needs to be specified')
    if overlap_frac is not None:
        n_overlap = int(n_fft*overlap_frac)
    if n_overlap is None:
        raise ValueError('overlap_frac or n_overlap needs to be specified')
    if n_fft >= len(data):
        n_overlap = len(data) - 1
    if psdscipy:
        if detrend == 'none':
            detrend = False
        elif detrend == 'mean':
            detrend = 'constant'
        freqs, power = swelch(data, fs=rate, nperseg=n_fft, nfft=None,
                              noverlap=n_overlap, detrend=detrend,
                              window=window, scaling='density')
    else:
        if detrend == 'constant':
            detrend = 'mean'
        power, freqs = mpsd(data, Fs=rate, NFFT=n_fft,
                            noverlap=n_overlap, detrend=detrend,
                            window=get_window(window, n_fft),
                            scale_by_freq=True)
    # squeeze is necessary when n_fft is too large with respect to the data:
    return freqs, np.squeeze(power)


def spectrogram(data, ratetime, freq_resolution=0.2, overlap_frac=0.5,
                n_fft=None, n_overlap=None, min_nfft=16, max_nfft=None,
                detrend='constant', window='hann'):
    """Spectrogram of a given frequency resolution.

    Check the returned frequency array for the actually used frequency
    resolution.
    The actual frequency resolution is smaller or equal to `freq_resolution`.
    The used number of data points per FFT segment (NFFT) is the
    sampling rate divided by the actual frequency resolution:

    ```
    spec, freq, time = spectrum(data, samplingrate, 0.1) # request 0.1Hz resolution
    df = np.mean(np.diff(freq))  # the actual frequency resolution
    nfft = int(samplingrate/df)
    ```

    Uses `scipy signal.spectrogram()` if available, otherwise
    `matplotlib.mlab.specgram()`.
    
    Parameters
    ----------
    data: 1D or 2D array of floats
        Data for the spectrograms. First dimension is time,
        optional second dimension is channel.
    ratetime: float or array
        If float, sampling rate of the data in Hertz.
        If array, assume `ratetime` to be the time array
        corresponding to the data.
        The sampling rate is then computed as `1/(ratetime[1]-ratetime[0])`.
    freq_resolution: float
        Desired frequency resolution of the spectrogram in Hertz.
        See `nfft()` for details.
        Alternatively, the number of samples used for computing FFTs
        can be specified by the `n_fft` argument.
    overlap_frac: float
        Fraction of overlap between subsequent FFT segments
        (0: no overlap, 1: complete overlap).
        Alternatively, the overlap can be specified by the `n_overlap` argument.
    n_fft: int or None
        If `freq_resolution` is None, then this is the number of
        samples used for each FFT segment.
    n_overlap: int or None    
        If `overlap_frac` is None, then this is the number of
        samples subsequent FFT segments overlap.
    min_nfft: int
        Smallest value of nfft to be used. See `nfft()` for details.
    max_nfft: int or None
        If not None, largest value of nfft to be used.
        See `nfft()` for details.
    detrend: string or False
        If 'constant' subtract mean of each data segment.
        If 'linear' subtract line fitted to each data segment.
        If `False` do not detrend the data segments.
    window: string
        Function used for windowing data segements.
        One of hann, blackman, hamming, bartlett, boxcar, triang, parzen,
        bohman, blackmanharris, nuttall, fattop, barthann, tukey
        (see scipy.signal window functions).

    Returns
    -------
    freqs: array
        Frequencies of the spectrogram.
    time: array
        Time of the nfft windows.
    spectrum: 2D or 3D array
        Power spectral density for each frequency and time.
        First dimension is frequency and second dimension is time.
        Optional last dimension is channel.

    Raises
    ------
    ValueError:
        Both, `freq_resolution`and `n_fft` are not specified, or
        both, `overlap_frac`and `n_overlap` are not specified.
    """
    rate = ratetime if np.isscalar(ratetime) else 1.0/(ratetime[1]-ratetime[0])
    if freq_resolution is not None:
        n_fft = nfft(rate, freq_resolution, min_nfft, max_nfft)
    if n_fft is None:
        raise ValueError('freq_resolution or n_fft needs to be specified')
    if overlap_frac is not None:
        n_overlap = int(n_fft*overlap_frac)
    if n_overlap is None:
        raise ValueError('overlap_frac or n_overlap needs to be specified')
    if n_fft >= len(data):
        n_overlap = len(data) - 1
    if specgramscipy:
        freqs, time, spec = sspectrogram(data, fs=rate, window=window,
                                         nperseg=n_fft, noverlap=n_overlap,
                                         detrend=detrend, scaling='density',
                                         mode='psd', axis=0)
        if data.ndim > 1:
            # scipy spectrogram() returns f x n x t:
            spec = np.transpose(spec, (0, 2, 1))
    else:
        if data.ndim > 1:
            spec = None
            for k in range(data.shape[1]):
                try:
                    ssx, freqs, time = mspecgram(data[:,k], NFFT=n_fft, Fs=rate,
                                                 noverlap=n_overlap,
                                                 detrend=detrend,
                                                 scale_by_freq=True,
                                                 scale='linear',
                                                 mode='psd',
                                                 window=get_window(window, n_fft))
                except TypeError:
                    ssx, freqs, time = mspecgram(data[:,k], NFFT=n_fft, Fs=rate,
                                                 noverlap=n_overlap,
                                                 detrend=detrend,
                                                 scale_by_freq=True,
                                                 window=get_window(window, n_fft))
                if spec is None:
                    spec = np.zeros((len(freqs), len(time), data.shape[1]))
                spec[:,:,k] = ssx
        else:
            try:
                spec, freqs, time = mspecgram(data, NFFT=n_fft, Fs=rate,
                                              noverlap=n_overlap,
                                              detrend=detrend,
                                              scale_by_freq=True, scale='linear',
                                              mode='psd',
                                              window=get_window(window, n_fft))
            except TypeError:
                spec, freqs, time = mspecgram(data, NFFT=n_fft, Fs=rate,
                                              noverlap=n_overlap,
                                              detrend=detrend,
                                              scale_by_freq=True,
                                              window=get_window(window, n_fft))
    return freqs, time, spec


def plot_decibel_psd(ax, freqs, power, ref_power=1.0, min_power=1e-20,
                     log_freq=False, min_freq=0.0, max_freq=2000.0,
                     ymarg=0.0, sstyle=dict(color='tab:blue', lw=1)):
    """Plot the powerspectum in decibel relative to `ref_power`.

    Parameters
    ----------
    ax:
        Axis for plot.
    freqs: 1-D array
        Frequency array of the power spectrum.
    power: 1-D array
        Power values of the power spectrum.
    ref_power: float
        Reference power for computing decibel. If set to `None` the maximum power is used.
    min_power: float
        Power values smaller than `min_power` are set to `np.nan`.
    log_freq: boolean
        Logarithmic (True) or linear (False) frequency axis.
    min_freq: float
        Limits of frequency axis are set to `(min_freq, max_freq)`
        if `max_freq` is greater than zero
    max_freq: float
        Limits of frequency axis are set to `(min_freq, max_freq)`
        and limits of power axis are computed from powers below max_freq
        if `max_freq` is greater than zero
    ymarg: float
        Add this to the maximum decibel power for setting the ylim.
    sstyle: dict
        Plot parameter that are passed on to the `plot()` function.
    """
    decibel_psd = decibel(power, ref_power=ref_power, min_power=min_power)
    ax.plot(freqs, decibel_psd, **sstyle)
    ax.set_xlabel('Frequency [Hz]')
    if max_freq > 0.0:
        if log_freq and min_freq < 1e-8:
            min_freq = 1.0
        ax.set_xlim(min_freq, max_freq)
    else:
        max_freq = freqs[-1]
    if log_freq:
        ax.set_xscale('log')
    dpmf = decibel_psd[freqs < max_freq]
    pmin = np.min(dpmf[np.isfinite(dpmf)])
    pmin = np.floor(pmin / 10.0) * 10.0
    pmax = np.max(dpmf[np.isfinite(dpmf)])
    pmax = np.ceil((pmax + ymarg) / 10.0) * 10.0
    ax.set_ylim(pmin, pmax)
    ax.set_ylabel('Power [dB]')


def peak_freqs(onsets, offsets, data, rate, freq_resolution=0.2,
               thresh=None, **kwargs):
    """Peak frequencies computed from power spectra of data snippets.

    Parameters
    ----------
    onsets: array of ints
        Indices indicating the onsets of the snippets in `data`.
    offsets: array of ints
        Indices indicating the offsets of the snippets in `data`.
    data: 1-D array
        Data array that contains the data snippets defined by
        `onsets` and `offsets`.
    rate: float
        Sampling rate of data in Hertz.
    freq_resolution: float
        Desired frequency resolution of the computed power spectra in Hertz.
    thresh: None or float
        If not None than this is the threshold required for the minimum height
        of the peak in the decibel power spectrum. If the peak is too small
        than the peak frequency of that snippet is set to NaN.
    kwargs: dict
        Further arguments passed on to psd().

    Returns
    -------
    freqs: array of floats
        For each data snippet the frequency of the maximum power.
    """
    freqs = []
    for i0, i1 in zip(onsets, offsets):
        if 'max_nfft' in kwargs:
            del kwargs['max_nfft']
        f, power = psd(data[i0:i1], rate, freq_resolution,
                       max_nfft=i1 - i0, **kwargs)
        if thresh is None:
            fpeak = f[np.argmax(power)]
        else:
            p, _ = detect_peaks(decibel(power, None), thresh)
            if len(p) > 0:
                ipeak = np.argmax(power[p])
                fpeak = f[p[ipeak]]
            else:
                fpeak = float('NaN')
        freqs.append(fpeak)
    return np.array(freqs)


def add_spectrum_config(cfg, freq_resolution=0.2, overlap_frac=0.5,
                        detrend='constant', window='hann'):
    """Add all parameters needed for the psd() and spectrogram() functions as a new section to a configuration.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
        
    See psd() and spectrogram() for details on the remaining arguments.
    """
    cfg.add_section('Power spectra and spectrograms:')
    cfg.add('frequencyResolution', freq_resolution, 'Hz', 'Frequency resolution of power spectra and spectrograms.')
    cfg.add('overlapFraction', 100*overlap_frac, '%', 'Overlap of subsequent data segments in power spectra and spectrograms.')
    cfg.add('detrendMethod', detrend, '', 'Detrend method to be applied on data segments for computing power spectra and spectrograms ("constant", "linear", or "none".')
    cfg.add('windowFunction', window, '', 'Window function applied on data segements for computing power spectra and spectrograms (one of "hann", "blackman", "hamming", "bartlett", "boxcar", "triang", "parzen", "bohman", "blackmanharris", "nuttall", "fattop", "barthann").')


def spectrum_args(cfg):
    """Translates a configuration to the respective parameter names of the psd() and spectrogram() functions.
    
    The return value can then be passed as key-word arguments to
    these functions.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.

    Returns
    -------
    a: dict
        Dictionary with names of arguments of the psd() and spectrogram()
        functions and their values as supplied by `cfg`.
    """
    a = cfg.map({'freq_resolution': 'frequencyResolution',
                 'overlap_frac': 'overlapFraction',
                 'detrend': 'detrendMethod',
                 'window': 'windowFunction'})
    a['overlap_frac'] *= 0.01
    return a


def main():
    import matplotlib.pyplot as plt
    print('Compute powerspectra of two sine waves (300 and 450 Hz)')

    # generate data:
    fundamentals = [300, 450]  # Hz
    rate = 100000.0      # Hz
    time = np.arange(0.0, 8.0, 1.0/rate)
    data = np.sin(2*np.pi*fundamentals[0]*time) + 0.5*np.sin(2*np.pi*fundamentals[1]*time)

    # compute power spectrum:
    freqs, power = psd(data, rate, freq_resolution=0.5,
                       detrend='none', window='hann')
    df = np.mean(np.diff(freqs))
    nfft = int(rate/df)

    # plot power spectrum:
    fig, ax = plt.subplots()
    plot_decibel_psd(ax, freqs, power,
                     sstyle=dict(lw=2,
                                 label=f'$\\Delta f={df:.1f}$ Hz, nnft={nfft}'))
    ax.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()
