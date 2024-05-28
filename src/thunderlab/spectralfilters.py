import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, sosfilt_zi
from .inputcheck import check_list, ensure_array
from .resampledata import downsampling


def sosfilter(data, rate, cutoff, mode='lp', order=1, refilter=True,
              padtype='even', padlen=None, padval=None, zi=None, init=None):
    """ Applies a digital Butterworth filter in second-order sections format.
        Includes both the sosfilt() and sosfiltfilt() function of scipy.signal.
        Data can be filtered once (forward) or twice (forward-backward, which
        doubles the order of the filter and centers the phase of the output).
        Supports low-pass, high-pass, and band-pass filter types.

        Forward filters by sosfilt() can be set to a given initial state to
        start at a certain value. Forward-backward filters by sosfiltfilt() can
        be used with padding by all built-in methods, plus an option to pad
        with a custom value. The refilter argument can be used to switch
        between the two functions.

    Parameters
    ----------
    data : 1D array (m,) or 2D array (m, n) or list of floats or ints
        Data to be filtered. Non-arrays are converted, if possible. If 1D,
        assumes a single time series. If 2D, assumes that each column is a
        separate time series and performs filtering along the first axis.
    rate : float or int
        Sampling rate of the data in Hz. Must be the same for all columns.
    cutoff : float or int
        Cut-off frequency of the applied filter in Hz. If mode is 'lp' or 'hp',
        must be a single value. If mode is 'bp', must be a tuple of two values.
    mode : str, optional
        Type of the applied filter. Options are 'lp' (low-pass), 'hp'
        (high-pass), and 'bp' (band-pass). The default is 'lp'.
    order : int, optional
        Order of the applied filter. If refilter is True, actual filter order
        is twice the specified order. The default is 1.
    refilter : bool, optional
        If True, uses the forward-backward filtering method of sosfiltfilt().
        This enables the use of padtype, padlen, and padval to control padding.
        If False, uses the forward filtering method of sosfilt(). This enables
        the use of zi and init to set the filter state. The default is True.
    padtype : str, optional
        Method used to pad the data before forward-backward filtering. Can be:
        # 'constant': Pads with first and last value of data, respectively.
        -> For signals with assessable endpoints (e.g. bound to some baseline).
        # 'even': Mirrors the data around each endpoint.
        -> For (noisy) signals whose statistics do not change much over time.
        # 'odd': Mirrors data, then turns it 180° around the endpoint.
        -> For oscillatory signals or where stable phase and smoothness is key.
        # 'fixed': Pads with custom value (managed externally, not by scipy).
        -> For signals that are meant to be seen in a certain temporal context.
        # None: No padding.
        Ignored if refilter is False. The default is 'even'.
    padlen : int, optional
        Number of points added to both sides of data. Applies for each padtype.
        Calculated by sosfiltfilt() as a very small number of points if None.
        Ignored if refilter is False or padtype is None. The default is None. 
    padval : float or int, optional
        If specified and padtype is 'fixed', uses a constant padding with this
        value. Set to the mean of each column if None. Ignored if refilter is
        False or padtype is not 'fixed'. The default is None.
    zi : ND array of floats, optional
        If specified, defines the initial state of the applied forward filter.
        Required dimensionality depends on input shape and type and order of
        the filter. If None, assumes the start value provided by init and sets
        the filter state accordingly. Ignored if refilter is True. The default
        is None.
    init : float or int, optional
        If specified and zi is None, sets the initial filter state to this
        start value. If None, assumes the first value of each column. Ignored
        if refilter is True or zi is specified. The default is None.

    Returns
    -------
    filtered : 1D array of floats (m,) or 2D array of floats (m, n)
        Filtered data. If refilter is True, output of sosfiltfilt(), else
        sosfilt(). Returns unchanged (instead of low-pass filtered) data if the
        cut-off frequency is larger than Nyquist frequency (rate / 2). Returns
        high-pass (instead of band-pass) filtered data if the upper cut-off
        frequency is above Nyquist. Returns 1D if input was 1D, else 2D.
    next_state : ND array of floats
        State of the applied filter at the end of forward filtering. Only
        returned if refilter is False. Can be used as initial state for the
        next filtering step.
    """   
    # Nyquist early exit (low-pass):
    if mode == 'lp' and cutoff > rate / 2:
        return data
    # Assert 2D array (columns):
    data, in_shape = ensure_array(var=data, dims=(1, 2), shape=(-1, None),
                                  list_shapes=True)

    # Initialize filter as second-order sections:
    if mode == 'bp' and cutoff[1] > rate / 2:
        # Nyquist fallback (band-pass):
        mode, cutoff = 'hp', cutoff[0]
    sos = butter(order, cutoff, mode, fs=rate, output='sos')

    # Forward filtering:
    if not refilter:
        if zi is None:
            # Set initial state to start value:
            zi = sosfilt_zi(sos).reshape(1, -1, 1)
            zi += data[0, :] if init is None else init
        # Apply filter once with given initial state (no padding):
        filtered, next_state = sosfilt(sos, data, axis=0, zi=zi)
        return filtered.ravel() if len(in_shape) == 1 else filtered, next_state
        
    # Forward-backward filtering:
    if padtype == 'fixed':
        if padlen is None:
            # Auto-generated with scipy default:
            padlen = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(),
                                                 (sos[:, 5] == 0).sum()))
        if padval is None:
            # Mean-centered padding:
            padval = np.mean(data, axis=0)
        # Add constant padding with custom value along first axis:
        data = np.pad(data, ((padlen, padlen), (0, 0)), constant_values=padval)

    # Apply filter twice with given padding method:
    filtered = sosfiltfilt(sos, data, axis=0, padlen=padlen,
                           padtype=None if padtype == 'fixed' else padtype)

    # Remove custom padding:
    if padtype == 'fixed':
        filtered = filtered[padlen:data.shape[0] - padlen, :]
    return filtered.ravel() if len(in_shape) == 1 else filtered


def multi_lowpass(data, rate, cutoffs, new_rate=None, rectify=False, **kwargs):
    """ Temporal averaging of data on multiple different time scales.
        Applies separate low-pass filters with the given cut-off frequencies.
        Data can be rectified before filtering to perform envelope extraction.
        Filtered data can then be downsampled to a new rate.

    Parameters
    ----------
    data : 1D array (m,) or 2D array (m, n) or list of floats or ints
        Data to be averaged by low-pass filtering. Non-arrays are converted, if
        possible. If 1D, assumes a single time series. If 2D, assumes that each
        column is a separate time series and averages along the first axis.
    rate : float or int
        Sampling rate of data in Hz. Must be the same for all columns.
    cutoffs : float or int or list of floats or ints (p,)
        Cut-off frequency of each applied low-pass filter in Hz. Accepts
        scalars (discouraged, use sosfilter() with downsampling() instead). For
        each specified cut-off frequency, adds a block with as many columns as
        data to the filtered array.
    new_rate : float or int
        If specified, downsamples filtered data to this rate in Hz. Ignored if
        new_rate >= rate. The default is None.
    rectify : bool, optional
        If True, applies np.abs() to data before low-pass filtering, turning
        temporal averaging into envelope extraction. The default is False.
    **kwargs : dict, optional
        Additional keyword arguments passed to sosfilter() to control the order
        of the low-pass filter, the filtering method, padding and start values.

    Returns
    -------
    filtered : 2D array of floats (q, n * p)
        Temporally averaged data, optionally downsampled. Columns correspond to
        filtered time series and are ordered block-wise by cut-off frequency of
        each applied low-pass filter. Block order is the same as in cutoffs.
        Within-block column order is the same as in data.
    """    
    # Assert 2D array (columns):
    data = ensure_array(var=data, dims=(1, 2), shape=(-1, None))
    # Assert iterable:
    cutoffs = check_list(var=cutoffs)
    # Manage envelope mode:
    if rectify:
        data = np.abs(data)
    # Manage downsampling:
    if new_rate is None:
        new_rate = rate
    
    # Time series in data array:
    n_columns = data.shape[1]
    # Length of the (downsampled) time axis:
    n_resampled = int(np.round(new_rate / rate * data.shape[0]))
    # Initialize filtered array as horizontal tile of data array:
    filtered = np.zeros((n_resampled, n_columns * len(cutoffs)))
    for i, cutoff in enumerate(cutoffs):
        # Indices of next filter block along second axis:
        block = np.arange(i * n_columns, (i + 1) * n_columns, dtype=int)
        # Apply low-pass filter for given block to data array:
        filter_block = sosfilter(data, rate, cutoff, 'lp', **kwargs)
        filtered[:, block] = downsampling(filter_block, rate, new_rate)
    return filtered

