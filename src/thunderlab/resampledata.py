import numpy as np
from scipy.interpolate import interpn
from .inputcheck import ensure_array


def downsampling(data, rate, new_rate):
    """ Resamples time series data of given sampling rate to a lower new rate.

    Parameters
    ----------
    data : 1D array (m,) or 2D array (m, n) or list of floats or ints
        Data to be downsampled. Non-arrays are converted, if possible. If 1D,
        assumes a single time series. If 2D, assumes that each column is a
        separate time series and performs downsampling along the first axis.
    rate : float or int
        Current sampling rate of data in Hz. Must be the same for all columns.
    new_rate : float or int
        New sampling rate of data in Hz. Must be smaller than rate.

    Returns
    -------
    downsampled : 1D array (p,) or 2D array (p, n) of floats or ints
        Resampled data with the specified new rate. Returns data unchanged if
        new_rate is above rate. Returns 1D if input was 1D, else 2D.
    """
    # Rate conflict early exit:
    if new_rate >= rate:
        return data
    # Assert 2D array (columns):
    data, in_shape = ensure_array(var=data, dims=(1, 2), shape=(-1, None),
                                  list_shapes=True)
    # Downsampling ratio:
    n = rate / new_rate
    # New rate is clean multiple of old rate:
    if abs(n - np.round(n)) < 0.01:
        # Simple nth-entry selection along first axis:
        downsampled = data[::int(np.round(n)), :]
        return downsampled.ravel() if len(in_shape) == 1 else downsampled

    # Non-integer ratio requires interpolation:
    t = np.arange(data.shape[0]) / rate
    new_t = np.arange(0, t[-1], 1 / new_rate)
    if data.shape[1] == 1:
        # Standard 1D interpolation:
        downsampled = np.interp(new_t, t, data.ravel())
        if len(in_shape) == 2:
            # Restore to original 2D shape:
            downsampled = downsampled.reshape(-1, 1)
    else:
        # 2D interpolation along first axis:
        columns = np.arange(data.shape[1])
        coords = (t, columns)
        # Mesh of new points to interpolate (per column):
        row_grid, col_grid = np.meshgrid(new_t, columns, indexing='ij')
        points = list(zip(row_grid.ravel(), col_grid.ravel()))
        # Bring interpolated points into original 2D shape:
        downsampled = interpn(coords, data, points).reshape(-1, len(columns))
    return downsampled

