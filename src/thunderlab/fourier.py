"""Fourier series

Extract and normalize Fourier coefficients of a Fourier series of
periodic functions, or synthesize periodic functions from Fourier
coefficients.

## Functions

- `fourier_coeffs()`: extract Fourier coefficients from data.
- `normalize_fourier_coeffs()`: set phase of first harmonics and offset to zero.
- `fourier_synthesis()`: compute waveform from Fourier coefficients.

"""

import numpy as np


def fourier_coeffs(data, ratetime, freq, n_harmonics):
    """ Extract Fourier coefficients from data.

    Decompose a periodic signal \\(x(t)\\) with known fundamental frequency
    \\(f_1\\) into a Fourier series:
    \\[ x(t) \\approx \\Re \\sum_{k=0}^n c_k e^{i 2 \\pi k f_1 t} \\]
    with the Fourier coefficients
    \\[ c_k = \\frac{2}{jT} \\int_{0}^{jT} x(t) e^{-i 2 \\pi k f_1 t} \\, dt \\]
    integrated over integer multiples of the period \\(T=1/f_1\\).

    Parameters
    ----------
    data: 1D array of float
        Time series of data.
    ratetime: float or 1-D array of float
        Times corresponding to `data`.
        If single float, then sampling rate of the data.
    freq: float
        Fundamental frequency of Fourier series.
    n_harmonics: int
        Number of harmonics inclusively the zeroth one.
        That is, if `n_harmonics` is set to two,
        then the zeroth harmonics (offset) and the fundamental
        frequency (first harmonics) are returned.

    Returns
    -------
    coeffs: 1D array of complex
        For each harmonics the complex valued Fourier coefficient.
        The first one is the offset and the second one is the coefficient
        of the fundamental. If the number ofdata samples is less than
        a single period, then a zero-sized array is returned.
    """
    if isinstance(ratetime, (list, tuple, np.ndarray)):
        time = ratetime
    else:
        time = np.arange(len(data))/ratetime
    # integrate over full periods:
    n_periods = int(np.floor((time[-1] - time[0])*freq))
    if n_periods < 1:
        return np.zeros(0, dtype=complex)
    n_max = np.argmax(time > time[0] + n_periods/freq)
    data = data[:n_max]
    time = time[:n_max]
    # Fourier projections:
    iomega = -2j*np.pi*freq*time
    fac = 2/len(data)       # = 2*deltat/T
    coeffs = np.zeros(n_harmonics, dtype=complex)
    for k in range(n_harmonics):
        coeffs[k] = np.sum(data*np.exp(iomega*k))*fac
    return coeffs


def normalize_fourier_coeffs(coeffs):
    """Set phase of first harmonics and offset to zero.

    Parameters
    ----------
    coeffs: 1D array of complex
        For each harmonic the complex valued Fourier coefficient
        as, for example, returned by `fourier_coeffs()`.
        The first one is the offset and the second one is the coefficient
        of the fundamental.

    Returns
    -------
    coeffs: 1D array of complex
        The normalized Fourier coefficients.
    """
    phi1 = np.angle(coeffs[1])
    for k in range(1, len(coeffs)):
        coeffs[k] *= np.exp(-1j*k*phi1)
    coeffs[0] = 0 + 0j
    return coeffs


def fourier_synthesis(freq, coeffs, ratetime, n=None):
    """ Compute periodic waveform from Fourier coefficients.

    Given the Fourier coefficients
    \\[ c_k = \\frac{2}{jT} \\int_{0}^{jT} x(t) e^{-i 2 \\pi k f_1 t} \\, dt \\]
    integrated over integer multiples of the period \\(T=1/f_1\\) of a signal
    \\(x(t)\\) with fundamental frequency \\(f_1\\), compute
    the Fourier series
    \\[ x(t) \\approx \\Re \\sum_{k=0}^n c_k e^{i 2 \\pi k f_1 t} \\]

    Parameters
    ----------
    freq: float
        Fundamental frequency.
    coeffs: 1D array of complex
        For each harmonics the complex valued Fourier coefficient
        as, for example, returned by `fourier_coeffs()`.
        The first one is the offset.
    ratetime: float or 1-D array of float
        Time points for which the waveform is calculated.
        If single float, then sampling rate of the computed waveform.
    n: int
        Number of samples if `ratetime` is float.

    Returns
    -------
    wave: 1D array of float
        Waveform computed from Fourier series with fundamental frequency
        `freq` and Fourier coefficients `coeffs` for each harmonic.
        The waveform is computed for a sampling rate `rate` and contains
        `n` samples.
    """
    if isinstance(ratetime, (list, tuple, np.ndarray)):
        time = ratetime
    else:
        time = np.arange(n)/ratetime
    iomega = 2j*np.pi*freq*time
    wave = np.zeros(len(time))
    for k in range(len(coeffs)):
        wave += np.real(coeffs[k]*np.exp(iomega*k))
    return wave

