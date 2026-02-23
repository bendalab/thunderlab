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


def fourier_coeffs(data, rate, freq, n_harmonics):
    """ Extract Fourier coefficients from data.

    Parameters
    ----------
    data: 1D array of float
        Time series of data.
    rate: float
        Sampling rate of data.
    freq: float
        Fundamental frequency of Fourier series.
    n_harmonics: int
        Number of harmonics.

    Returns
    -------
    coeffs: 1D array of complex
        For each harmonics the complex valued Fourier coefficient.
        The first one is the offset and the second one is the coefficient
        of the fundamental.
    """
    deltat = 1/rate
    t = np.arange(len(data))*deltat
    iomega = -2j*np.pi*freq*t
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
    phi0 = np.angle(coeffs[1])
    for k in range(1, len(coeffs)):
        coeffs[k] *= np.exp(-1j*k*phi0)
    coeffs[0] = 0 + 0j
    return coeffs


def fourier_synthesis(freq, coeffs, ratetime, n=None):
    """ Compute periodic waveform from Fourier coefficients.

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

