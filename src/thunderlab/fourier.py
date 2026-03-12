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


def fourier_coeffs(data, ratetime, freq, max_harmonics):
    """ Extract Fourier coefficients from data.

    Decompose a periodic, real-valued signal \\(x(t)\\) with known fundamental frequency
    \\(f_1\\) into a Fourier series:
    \\[ x(t) \\approx \\Re \\sum_{k=0}^n c_k e^{i 2 \\pi k f_1 t} \\]
    with the Fourier coefficients
    \\[ c_0 = \\frac{1}{jT} \\int_{0}^{jT} x(t) \\, dt , \\quad k = 0 \\]
    and
    \\[ c_k = \\frac{2}{jT} \\int_{0}^{jT} x(t) e^{-i 2 \\pi k f_1 t} \\, dt , \\quad k > 0 \\]
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
    max_harmonics: int
        The highest harmonics for which to compute the Fourier coefficient.
        The number of coefficients returned is one more than `max_harmonics`,
        because the first coefficient is the zeroth harmonics.
        For example, if `max_harmonics` is set to three,
        then the zeroth harmonics (offset), the fundamental
        frequency (first harmonics), and the second harmonics are returned.

    Returns
    -------
    coeffs: 1D array of complex
        For each harmonics the complex valued Fourier coefficient.
        The first one is the offset and the second one is the coefficient
        of the fundamental. If the number of data samples is less than
        a single period, then a zero-sized array is returned.
    """
    if isinstance(ratetime, (list, tuple, np.ndarray)):
        time = ratetime
    else:
        time = np.arange(len(data))/ratetime
    # integrate over full periods:
    n_periods = int(np.floor((time[-1] - time[0])*freq))
    if n_periods < 1 or max_harmonics < 0:
        return np.zeros(0, dtype=complex)
    n_max = np.argmax(time >= time[0] + n_periods/freq)
    data = data[:n_max]
    time = time[:n_max]
    # Fourier projections:
    iomega = -2j*np.pi*freq*time
    fac = 2/len(data)       # = 2*deltat/T
    coeffs = np.zeros(max_harmonics + 1, dtype=complex)
    coeffs[0] = np.sum(data)*fac/2
    for k in range(1, max_harmonics + 1):
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
    \\[ c_0 = \\frac{1}{jT} \\int_{0}^{jT} x(t) \\, dt , \\quad k = 0 \\]
    and
    \\[ c_k = \\frac{2}{jT} \\int_{0}^{jT} x(t) e^{-i 2 \\pi k f_1 t} \\, dt , \\quad k > 0 \\]
    integrated over integer multiples of the period \\(T=1/f_1\\) of a
    periodic,  real-valued signal \\(x(t)\\) with fundamental frequency \\(f_1\\),
    compute the Fourier series
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
        Real-valued waveform computed from Fourier series with fundamental frequency
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
    if len(coeffs) == 0:
        return wave
    for k, c in enumerate(coeffs):
        wave += np.real(coeffs[k]*np.exp(iomega*k))
    return wave


def main():
    """Demonstrate the fourier module.
    """
    import matplotlib.pyplot as plt
    
    f = 1/1.5
    t = np.linspace(0, 3.0, 200)
    x = 1.7 + 2.3*np.cos(2*np.pi*f*t) + 0.8*np.cos(2*np.pi*2*f*t + 0.5*np.pi) + 0.4*np.cos(2*np.pi*3*f*t - 1.6*np.pi)
    coeffs = fourier_coeffs(x, t, f, 6)
    y = fourier_synthesis(f, coeffs, t)
    fig, (axt, axc) = plt.subplots(1, 2, layout='constrained')
    axt.plot(t, x, label='original', lw=6)
    axt.plot(t, y, label='synthesized', lw=2)
    axt.legend()
    axc.plot(np.real(coeffs), '-o', label='real')
    axc.plot(np.imag(coeffs), '-o', label='imag')
    axc.legend()
    plt.show()


if __name__ == '__main__':
    main()
