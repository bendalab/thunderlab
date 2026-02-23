import pytest
import numpy as np
import thunderlab.fourier as f


def test_fourier():
    amplitudes = (1.0087, 0.23201, 0.060524, 0.020175, 0.010087, 0.0080699)
    phases = (1.3414, 1.3228, 2.9242, 2.8157, 2.6871, -2.8415)
    # make coefficients:
    acoeffs = np.zeros(len(amplitudes) + 1, dtype=complex)
    for k in range(len(amplitudes)):
        acoeffs[k + 1] = amplitudes[k]*np.exp(1j*phases[k])

    rate = 100_000
    n = 100_000
    freq = 473.5
    data1 = f.fourier_synthesis(freq, acoeffs, rate, n)
    time = np.arange(n)/rate
    data = f.fourier_synthesis(freq, acoeffs, time)
    assert np.all(np.abs(data1-data) < 1e-8), 'synthesized waveforms differ'

    bcoeffs = f.fourier_coeffs(data, rate, freq, len(acoeffs))
    assert len(acoeffs) == len(bcoeffs), 'different number of coefficients'
    assert np.all(np.abs(np.abs(acoeffs) - np.abs(bcoeffs)) < 1e-2), 'magnitudes differ'
    assert np.all(np.abs(np.angle(acoeffs[1:]) - np.angle(bcoeffs[1:])) < 1e-2), 'phases differ'

    ncoeffs = f.normalize_fourier_coeffs(bcoeffs)
    assert np.abs(ncoeffs[0]) == 0.0, 'offset not null'
    assert np.abs(np.angle(ncoeffs[1])) < 1e-8, 'phase of fundamental not null'

        
