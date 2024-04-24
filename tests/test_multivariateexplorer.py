import pytest
import numpy as np
import matplotlib.pyplot as plt
import thunderlab.multivariateexplorer as me


def test_multivariateexplorer():
    # generate data:
    n = 100
    data = []
    data.append(np.random.randn(n) + 2.0)
    data.append(1.0+0.1*data[0] + 1.5*np.random.randn(n))
    data.append(-3.0*data[0] - 2.0*data[1] + 1.8*np.random.randn(n))
    idx = np.random.randint(0, 3, n)
    names = ['aaa', 'bbb', 'ccc']
    data.append([names[i] for i in idx])
    # generate waveforms:
    waveforms = []
    time = np.arange(0.0, 10.0, 0.01)
    for r in range(len(data[0])):
        x = data[0][r]*np.sin(2.0*np.pi*data[1][r]*time + data[2][r])
        y = data[0][r]*np.exp(-0.5*((time-data[1][r])/(0.2*data[2][r]))**2.0)
        waveforms.append(np.column_stack((time, x, y)))
    # initialize explorer:
    expl = me.MultivariateExplorer(data,
                                   list(map(chr, np.arange(len(data))+ord('A'))),
                                   'Explorer')
    expl.set_wave_data(waveforms, 'Time', ['Sine', 'Gauss'])
    # explore data:
    expl.set_colors()
    expl.show(False)
    class kev: pass
    kev.key = 'c'
    for k in range(5):
        expl._on_key(kev)
    kev.key = 'C'
    for k in range(5):
        expl._on_key(kev)
    kev.key = 'n'
    for k in range(5):
        expl._on_key(kev)
    kev.key = 'N'
    for k in range(5):
        expl._on_key(kev)
    kev.key = 'h'
    expl._on_key(kev)
    expl._on_key(kev)
    kev.key = 'down'
    expl._on_key(kev)
    expl._on_key(kev)
    kev.key = 'right'
    expl._on_key(kev)
    kev.key = 'down'
    expl._on_key(kev)
    kev.key = 'left'
    expl._on_key(kev)
    kev.key = 'up'
    expl._on_key(kev)
    kev.key = 'escape'
    expl._on_key(kev)
    kev.key = '+'
    expl._on_key(kev)
    kev.key = '-'
    expl._on_key(kev)
    kev.key = '0'
    expl._on_key(kev)
    kev.key = 'pageup'
    for k in range(5):
        expl._on_key(kev)
    kev.key = 'pagedown'
    for k in range(5):
        expl._on_key(kev)
    kev.key = 'p'
    for k in range(3):
        expl._on_key(kev)
    kev.key = 'P'
    for k in range(3):
        expl._on_key(kev)
    kev.key = 'w'
    for k in range(2):
        expl._on_key(kev)
    kev.key = 'ctrl+a'
    expl._on_key(kev)
    kev.key = 'l'
    expl._on_key(kev)

    
def test_main():
    me.main('')
