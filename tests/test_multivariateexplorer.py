import pytest
import numpy as np
import matplotlib.pyplot as plt
import thunderlab.tabledata as td
import thunderlab.multivariateexplorer as me


def generate_table():
    n = 20
    data = td.TableData()
    data.append('A', unit='s', value=np.random.randn(n) + 2.0)
    data.append('B', value=1.0+0.1*data[:, 'A'] + 1.5*np.random.randn(n))
    data.append('C', value=-3.0*data[:, 'A'] - 2.0*data[:, 'B'] + 1.8*np.random.randn(n))
    idx = np.random.randint(0, 3, n)
    names = ['aaa', 'bbb', 'ccc']
    data.append('D', value=[names[i] for i in idx])
    return data


def test_categorize():
    n = 20
    idx = np.random.randint(0, 3, n)
    names = ['aaa', 'bbb', 'ccc']
    data = [names[i] for i in idx]
    cat, cdata = me.categorize(data)
    assert len(cat) == 3, 'categorize() number of categories'
    for s in names:
        assert s in cat, f'categorize() containes "{s}"'
    assert len(cdata) == len(data), 'categorize() data rows'
    assert np.all(cdata == idx), 'categorize() indices match'


def test_select_features():
    data = generate_table()
    data_cols = me.select_features(data, [])
    assert len(data_cols) == len(data), 'select_features(): select all'
    data_cols = me.select_features(data, ['A', 'B', 'A', 'X', 'C'])
    assert len(data_cols) == 2, 'select_features(): select'
    assert data_cols[0] == 1, 'select_features(): select'
    assert data_cols[1] == 2, 'select_features(): select'


def test_select_coloring():
    data = generate_table()
    data_cols = [1, 3]
    colors, color_label, color_idx, msg = me.select_coloring(data,
                                                             data_cols,
                                                             'X')
    assert colors is None, 'select_coloring() invalid color feature'
    assert color_label is None, 'select_coloring() invalid color feature'
    assert color_idx is None, 'select_coloring() invalid color feature'
    assert 'not a valid' in msg, 'select_coloring() invalid color feature'

    colors, color_label, color_idx, msg = me.select_coloring(data,
                                                             data_cols,
                                                             'row')
    assert colors == -2, 'select_coloring() row color feature'
    assert color_label is None, 'select_coloring() row color feature'
    assert color_idx is None, 'select_coloring() row color feature'
    assert msg is None, 'select_coloring() row color feature'

    colors, color_label, color_idx, msg = me.select_coloring(data,
                                                             data_cols,
                                                             'D')
    assert colors == 1, 'select_coloring() color feature in selected data columns'
    assert color_label is None, 'select_coloring() color feature in selected data columns'
    assert color_idx == 3, 'select_coloring() color feature in selected data columns'
    assert msg is None, 'select_coloring() color feature in selected data columns'

    colors, color_label, color_idx, msg = me.select_coloring(data,
                                                             data_cols,
                                                             'C')
    assert np.all(colors == data[:, 'C']), 'select_coloring() color feature not in selected data columns'
    assert 'C' in color_label, 'select_coloring() color feature not in selected data columns'
    assert color_idx == 2, 'select_coloring() color feature not in selected data columns'
    assert msg is None, 'select_coloring() color feature not in selected data columns'

    colors, color_label, color_idx, msg = me.select_coloring(data,
                                                             data_cols,
                                                             'A')
    assert np.all(colors == data[:, 'A']), 'select_coloring() color feature not in selected data columns'
    assert 'A' in color_label, 'select_coloring() color feature not in selected data columns'
    assert color_idx == 0, 'select_coloring() color feature not in selected data columns'
    assert msg is None, 'select_coloring() color feature not in selected data columns'


def test_list_available_features():
    data = generate_table()
    data_cols = [1, 3]
    me.list_available_features(data, data_cols, color_col=1)

    
def test_multivariateexplorer_inputs():
    data = generate_table()
    labels = ['a', 'b', 'c', 'd']
    
    expl = me.MultivariateExplorer(data)
    assert len(expl.raw_labels) == len(data.keys()), 'MultivariateExplore: table labels'
    
    expl = me.MultivariateExplorer(data, labels)
    assert expl.raw_labels == ['a', 'b', 'c', 'd'], 'MultivariateExplore: extra labels'

    adata = np.zeros((data.rows(), 3))
    adata[:, 0] = data[:, 'A']
    adata[:, 1] = data[:, 'B']
    adata[:, 2] = data[:, 'C']
    expl = me.MultivariateExplorer(adata, labels)
    assert np.all(expl.raw_data == adata), 'MultivariateExplore: ndarray'

    ldata = []
    ldata.append(data[:, 'A'])
    ldata.append(data[:, 'B'])
    ldata.append(data[:, 'C'])
    ldata.append(data[:, 'D'])
    expl = me.MultivariateExplorer(ldata, labels)
    assert expl.raw_data.shape[1] == len(ldata), 'MultivariateExplore: list'
    assert expl.raw_data.shape[0] == len(ldata[0]), 'MultivariateExplore: list'
    assert np.all(expl.raw_data[:, 0] == ldata[0]), 'MultivariateExplore: list'
    assert np.all(expl.raw_data[:, 1] == ldata[1]), 'MultivariateExplore: list'
    assert np.all(expl.raw_data[:, 2] == ldata[2]), 'MultivariateExplore: list'

    
def test_multivariateexplorer_nans():
    data = generate_table()
    data[data.rows()//2, 'B'] = np.nan
    data.append('E', value=[np.nan for i in range(data.rows())])
    expl = me.MultivariateExplorer(data)
    assert len(data) == expl.raw_data.shape[1] + 1, 'MultivariateExplore: remove all nan column'
    assert len(data[:, 'A']) == len(expl.raw_data) + 1, 'MultivariateExplore: remove nan rows'

    
def test_multivariateexplorer_onkey():
    data = generate_table()
    # generate waveforms:
    waveforms = []
    time = np.arange(0.0, 10.0, 0.01)
    for r in range(len(data[:,0])):
        x = data[r, 0]*np.sin(2.0*np.pi*data[r, 1]*time + data[r, 2])
        y = data[r, 0]*np.exp(-0.5*((time-data[r, 1])/(0.2*data[r, 2]))**2.0)
        waveforms.append(np.column_stack((time, x, y)))
    # initialize explorer:
    expl = me.MultivariateExplorer(data, title='Explorer')
    expl.set_wave_data(waveforms, 'Time', ['Sine', 'Gauss'])
    # explore data:
    expl.set_colors()
    expl.show(False)
    # test key shortcuts:
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
    kev.key = 'H'
    expl._on_key(kev)
    expl._on_key(kev)
    kev.key = 'down'
    for k in range(10):
        expl._on_key(kev)
    kev.key = 'right'
    for k in range(10):
        expl._on_key(kev)
    kev.key = 'down'
    for k in range(10):
        expl._on_key(kev)
    kev.key = 'left'
    for k in range(10):
        expl._on_key(kev)
    kev.key = 'up'
    for k in range(10):
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

    
def test_multivariateexplorer_onselect():
    data = generate_table()
    # generate waveforms:
    waveforms = []
    time = np.arange(0.0, 10.0, 0.01)
    for r in range(len(data[:,0])):
        x = data[r, 0]*np.sin(2.0*np.pi*data[r, 1]*time + data[r, 2])
        y = data[r, 0]*np.exp(-0.5*((time-data[r, 1])/(0.2*data[r, 2]))**2.0)
        waveforms.append(np.column_stack((time, x, y)))
    # initialize explorer:
    expl = me.MultivariateExplorer(data, title='Explorer')
    expl.set_wave_data(waveforms, 'Time', ['Sine', 'Gauss'])
    # explore data:
    expl.set_colors()
    expl.show(False)
    # test _on_select():
    class eclick: pass
    eclick.dblclick = False
    eclick.xdata = 1 
    eclick.ydata = 0 
    eclick.inaxes = expl.scatter_ax[0]
    eclick.key = []
    class erelease: pass
    erelease.xdata = 3
    erelease.ydata = 1
    erelease.inaxes = None
    erelease.key = []
    expl._on_select(eclick, erelease)

    erelease.xdata = 1.01
    erelease.ydata = 0.01
    expl._on_select(eclick, erelease)
    
    # test _on_pick():
    #class ev: pass
    #ev.artist = expl.wave_ax[0].lines[0]
    #expl._on_pick(ev)

    
def test_main():
    me.main('')
