import pytest
import numpy as np
import thunderlab.datawriter as dw

from pathlib import Path


def test_formats():
    dw.available_formats()
    for fmt, lib, formats_func in dw.data_formats_funcs:
        if lib:
            dw.data_modules[lib] = False
            f = formats_func()
            assert len(f) == 0, f'{lib} format did not return empty list'
            dw.data_modules[lib] = True


def test_write():
    with pytest.raises(ValueError):
        dw.write_data('', np.zeros((1000, 2)), 48000)
    with pytest.raises(ValueError):
        dw.write_data('test', np.zeros((1000, 2)), 48000)
    with pytest.raises(IOError):
        dw.write_data('test', np.zeros((1000, 2)), 48000, format='xxx')
    dw.data_modules['pkl'] = False
    with pytest.raises(IOError):
        dw.write_data('test', np.zeros((1000, 2)), 48000, format='xxx')
    dw.data_modules['pkl'] = True
    for fmt, lib, formats_func in dw.data_formats_funcs:
        writer_func = dw.data_writer_funcs[fmt]
        fn = writer_func('test', np.zeros((1000, 2)), 48000)
        if fn.is_dir():
            for f in fn.glob('*'):
                f.unlink()
            fn.rmdir()
        else:
            fn.unlink()
        if lib:
            dw.data_modules[lib] = False
            with pytest.raises(ImportError):
                writer_func('test.dat', np.zeros((1000, 2)), 48000)
            dw.data_modules[lib] = True

    
def test_extensions():
    f = dw.format_from_extension(None)
    assert f is None
    f = dw.format_from_extension('')
    assert f is None
    f = dw.format_from_extension('test')
    assert f is None
    f = dw.format_from_extension('test.')
    assert f is None
    f = dw.format_from_extension('test.pkl')
    assert f == 'PKL'

    
def test_main():
    dw.main('-c', '2', 'test.npz')
    Path('test.npz').unlink()
    
