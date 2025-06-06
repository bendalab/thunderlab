import pytest
import os
import sys
import numpy as np
from thunderlab.configfile import ConfigFile
import thunderlab.tabledata as td

def setup_table(nanvalue=True):
    df = td.TableData()
    df.append(["data", "specimen", "size"], "m", "%6.2f",
              value=[2.34, 56.7, 8.9])
    df.append("full weight", "kg", "%.0f", value=122.8)
    df.append_section("all measures")
    df.append("speed", "m/s", "%.3g", value=98.7)
    df.append("median jitter", "mm", "%.1f", value=23)
    df.append("size", "g", "%.2e", value=1.234)
    df.set_descriptions({'size': 'The total length of each snake.',
                         'full weight': 'Weight of each snake',
                         'speed': 'Maximum speed the snake can climb a tree.',
                         'median jitter': 'The jitter around a given path the snake should follow.',
                         'all measures>size': 'Weight of mouse the snake has eaten before.'})
    if nanvalue:
        df.add(float('NaN'), 1)  # single value
    else:
        df.add(27.56, 1)  # single value
    df.add((0.543, 45, 1.235e2)) # remaining row
    df.add((43.21, 6789.1, 3405, 1.235e-4), 1) # next row
    a = 0.5*np.arange(1, 6)*np.random.randn(5, 5) + 10.0 + np.arange(5)
    df.add(a.T, 0) # rest of table
    df[3:6,'weight'] = [11.0]*3
    return df

def test_write():
    df = setup_table()
    for unit_style in [None, 'none', 'row', 'header']:
        for column_numbers in [None, 'index', 'num', 'aa', 'AA']:
            for tf in td.TableData.formats + ['x']:
                df.write(table_format=tf, column_numbers=column_numbers)
    df.write_file_stream(sys.stdout, 'file')
    fn = df.write_file_stream('table', 'file')
    fn.unlink()
    fn = fn.with_name(fn.stem + '-description.md')
    fn.unlink()

def test_properties():
    df = setup_table()
    assert len(df) == 8, 'len() failed %d' % len(df)
    assert df.columns() == 5, 'columns() failed %d' % df.columns()
    assert df.rows() == 8, 'rows() failed %d' % df.rows()
    assert df.shape == (8, 5), 'shape failed %d %d' % df.shape
    assert df.ndim == 2, 'ndim failed %d' % df.ndim
    assert df.size == 8*5, 'size failed %d' % df.ndim

def test_columns():
    df = setup_table(False)
    units = ['m', 'kg', 'm/s', 'mm', 'g']
    formats = ['%6.2f', '%.0f', '%.3g', '%.1f', '%.2e']
    sec1 = [('specimen', 0), ('specimen', 0), ('all measures', 2), ('all measures', 2), ('all measures', 2)]
    for c, k in enumerate(df.keys()):
        assert c == df.index(k), 'index %s is not %d' % (k, c)
        assert k in df, 'key %s not found in table' % k
        assert not ('xx'+k) in df, 'key %s found in table' % k
        assert not (k+'xx') in df, 'key %s found in table' % k
    for c, k in enumerate(df):
        assert c == df.index(k), 'index %s is not %d' % (k, c)
        assert k in df, 'key %s not found in table' % k
    for c in range(df.columns()):
        k = df.column_spec(c)
        assert c == df.index(k), 'index %s is not %d' % (k, c)
        assert units[c] == df.unit(c), 'unit of column %d is not %s' % (c, units[c])
        assert formats[c] == df.format(c), 'format of column %d is not %s' % (c, formats[c])
    for c, (s1, c1) in enumerate(sec1):
        ds, cs = df.section(c, 1)
        assert s1 == ds, 'section level 1 name of column %d is not %s' % (c, s1)
        assert c1 == cs, 'section level 1 index of column %d is not %d' % (c, c1)
        ds, cs = df.section(c, 2)
        s2 = 'data'
        c2 = 0
        assert s2 == ds, 'section level 2 name of column %d is not %s' % (c, s2)
        assert c2 == cs, 'section level 2 index of column %d is not %d' % (c, c2)
    for c in range(df.columns()):
        l = 'aaa%d' % c
        df.set_label(c, l)
        assert df.label(c) == l, 'label of column %d is not %s' % (c, l)
        df.set_unit(c, 'km/h')
        assert df.unit(c) == 'km/h', 'unit of column %d is not km/h' % c
        df.set_format(c, '%g')
        assert df.format(c) == '%g', 'format of column %d is not %%g' % c
    df.set_units(list(reversed(units)))
    df.set_formats(list(reversed(formats)))
    for c, (u, f) in enumerate(zip(reversed(units), reversed(formats))):
        assert df.unit(c) == u, 'unit of column %d is not %s' % (c, u)
        assert df.format(c) == f, 'format of column %d is not %s' % (c, f)
    for dc, vc in zip(df.data, df.values()):
        assert np.all(dc == vc), 'data and value columns differ'
    for k, v in df.items():
        assert np.all(df.column(k).array()[:,0] == v), 'data and value for column %s differ' % k

def test_removal():
    for i in range(20):
        df = setup_table()
        for k in reversed(range(df.columns())):
            c = np.random.randint(df.columns())
            df.remove(c)
            assert df.columns() == k, 'after removal of column len should be %d' % k
    for i in range(20):
        df = setup_table()
        for k in reversed(range(df.columns())):
            c = np.random.randint(df.columns())
            del df[:,c]
            assert df.columns() == k, 'after removal of column len should be %d' % k
    for i in range(20):
        df = setup_table()
        for k in reversed(range(df.rows())):
            r = np.random.randint(df.rows())
            del df[r,:]
            assert df.rows() == k, 'after removal of row len should be %d' % k
    for i in range(20):
        df = setup_table()
        n = df.rows()
        r = np.unique(np.random.randint(0, df.rows(), n//2))
        del df[r,:]
        assert df.rows() == n - len(r), 'after removal of row len should be %d' % k

def test_insertion():
    for i in range(20):
        df = setup_table()
        nc = df.columns()
        for k in range(1,10):
            c = np.random.randint(df.columns())
            df.insert(c, 'aaa', 'm', '%g', 'some description')
            assert df.columns() == nc+k, 'after insertion of column len should be %d' % (nc+k)
            assert df.label(c) == 'aaa', 'label of inserted column should be %s' % 'aaa'
            assert df.unit(c) == 'm', 'unit of inserted column should be %s' % 'm'
            assert df.format(c) == '%g', 'format of inserted column should be %s' % '%g'
            assert df.description(c) == 'some description', 'description of inserted column should be %s' % 'some description'

def test_key_value():
    df = setup_table()
    df.key_value(1, 'xxx')
    df.key_value(1, 'speed')
    df.dicts()
    df.dicts(False)
    df.table_header()

def test_fill():
    df = setup_table()
    df.clear_data()
    for c in range(df.columns()):
        df.append_data_column(np.random.randn(2+c), c)
    for c in range(df.columns()):
        assert len(df.data[c]) == 2+c, 'column should have %d data elements' % (2+c)
    df.fill_data()
    for c in range(df.columns()):
        assert len(df.data[c]) == df.rows(), 'column should have %d data elements' % df.rows()

def test_statistics():
    df = setup_table()
    st = df.statistics()
    assert st.rows() == 8, 'statistics should have 8 rows'
    assert st.columns() == df.columns()+1, 'statistics should have %d columns' % (df.columns()+1)
    
def test_sort():
    df = setup_table(False)
    for c in range(df.columns()):
        df.sort(c)
        assert np.all(np.diff(df.data[c])>=0), 'values in columns %d are not sorted' % c
        df.sort(c, reverse=True)
        assert np.all(np.diff(df.data[c])<=0), 'values in columns %d are not sorted' % c
    
def test_write_load():
    df = setup_table()
    for unit_style in [None, 'none', 'row', 'header']:
        for column_numbers in [None, 'none', 'index', 'num', 'aa', 'AA']:
            for delimiter in [None, ';', '| ', '\t']:
                for align_columns in [None, True, False]:
                    for sections in [None, 0, 1, 2]:
                        for tf in td.TableData.formats[:-1]:
                            orgfilename = 'tabletest.' + td.TableData.extensions[tf]
                            df.write(orgfilename, table_format=tf, column_numbers=column_numbers,
                                     unit_style=unit_style, delimiter=delimiter,
                                     align_columns=align_columns, sections=sections)
                            sf = td.TableData(orgfilename)
                            filename = 'tabletest-read.' + td.TableData.extensions[tf]
                            sf.write(filename, table_format=tf, column_numbers=column_numbers,
                                     unit_style=unit_style, delimiter=delimiter,
                                     align_columns=align_columns, sections=sections)
                            with open(orgfilename, 'r') as f1, open(filename, 'r') as f2:
                                for k, (line1, line2) in enumerate(zip(f1, f2)):
                                    if line1 != line2:
                                        print('%s: %s' % (tf, td.TableData.descriptions[tf]))
                                        print('files differ!')
                                        print('original table:')
                                        df.write(table_format=tf, column_numbers=column_numbers,
                                                 unit_style=unit_style, delimiter=delimiter,
                                                 align_columns=align_columns, sections=sections)
                                        print('')
                                        print('read in table:')
                                        sf.write(table_format=tf, column_numbers=column_numbers,
                                                 unit_style=unit_style, delimiter=delimiter,
                                                 align_columns=align_columns, sections=sections)
                                        print('')
                                        print('line %2d "%s" from original table does not match\n        "%s" from read in table.' % (k+1, line1.rstrip('\n'), line2.rstrip('\n')))
                                    assert line1 == line2, 'files differ at line %d:\n%s\n%s' % (k, line1, line2)
                            os.remove(orgfilename)
                            os.remove(filename)
    os.remove('tabletest-description.md')
    os.remove('tabletest-description.tex')

    
def test_read_access():
    df = setup_table()
    df.clear_data()
    data = np.random.randn(10, df.columns())
    df.add(data)
    n = 1000
    # reading values by index:
    for c, r in zip(np.random.randint(0, df.columns(), n), np.random.randint(0, df.rows(), n)):
        assert df[r,c] == data[r,c], 'element access by index failed'
    # reading values by column name:
    for c, r in zip(np.random.randint(0, df.columns(), n), np.random.randint(0, df.rows(), n)):
        assert df[r,df.keys()[c]] == data[r,c], 'element access by column name failed'
    # reading row slices:
    for c in range(df.columns()):
        for r in np.random.randint(0, df.rows(), (n,2)):
            r0, r1 = np.sort(r)
            assert np.array_equal(df[r0:r1,c], data[r0:r1,c]), 'slicing of rows failed'
    # reading column slices:
    for r in range(df.rows()):
        for c in np.random.randint(0, df.columns(), (n,2)):
            c0, c1 = np.sort(c)
            if c1-c0 < 2:
                continue
            assert np.array_equal(df[r,c0:c1].array(0), data[r,c0:c1]), 'slicing of columns failed'
    # reading row and column slices:
    for c, r in zip(np.random.randint(0, df.columns(), (n,2)), np.random.randint(0, df.rows(), (n,2))):
        r0, r1 = np.sort(r)
        c0, c1 = np.sort(c)
        if c1-c0 < 2:
            continue
        assert np.array_equal(df[r0:r1,c0:c1].array(), data[r0:r1,c0:c1]), 'slicing of rows and columns failed'
    # reading full column slices:
    for c in range(df.columns()):
        assert np.array_equal(df(c), data[:,c]), 'slicing of full column failed'
        assert np.array_equal(df[:,c], data[:,c]), 'slicing of full column failed'
        assert np.array_equal(df.column(c)[:,0], data[:,c]), 'slicing of full column failed'
    # reading full column slices by name:
    for k in df:
        assert np.array_equal(df(k), df[:,k]), f'slicing of full column failed'
        assert np.array_equal(df[k], df[:,k]), 'slicing of full column failed'
    # reading full row slices:
    for r in range(df.rows()):
        assert np.array_equal(df[r,:].array(0), data[r,:]), 'slicing of full row failed'
        assert np.array_equal(df.row(r)[0,:].array(0), data[r,:]), 'slicing of full row failed'
        d = df.row_dict(r)
        for i, k in enumerate(d):
            assert d[k] == data[r, i], 'row_dict() failed'
        d = df.row_list(r)
        for i, v in enumerate(d):
            assert v == data[r, i], 'row_list() failed'
    for r, d in enumerate(df.row_data()):
        for i, v in enumerate(d):
            assert v == data[r, i], 'row_data() failed'


def test_write_access():
    df = setup_table()
    df.clear_data()
    data = np.random.randn(10, df.columns())
    df.add(data)
    n = 1000
    # writing and reading values by index:
    for c, r in zip(np.random.randint(0, df.columns(), n), np.random.randint(0, df.rows(), n)):
        v = np.random.randn()
        df[r,c] = v
        assert df[r,c] == v, 'set item by index failed'
    # writing and reading row slices:
    for c in range(df.columns()):
        for r in np.random.randint(0, df.rows(), (n,2)):
            r0, r1 = np.sort(r)
            v = np.random.randn(r1-r0)
            df[r0:r1,c] = v
            assert np.array_equal(df[r0:r1,c], v), 'slicing of rows failed'
    # writing and reading column slices:
    for r in range(df.rows()):
        for c in np.random.randint(0, df.columns(), (n,2)):
            c0, c1 = np.sort(c)
            if c1-c0 < 2:
                continue
            v = np.random.randn(c1-c0)
            df[r,c0:c1] = v
            assert np.array_equal(df[r,c0:c1].array(0), v), 'slicing of columns failed'
    # writing and reading row and column slices:
    for c, r in zip(np.random.randint(0, df.columns(), (n,2)), np.random.randint(0, df.rows(), (n,2))):
        r0, r1 = np.sort(r)
        c0, c1 = np.sort(c)
        if c1-c0 < 2:
            continue
        v = np.random.randn(r1-r0, c1-c0)
        df[r0:r1,c0:c1] = v
        assert np.array_equal(df[r0:r1,c0:c1].array(), v), 'slicing of rows and columns failed'

def test_hide_show():
    filename = 'tabletest.dat'
    df = setup_table()
    for c in range(df.columns()):
        df.hide(c)
        df.write(filename)
        sf = td.TableData(filename)
        assert sf.columns() == df.columns()-1, 'wrong number of columns written'
        df.show(c)
        df.write(filename)
        sf = td.TableData(filename)
        assert sf.columns() == df.columns(), 'wrong number of columns written'
    for c in range(df.columns()):
        df = setup_table()
        df.data[c] = []
        df.hide_empty_columns()
        df.write(filename)
        sf = td.TableData(filename)
        assert sf.columns() == df.columns()-1, 'wrong number of columns written'
        df.show(c)
        df.write(filename)
        sf = td.TableData(filename)
        assert sf.columns() == df.columns(), 'wrong number of columns written'
        df.data[c] = [float('nan')] * df.rows()
        df.hide_empty_columns()
        df.write(filename)
        sf = td.TableData(filename)
        assert sf.columns() == df.columns()-1, 'wrong number of columns written'
        df.show(c)
        df.write(filename)
        sf = td.TableData(filename)
        assert sf.columns() == df.columns(), 'wrong number of columns written'
    df = setup_table()
    df.hide_all()
    for c in range(df.columns()):
        df.show(c)
        df.write(filename)
        sf = td.TableData(filename)
        assert sf.columns() == 1, 'wrong number of columns written'
        df.hide(c)
    os.remove(filename)
    os.remove('tabletest-description.md')

    
def test_write_descriptions():
    filename = 'tabletest.dat'
    df = setup_table()
    for tf in ['md', 'tex', 'html']:
        for sections in [None, 1000]:
            for section_headings in [None, 0, 1]:
                df.write_descriptions(table_format=tf,
                                      sections=sections,
                                      section_headings=section_headings,
                                      maxc=30)

    
def test_config():
    cfg = ConfigFile()
    td.add_write_table_config(cfg)
    assert type(td.write_table_args(cfg)) is dict, 'write_table_args()'

    
def test_main():
    td.main()
    
