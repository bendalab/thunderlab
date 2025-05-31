"""Tables with hierarchical headers and units

## Classes

- `class TableData`: tables with hierarchical header including units
  and column-specific formats. Kind of similar to a pandas data frame,
  but without index column and with intuitive numpy-style indexing and
  nicely formatted output to csv, markdown, html, and latex.


## Helper functions

- `write()`: shortcut for constructing and writing a TableData.
- `latex_unit()`: translate unit string into SIunit LaTeX code.
- `index2aa()`: convert an integer into an alphabetical representation.
- `aa2index()`: convert an alphabetical representation to an index.


## Configuration

- `add_write_table_config()`: add parameter specifying how to write a table to a file as a new section to a configuration.
- `write_table_args()`: translates a configuration to the respective parameter names for writing a table to a file.

"""

import sys
import os
import re
import math as m
import numpy as np
from pathlib import Path
from itertools import product
from io import StringIO
try:
    import pandas as pd
    has_pandas = True
except ImportError:
    has_pandas = False


__pdoc__ = {}
__pdoc__['TableData.__contains__'] = True
__pdoc__['TableData.__len__'] = True
__pdoc__['TableData.__iter__'] = True
__pdoc__['TableData.__next__'] = True
__pdoc__['TableData.__setupkey__'] = True
__pdoc__['TableData.__call__'] = True
__pdoc__['TableData.__getitem__'] = True
__pdoc__['TableData.__setitem__'] = True
__pdoc__['TableData.__delitem__'] = True
__pdoc__['TableData.__str__'] = True


default_missing_str = '-'
"""Default string indicating nan data elements when outputting data."""

default_missing_inputs = ['na', 'NA', 'nan', 'NAN', '-']
"""Default strings that are translated to nan when loading table data."""


class TableData(object):
    """Table with numpy-style indexing and hierarchical header including units and formats.
    
    Parameters
    ----------
    data: str, stream, ndarray
        - a filename: load table from file with name `data`.
        - a stream/file handle: load table from that stream.
        - 1-D or 2-D ndarray of data: the data of the table.
          Requires als a specified `header`.
        - pandas data frame.
    header: TableData, dict, list of str, list of list of str
        Header labels for each column.
        See `set_labels()' for details.
    units: None, TableData, dict, list of str, str
        Optional unit strings for each column.
        See `set_units()' for details.
    formats: None, TableData, dict, list of str, str
        Optional format strings for each column.
        See `set_formats()' for details.
    descriptions: None, TableData, dict, list of str, str
        Optional description strings for each column.
        See `set_descriptions()' for details.
    missing: list of str
        Missing data are indicated by one of these strings.
    sep: str or None
        If `data` is a file, force `sep` as column separator.
    stop: str or None
        If a line matches `stop`, stop reading the file.  `stop`
        can be an empty string to stop reading at the first empty
        line.

    Manipulate table header
    -----------------------

    Each column of the table has a label (the name of the column), a
    unit, and a format specifier. Sections group columns into a hierarchy.

    - `__init__()`: initialize a TableData from data or a file.
    - `append()`: append column to the table.
    - `insert()`: insert a table column at a given position.
    - `remove()`: remove columns from the table.
    - `section()`: the section name of a specified column.
    - `set_section()`: set a section name.
    - `append_section()`: add sections to the table header.
    - `insert_section()`: insert a section at a given position of the table header.
    - `label()`: the name of a column.
    - `set_label()`: set the name of a column.
    - `set_labels()`: set the labels of some columns.
    - `unit()`: the unit of a column.
    - `set_unit()`: set the unit of a column.
    - `set_units()`: set the units of some columns.
    - `format()`: the format string of the column.
    - `set_format()`: set the format string of a column.
    - `set_formats()`: set the format strings of some columns.
    - `description()`: the description of a column.
    - `set_description()`: set the description of a column.
    - `set_descriptions()`: set the descriptions of some columns.

    For example:
    ```
    tf = TableData('data.csv')
    ```
    loads a table directly from a file. See `load()` for details.
    ```
    tf = TableData(np.random.randn(4,3), header=['aaa', 'bbb', 'ccc'], units=['m', 's', 'g'], formats='%.2f')    
    ```
    results in
    ``` plain
    aaa    bbb    ccc
    m      s      g    
     1.45   0.01   0.16
    -0.74  -0.58  -1.34
    -2.06   0.08   1.47
    -0.43   0.60   1.38
    ```

    A more elaborate way to construct a table is:
    ```
    df = TableData()
    # first column with section names and 3 data values:
    df.append(["data", "specimen", "size"], "m", "%6.2f",
              [2.34, 56.7, 8.9])
    # next columns with single data values:
    df.append("full weight", "kg", "%.0f", 122.8)
    df.append_section("all measures")
    df.append("speed", "m/s", "%.3g", 98.7)
    df.append("median jitter", "mm", "%.1f", 23)
    df.append("size", "g", "%.2e", 1.234)
    # add a missing value to the second column:
    df.add(np.nan, 1)
    # fill up the remaining columns of the row:
    df.add((0.543, 45, 1.235e2))
    # add data to the next row starting at the second column:
    df.add([43.21, 6789.1, 3405, 1.235e-4], 1) # next row
    ```
    results in
    ``` plain
    data
    specimen             all measures
    size    full weight  speed     median jitter  size
    m       kg           m/s       mm             g       
      2.34          123      98.7           23.0  1.23e+00
     56.70            -     0.543           45.0  1.24e+02
      8.90           43  6.79e+03         3405.0  1.23e-04
    ```
    
    Table columns
    -------------

    Columns can be specified by an index or by the name of a column. In
    table headers with sections the colum can be specified by the
    section names and the column name separated by '>'.
    
    - `index()`: the column index of a column specifier.
    - `__contains__()`: check for existence of a column.
    - `find_col()`: find the start and end index of a column specification.
    - `column_spec()`: full specification of a column with all its section names.
    - `column_head()`: the name, unit, and format of a column.
    - `table_header()`: the header of the table without content.

    For example:
    ```
    df.index('all measures>size)   # returns 4
    'speed' in df                       # is True
    ```

    Iterating over columns
    ----------------------

    A table behaves like an ordered dictionary with column names as
    keys and the data of each column as values.
    Iterating over a table goes over columns.
    Note, however, that the len() of a table is the number of rows,
    not the number of columns!
    
    - `keys()`: list of unique column keys for all available columns.
    - `values()`: list of column data corresponding to keys().
    - `items()`: generator over column names and the corresponding data.
    - `__iter__()`: initialize iteration over data columns.
    - `__next__()`: return unique column key of next column.
    - `data`: the table data as a list over columns each containing a list of data elements.

    For example:
    ```
    print('column specifications:')
    for c in range(df.columns()):
        print(df.column_spec(c))
    print('iterating over column specifications:')
    for c, k in enumerate(df):
        print(f'{c}: {k}')
    print('keys():')
    for c, k in enumerate(df.keys()):
        print(f'{c}: {k}')
    print('values():')
    for c, v in enumerate(df.values()):
        print(v)
    print('iterating over the table:')
    for v in df:
        print(v)
    ```
    results in
    ``` plain
    column specifications:
    data>specimen>size
    data>specimen>full weight
    data>all measures>speed
    data>all measures>median jitter
    data>all measures>size
    iterating over column specifications:
    0: data>specimen>size
    1: data>specimen>full weight
    2: data>all measures>speed
    3: data>all measures>median jitter
    4: data>all measures>size
    keys():
    0: data>specimen>size
    1: data>specimen>full weight
    2: data>all measures>speed
    3: data>all measures>median jitter
    4: data>all measures>size
    values():
    [2.34, 56.7, 8.9]
    [122.8, nan, 43.21]
    [98.7, 0.543, 6789.1]
    [23, 45, 3405]
    [1.234, 123.5, 0.0001235]
    iterating over the table:
    [2.34, 56.7, 8.9]
    [122.8, nan, 43.21]
    [98.7, 0.543, 6789.1]
    [23, 45, 3405]
    [1.234, 123.5, 0.0001235]
    ```

    Accessing data
    --------------

    In contrast to the iterator functions the [] operator treats the
    table as a 2D-array where the first index indicates the row and
    the second index the column.

    Rows are indexed by integer row numbers or boolean arrays.
    Columns are also indexed by integer column numbers, but in
    addition can be index by their names.

    A single index selects rows, unless it is specified by
    strings. Since strings can only specify column names, this selects
    whole columns.

    Like a numpy array the table can be sliced, and logical indexing can
    be used to select specific parts of the table.
    
    As for any function, columns can be specified as indices or strings.
    
    - `rows()`: the number of rows.
    - `columns()`: the number of columns.
    - `__len__()`: the number of rows.
    - `ndim`: always 2.
    - `size`: number of elements (sum of length of all data columns), can be smaller than `columns()*rows()`.
    - `shape`: number of rows and columns.
    - `row()`: a single row of the table as TableData.
    - `row_list()`: a single row of the table as list.
    - `row_data()`: a generator for iterating over rows of the table.
    - `row_dict()`: a single row of the table as dictionary.
    - `col()`: a single column of the table as TableData.
    - `__call__()`: a single column of the table as ndarray.
    - `__getitem__()`: data elements specified by slice.
    - `__setitem__()`: assign values to data elements specified by slice.
    - `__delitem__()`: delete data elements or whole columns or rows.
    - `array()`: the table data as a ndarray.
    - `data_frame()`: the table data as a pandas DataFrame.
    - `dicts()`: the table as a list of dictionaries.
    - `dict()`: the table as a dictionary.
    - `add()`: add data elements row-wise.
    - `append_data_column()`: append data elements to a column.
    - `set_column()`: set the column where to add data.
    - `fill_data()`: fill up all columns with missing data.
    - `clear_data()`: clear content of the table but keep header.
    - `clear()`: clear the table of any content and header information.
    - `key_value()`: a data element returned as a key-value pair.
    - `aggregate()`: apply functions to columns.
    - `groupby()`: iterate through unique values of columns.
    
    - `sort()`: sort the table rows in place.
    - `statistics()`: descriptive statistics of each column.

    For example:
    ```
    # single column:    
    df('size')      # data of 'size' column as ndarray
    df['size']      # data of 'size' column as ndarray
    df[:, 'size']   # data of 'size' column as ndarray
    df.col('size')  # table with the single column 'size'

    # single row:    
    df[2, :]   # table with data of only the third row
    df.row(2)  # table with data of only the third row

    # slices:
    df[2:5,['size','jitter']]          # sub-table
    df[2:5,['size','jitter']].array()  # ndarray with data only

    # logical indexing:
    df[df['speed'] > 100.0, 'size'] = 0.0 # set size to 0 if speed is > 100

    # delete:
    del df[3:6, 'weight']  # delete rows 3-6 from column 'weight'
    del df[3:5, :]         # delete rows 3-5 completeley
    del df[:, 'speed']     # remove column 'speed' from table
    del df['speed']        # remove column 'speed' from table
    df.remove('weight')    # remove column 'weigth' from table

    # sort and statistics:
    df.sort(['weight', 'jitter'])
    df.statistics()
    ```
    statistics() returns a table with standard descriptive statistics:
    ``` plain
    statistics  data
    -           specimen             all measures
    -           size    full weight  speed     median jitter  size
    -           m       kg           m/s       mm             g       
    mean         22.65           83   2.3e+03         1157.7  4.16e+01
    std          24.23           40  3.18e+03         1589.1  5.79e+01
    min           2.34           43     0.543           23.0  1.23e-04
    quartile1     5.62           83      49.6           34.0  6.17e-01
    median        8.90          123      98.7           45.0  1.23e+00
    quartile3    32.80            -  3.44e+03         1725.0  6.24e+01
    max          56.70          123  6.79e+03         3405.0  1.24e+02
    count         3.00            2         3            3.0  3.00e+00
    ```

    Write and load tables
    ---------------------

    Table data can be written to a variety of text-based formats
    including comma separated values, latex and html files.  Which
    columns are written can be controlled by the hide() and show()
    functions. TableData can be loaded from all the written file formats
    (except html), also directly via the constructor.
    
    - `hide()`: hide a column or a range of columns.
    - `hide_all()`: hide all columns.
    - `hide_empty_columns()`: hide all columns that do not contain data.
    - `show()`: show a column or a range of columns.
    - `write()`: write table to a file or stream.
    - `write_file_stream()`: write table to file or stream and return appropriate file name.
    - `__str__()`: write table to a string.
    - `write_descriptions()`: write column descriptions of the table to a file or stream.
    - `load()`: load table from file or stream.
    - `formats`: list of supported file formats for writing.
    - `descriptions`: dictionary with descriptions of the supported file formats.
    - `extensions`: dictionary with default filename extensions for each of the file formats.
    - `ext_formats`: dictionary mapping filename extensions to file formats.

    See documentation of the `write()` function for examples of the supported file formats.

    """
    
    formats = ['dat', 'ascii', 'csv', 'rtai', 'md', 'tex', 'html']
    """list of strings: Supported output formats."""
    
    descriptions = {'dat': 'data text file', 'ascii': 'ascii-art table',
                    'csv': 'comma separated values', 'rtai': 'rtai-style table',
                    'md': 'markdown', 'tex': 'latex tabular',
                    'html': 'html markup'}
    """dict: Decription of output formats corresponding to `formats`."""
    
    extensions = {'dat': 'dat', 'ascii': 'txt', 'csv': 'csv', 'rtai': 'dat',
                  'md': 'md', 'tex': 'tex', 'html': 'html'}
    """dict: Default file extensions for the output `formats`. """
    
    ext_formats = {'dat': 'dat', 'DAT': 'dat', 'txt': 'dat', 'TXT': 'dat',
                   'csv': 'csv', 'CSV': 'csv', 'md': 'md', 'MD': 'md',
                   'tex': 'tex', 'TEX': 'tex', 'html': 'html', 'HTML': 'html'}
    """dict: Mapping of file extensions to the output formats."""

    stdev_labels = ['sd', 'std', 's.d.', 'stdev', 'error']
    """list: column labels recognized as standard deviations."""

    def __init__(self, data=None, header=None, units=None, formats=None,
                 descriptions=None, missing=default_missing_inputs,
                 sep=None, stop=None):
        self.clear()
        if header is not None and len(header) > 0:
            for h in header:
                self.append(h)            
        if data is not None:
            if isinstance(data, TableData):
                self.ndim = data.ndim
                self.size = data.size
                self.shape = data.shape
                self.nsecs = data.nsecs
                self.setcol = data.setcol
                self.addcol = data.addcol
                for c in range(data.columns()):
                    self.header.append(list(data.header[c]))
                    self.units.append(data.units[c])
                    self.formats.append(data.formats[c])
                    self.descriptions.append(data.descriptions[c])
                    self.hidden.append(data.hidden[c])
                    self.data.append(list(data.data[c]))
                self.set_labels(header)
                self.set_units(units)
                self.set_formats(formats)
                self.set_descriptions(descriptions)
            elif has_pandas and isinstance(data, pd.DataFrame):
                for c, key in enumerate(data.keys()):
                    new_key = key
                    new_unit = ''
                    if '/' in key:
                        p = key.split('/')
                        new_key = p[0].strip()
                        new_unit = '/'.join(p[1:])
                    formats = '%s' if isinstance(values[0], str) else '%g'
                    values = data[key].tolist()
                    self.append(new_key, new_unit, formats, value=values)
                self.set_labels(header)
                self.set_units(units)
                self.set_formats(formats)
                self.set_descriptions(descriptions)
            elif isinstance(data, (list, tuple, np.ndarray)) and not \
                 (isinstance(data, np.ndarray) and len(data.shape) == 0):
                if len(data) > 0 and \
                   isinstance(data[0], (list, tuple, np.ndarray)) and not \
                   (isinstance(data[0], np.ndarray) and \
                    len(data[0].shape) == 0):
                    # 2D list, rows first:
                    for row in data:
                        for c, val in enumerate(row):
                            self.data[c].append(val)
                elif len(data) > 0 and isinstance(data[0], dict):
                    # list of dictionaries:
                    for d in data:
                        self._add_dict(d, True)
                    self.fill_data()
                    self.set_labels(header)
                    self.set_units(units)
                    self.set_formats(formats)
                    self.set_descriptions(descriptions)
                else:
                    # 1D list:
                    for c, val in enumerate(data):
                        self.data[c].append(val)
            elif isinstance(data, (dict)):
                self._add_dict(data, True)
                self.fill_data()
                self.set_labels(header)
                self.set_units(units)
                self.set_formats(formats)
                self.set_descriptions(descriptions)
            else:
                self.load(data, missing, sep, stop)
                self.set_labels(header)
                self.set_units(units)
                self.set_formats(formats)
                self.set_descriptions(descriptions)
            # fill in missing units and formats:
            for k in range(len(self.header)):
                if self.units[k] is None:
                    self.units[k] = ''
                if self.formats[k] is None:
                    self.formats[k] = '%g'
                if self.descriptions[k] is None:
                    self.descriptions[k] = ''

    def __recompute_shape(self):
        self.size = sum(map(len, self.data))
        self.shape = (self.rows(), self.columns())
        
    def append(self, label, unit=None, formats=None, description=None,
               value=None, fac=None, key=None):
        """Append column to the table.

        Parameters
        ----------
        label: str or list of str
            Optional section titles and the name of the column.
        unit: str or None
            The unit of the column contents.
        formats: str or None
            The C-style format string used for printing out the column content, e.g.
            '%g', '%.2f', '%s', etc.
            If None, the format is set to '%g'.
        description: str or None
            The description of the column contents.
        value: None, float, int, str, etc. or list thereof, or list of dict
            If not None, data for the column.
            If list of dictionaries, extract from each dictionary in the list
            the value specified by `key`. If `key` is `None` use `label` as
            the key.
        fac: float
            If not None, multiply the data values by this number.
        key: None or key of a dictionary
            If not None and `value` is a list of dictionaries,
            extract from each dictionary in the list the value specified
            by `key` and assign the resulting list as data to the column.

        Returns
        -------
        self: TableData
            This TableData
        """
        if self.addcol >= len(self.data):
            if isinstance(label, (list, tuple, np.ndarray)):
                label = list(reversed(label))
                # number of sections larger than what we have so far:
                n = max(0, len(label) - 1 - self.nsecs)
                # find matching sections:
                found = False
                for s in range(1, len(label)):
                    for c in range(len(self.header) - 1, -1, -1):
                        if len(self.header[c]) > s - n:
                            if s - n >= 0 and \
                               self.header[c][s - n] == label[s]:
                                # remove matching sections:
                                label = label[:s]
                                found = True
                            break
                    if found:
                        break
                # add label and unique sections:
                self.header.append(label)
                label = label[0]
                if n > 0:
                    # lift previous header label and sections:
                    for c in range(len(self.header) - 1):
                        self.header[c] = ['-']*n + self.header[c]
            else:
                self.header.append([label])
            self.units.append(unit or '')
            self.formats.append(formats or '%g')
            self.descriptions.append(description or '')
            self.hidden.append(False)
            self.data.append([])
            self.nsecs = max(map(len, self.header)) - 1
        else:
            if isinstance(label, (list, tuple, np.ndarray)):
                self.header[self.addcol] = list(reversed(label)) + self.header[self.addcol]
                label = label[-1]
            else:
                self.header[self.addcol] = [label] + self.header[self.addcol]
            self.units[self.addcol] = unit or ''
            self.formats[self.addcol] = formats or '%g'
            self.descriptions[self.addcol] = description or ''
            if self.nsecs < len(self.header[self.addcol]) - 1:
                self.nsecs = len(self.header[self.addcol]) - 1
        if not key:
            key = label
        if value is not None:
            if isinstance(value, (list, tuple, np.ndarray)):
                if key and len(value) > 0 and isinstance(value[0], dict):
                    value = [d[key] if key in d else float('nan') for d in value]
                self.data[-1].extend(value)
            else:
                self.data[-1].append(value)
        if fac:
            for k in range(len(self.data[-1])):
                self.data[-1][k] *= fac
        self.addcol = len(self.data)
        self.__recompute_shape()
        return self
        
    def insert(self, column, label, unit=None, formats=None, description=None,
               value=None, fac=None, key=None):
        """Insert a table column at a given position.

        .. WARNING::
           If no `value` is given, the inserted column is an empty list.

        Parameters
        ----------
        columns int or str
            Column before which to insert the new column.
            Column can be specified by index or name,
            see `index()` for details.
        label: str or list of str
            Optional section titles and the name of the column.
        unit: str or None
            The unit of the column contents.
        formats: str or None
            The C-style format string used for printing out the column content, e.g.
            '%g', '%.2f', '%s', etc.
            If None, the format is set to '%g'.
        description: str or None
            The description of the column contents.
        value: None, float, int, str, etc. or list thereof, or list of dict
            If not None, data for the column.
            If list of dictionaries, extract from each dictionary in the list
            the value specified by `key`. If `key` is `None` use `label` as
            the key.
        fac: float
            If not None, multiply the data values by this number.
        key: None or key of a dictionary
            If not None and `value` is a list of dictionaries,
            extract from each dictionary in the list the value specified
            by `key` and assign the resulting list as data to the column.

        Returns
        -------
        index: int
            The index of the inserted column.
            
        Raises
        ------
        self: TableData
            This TableData
        """
        col = self.index(column)
        if col is None:
            raise IndexError(f'Cannot insert before non-existing column "{column}"')
        if isinstance(label, (list, tuple, np.ndarray)):
            self.header.insert(col, list(reversed(label)))
        else:
            self.header.insert(col, [label])
        self.units.insert(col, unit or '')
        self.formats.insert(col, formats or '%g')
        self.descriptions.insert(col, description or '')
        self.hidden.insert(col, False)
        self.data.insert(col, [])
        if self.nsecs < len(self.header[col]) - 1:
            self.nsecs = len(self.header[col]) - 1
        if not key:
            key = label
        if value is not None:
            if isinstance(value, (list, tuple, np.ndarray)):
                if key and len(value) > 0 and isinstance(value[0], dict):
                    value = [d[key] if key in d else float('nan') for d in value]
                self.data[col].extend(value)
            else:
                self.data[col].append(value)
        if fac:
            for k in range(len(self.data[col])):
                self.data[col][k] *= fac
        self.addcol = len(self.data)
        self.__recompute_shape()
        return self

    def remove(self, columns):
        """Remove columns from the table.

        Parameters
        -----------
        columns: int or str or list of int or str
            Columns can be specified by index or name,
            see `index()` for details.

        Raises
        ------
        IndexError:
            If an invalid column was specified.
        """
        # fix columns:
        if not isinstance(columns, (list, tuple, np.ndarray)):
            columns = [ columns ]
        if not columns:
            return
        # remove:
        for col in columns:
            c = self.index(col)
            if c is None:
                if isinstance(col, (np.integer, int)):
                    col = '%d' % col
                raise IndexError('Cannot remove non-existing column ' + col)
                continue
            if c+1 < len(self.header):
                self.header[c+1].extend(self.header[c][len(self.header[c+1]):])
            del self.header[c]
            del self.units[c]
            del self.formats[c]
            del self.descriptions[c]
            del self.hidden[c]
            del self.data[c]
        if self.setcol >= len(self.data):
            self.setcol = 0
        self.__recompute_shape()

    def section(self, column, level):
        """The section name of a specified column.

        Parameters
        ----------
        column: None, int, or str
            A specification of a column.
            See self.index() for more information on how to specify a column.
        level: int
            The level of the section to be returned. The column label itself is level=0.

        Returns
        -------
        name: str
            The name of the section at the specified level containing
            the column.
        index: int
            The column index that contains this section
            (equal or smaller thant `column`).

        Raises
        ------
        IndexError:
            If `level` exceeds the maximum possible level.
        """
        if level < 0 or level > self.nsecs:
            raise IndexError('Invalid section level')
        column = self.index(column)
        while len(self.header[column]) <= level:
            column -= 1
        return self.header[column][level], column
    
    def set_section(self, column, label, level):
        """Set a section name.

        Parameters
        ----------
        column: None, int, or str
            A specification of a column.
            See self.index() for more information on how to specify a column.
        label: str
            The new name to be used for the section.
        level: int
            The level of the section to be set. The column label itself is level=0.
        """
        column = self.index(column)
        self.header[column][level] = label
        return column

    def append_section(self, label):
        """Add sections to the table header.

        Each column of the table has a header label. Columns can be
        grouped into sections. Sections can be nested arbitrarily.

        Parameters
        ----------
        label: stri or list of str
            The name(s) of the section(s).

        Returns
        -------
        index: int
            The column index where the section was appended.
        """
        if self.addcol >= len(self.data):
            if isinstance(label, (list, tuple, np.ndarray)):
                self.header.append(list(reversed(label)))
            else:
                self.header.append([label])
            self.units.append('')
            self.formats.append('')
            self.descriptions.append('')
            self.hidden.append(False)
            self.data.append([])
        else:
            if isinstance(label, (list, tuple, np.ndarray)):
                self.header[self.addcol] = list(reversed(label)) + self.header[self.addcol]
            else:
                self.header[self.addcol] = [label] + self.header[self.addcol]
        if self.nsecs < len(self.header[self.addcol]):
            self.nsecs = len(self.header[self.addcol])
        self.addcol = len(self.data) - 1
        self.__recompute_shape()
        return self.addcol
        
    def insert_section(self, column, section):
        """Insert a section at a given position of the table header.

        Parameters
        ----------
        columns int or str
            Column before which to insert the new section.
            Column can be specified by index or name,
            see `index()` for details.
        section: str
            The name of the section.

        Returns
        -------
        index: int
            The index of the column where the section was inserted.
            
        Raises
        ------
        IndexError:
            If an invalid column was specified.
        """
        col = self.index(column)
        if col is None:
            if isinstance(column, (np.integer, int)):
                column = '%d' % column
            raise IndexError('Cannot insert at non-existing column ' + column)
        self.header[col].append(section)
        if self.nsecs < len(self.header[col]) - 1:
            self.nsecs = len(self.header[col]) - 1
        return col

    def label(self, column):
        """The name of a column.

        Parameters
        ----------
        column: None, int, or str
            A specification of a column.
            See self.index() for more information on how to specify a column.

        Returns
        -------
        self: TableData
            This TableData
        """
        column = self.index(column)
        return self.header[column][0]

    def set_label(self, column, label):
        """Set the name of a column.

        Parameters
        ----------
        column: None, int, or str
            A specification of a column.
            See self.index() for more information on how to specify a column.
        label: str
            The new name to be used for the column.

        Returns
        -------
        self: TableData
            This TableData
        """        
        column = self.index(column)
        self.header[column][0] = label
        return self

    def set_labels(self, labels):
        """Set the labels of some columns.

        Parameters
        ----------
        labels: TableData, dict, list of str, list of list of str, None
            The new labels to be used.
            If TableData, take the labels of the respective column indices.
            If dict, keys are column labels (see self.index() for more
            information on how to specify a column), and values are
            the new labels for the respective columns as str or list of str.
            If list of str or list of list of str,
            set labels of the first successive columns to the list elements.
            If `None`, do nothing.

        Returns
        -------
        self: TableData
            This TableData
        """
        if isinstance(labels, TableData):
            for c in range(min(self.columns(), labels.columns())):
                self.header[c] = labels.header[c]
        elif isinstance(labels, dict):
            for c in labels:
                i = self.index(c)
                if i is None:
                    continue
                l = labels[c]
                if isinstance(l, (list, tuple)):
                    self.header[i] = l
                else:
                    self.header[i] = [l]
        elif isinstance(labels, (list, tuple, np.ndarray)) and not \
             (isinstance(labels, np.ndarray) and len(labels.shape) == 0):
            for c, l in enumerate(labels):
                if isinstance(l, (list, tuple)):
                    self.labels[c] = l
                else:
                    self.labels[c] = [l]
        return self

    def unit(self, column):
        """The unit of a column.

        Parameters
        ----------
        column: None, int, or str
            A specification of a column.
            See self.index() for more information on how to specify a column.

        Returns
        -------
        unit: str
            The unit.
        """
        column = self.index(column)
        return self.units[column]

    def set_unit(self, column, unit):
        """Set the unit of a column.

        Parameters
        ----------
        column: None, int, or str
            A specification of a column.
            See self.index() for more information on how to specify a column.
        unit: str
            The new unit to be used for the column.

        Returns
        -------
        self: TableData
            This TableData
        """
        column = self.index(column)
        self.units[column] = unit
        return self

    def set_units(self, units):
        """Set the units of some columns.

        Parameters
        ----------
        units: TableData, dict, list of str, str, None
            The new units to be used.
            If TableData, take the units of matching column labels.
            If dict, keys are column labels (see self.index() for more
            information on how to specify a column), and values are
            units for the respective columns as str.
            If list of str, set units of the first successive columns to
            the list elements.
            If `None`, do nothing.
            Otherwise, set units of all columns to `units`.

        Returns
        -------
        self: TableData
            This TableData
        """
        if isinstance(units, TableData):
            for c in units:
                i = self.index(c)
                if i is None:
                    continue
                self.units[i] = units.unit(c)
        elif isinstance(units, dict):
            for c in units:
                i = self.index(c)
                if i is None:
                    continue
                self.units[i] = units[c]
        elif isinstance(units, (list, tuple, np.ndarray)) and not \
             (isinstance(units, np.ndarray) and len(units.shape) == 0):
            for c, u in enumerate(units):
                self.units[c] = u
        elif units is not None:
            for c in range(len(units)):
                self.units[c] = units
        return self

    def format(self, column):
        """The format string of the column.

        Parameters
        ----------
        column: None, int, or str
            A specification of a column.
            See self.index() for more information on how to specify a column.

        Returns
        -------
        format: str
            The format string.
        """
        column = self.index(column)
        return self.formats[column]

    def set_format(self, column, format):
        """Set the format string of a column.

        Parameters
        ----------
        column: None, int, or str
            A specification of a column.
            See self.index() for more information on how to specify a column.
        format: str
            The new format string to be used for the column.

        Returns
        -------
        self: TableData
            This TableData
        """
        column = self.index(column)
        self.formats[column] = format
        return self

    def set_formats(self, formats):
        """Set the format strings of all columns.

        Parameters
        ----------
        formats: TableData, dict, list of str, str, None
            The new format strings to be used.
            If TableData, take the format strings of matching column labels.
            If dict, keys are column labels (see self.index() for more
            information on how to specify a column), and values are
            format strings for the respective columns as str.
            If list of str, set format strings of the first successive
            columns to the list elements.
            If `None`, do nothing.
            Otherwise, set format strings of all columns to `formats`.

        Returns
        -------
        self: TableData
            This TableData
        """
        if isinstance(formats, TableData):
            for c in formats:
                i = self.index(c)
                if i is None:
                    continue
                self.formats[i] = formats.format(c)
        elif isinstance(formats, dict):
            for c in formats:
                i = self.index(c)
                if i is None:
                    continue
                self.formats[i] = formats[c] or '%g'
        elif isinstance(formats, (list, tuple, np.ndarray)) and not \
             (isinstance(formats, np.ndarray) and len(formats.shape) == 0):
            for c, f in enumerate(formats):
                self.formats[c] = f or '%g'
        elif formats is not None:
            for c in range(len(formats)):
                self.formats[c] = formats or '%g'
        return self

    def description(self, column):
        """The description of a column.

        Parameters
        ----------
        column: None, int, or str
            A specification of a column.
            See self.index() for more information on how to specify a column.

        Returns
        -------
        description: str
            The description.
        """
        column = self.index(column)
        return self.descriptions[column]

    def set_description(self, column, description):
        """Set the description of a column.

        Parameters
        ----------
        column: None, int, or str
            A specification of a column.
            See self.index() for more information on how to specify a column.
        description: str
            The new description to be used for the column.

        Returns
        -------
        self: TableData
            This TableData
        """
        column = self.index(column)
        self.descriptions[column] = description
        return self

    def set_descriptions(self, descriptions):
        """Set the descriptions of some columns.

        Parameters
        ----------
        descriptions: TableData, dict, list of str, str, None
            The new descriptions to be used.
            If TableData, take the descriptions of matching column labels.
            If dict, keys are column labels (see self.index() for more
            information on how to specify a column), and values are
            descriptions for the respective columns as str.
            If list of str, set descriptions of the first successive columns to
            the list elements.
            If `None`, do nothing.

        Returns
        -------
        self: TableData
            This TableData
        """
        if isinstance(descriptions, TableData):
            for c in descriptions:
                i = self.index(c)
                if i is None:
                    continue
                self.descriptions[i] = descriptions.description(c)
        elif isinstance(descriptions, dict):
            for c in descriptions:
                i = self.index(c)
                if i is None:
                    continue
                self.descriptions[i] = descriptions[c]
        elif isinstance(descriptions, (list, tuple, np.ndarray)) and not \
             (isinstance(descriptions, np.ndarray) and len(descriptions.shape) == 0):
            for c, d in enumerate(descriptions):
                self.descriptions[c] = d
        return self

    def table_header(self):
        """The header of the table without content.

        Returns
        -------
        data: TableData
            A TableData object with the same header but empty data.
        """
        data = TableData()
        sec_indices = [-1] * self.nsecs
        for c in range(self.columns()):
            data.append(*self.column_head(c))
            for l in range(self.nsecs):
                s, i = self.section(c, l+1)
                if i != sec_indices[l]:
                    data.header[-1].append(s)
                    sec_indices[l] = i
        data.nsecs = self.nsecs
        return data

    def column_head(self, column, secs=False):
        """The name, unit, format, and description of a column.

        Parameters
        ----------
        column: None, int, or str
            A specification of a column.
            See self.index() for more information on how to specify a column.
        secs: bool
            If True, return all section names in addition to the column label.

        Returns
        -------
        name: str or list of str
            The column label or the label with all its sections.
        unit: str
            The unit.
        format: str
            The format string.
        description: str
            The description of the data column.
        """
        column = self.index(column)
        if secs:
            header = self.header[column]
            c = column - 1
            while len(header) < self.nsecs + 1 and c >= 0:
                if len(self.header[c]) > len(header):
                    header.extend(self.header[c][len(header):])
                c -= 1
            return list(reversed(header)), self.units[column], self.formats[column], self.descriptions[column]
        else:
            return self.header[column][0], self.units[column], self.formats[column], self.descriptions[column]

    def column_spec(self, column):
        """Full specification of a column with all its section names.

        Parameters
        ----------
        column: int or str
            Specifies the column.
            See self.index() for more information on how to specify a column.

        Returns
        -------
        s: str
            Full specification of the column by all its section names and its header label.
        """
        c = self.index(column)
        fh = [self.header[c][0]]
        for l in range(self.nsecs):
            fh.append(self.section(c, l+1)[0])
        return '>'.join(reversed(fh))
    
    def find_col(self, column):
        """Find the start and end index of a column specification.
        
        Parameters
        ----------
        column: None, int, or str
            A specification of a column.
            See self.index() for more information on how to specify a column.

        Returns
        -------
        c0: int or None
            A valid column index or None that is specified by `column`.
        c1: int or None
            A valid column index or None of the column following the range specified
            by `column`.
        """

        def find_column_indices(ss, si, minns, maxns, c0, strict=True):
            if si >= len(ss):
                return None, None, None, None
            ns0 = 0
            for ns in range(minns, maxns+1):
                nsec = maxns - ns
                if ss[si] == '':
                    si += 1
                    continue
                for c in range(c0, len(self.header)):
                    if nsec < len(self.header[c]) and \
                        ((strict and self.header[c][nsec] == ss[si]) or
                         (not strict and ss[si] in self.header[c][nsec])):
                        ns0 = ns
                        c0 = c
                        si += 1
                        if si >= len(ss):
                            c1 = len(self.header)
                            for c in range(c0+1, len(self.header)):
                                if nsec < len(self.header[c]):
                                    c1 = c
                                    break
                            return c0, c1, ns0, None
                        elif nsec > 0:
                            break
            return None, c0, ns0, si

        if column is None:
            return None, None
        if not isinstance(column, (np.integer, int)) and column.isdigit():
            column = int(column)
        if isinstance(column, (np.integer, int)):
            if column >= 0 and column < len(self.header):
                return column, column + 1
            else:
                return None, None
        # find column by header:
        ss = column.rstrip('>').split('>')
        maxns = self.nsecs
        si0 = 0
        while si0 < len(ss) and ss[si0] == '':
            maxns -= 1
            si0 += 1
        if maxns < 0:
            maxns = 0
        c0, c1, ns, si = find_column_indices(ss, si0, 0, maxns, 0, True)
        if c0 is None and c1 is not None:
            c0, c1, ns, si = find_column_indices(ss, si, ns, maxns, c1, False)
        return c0, c1

    def index(self, column):
        """The column index of a column specifier.
        
        Parameters
        ----------
        column: None, int, or str
            A specification of a column.
            - None: no column is specified
            - int: the index of the column (first column is zero), e.g. `index(2)`.
            - a string representing an integer is converted into the column index,
              e.g. `index('2')`
            - a string specifying a column by its header.
              Header names of descending hierarchy are separated by '>'.

        Returns
        -------
        index: int or None
            A valid column index or None.
        """
        c0, c1 = self.find_col(column)
        return c0

    def __contains__(self, column):
        """Check for existence of a column.

        Parameters
        ----------
        column: None, int, or str
            The column to be checked.
            See self.index() for more information on how to specify a column.

        Returns
        -------
        contains: bool
            True if `column` specifies an existing column key.
        """
        return self.index(column) is not None

    def keys(self):
        """List of unique column keys for all available columns.

        Returns
        -------
        keys: list of str
            List of unique column specifications.
        """
        return [self.column_spec(c) for c in range(self.columns())]

    def values(self):
        """List of column data corresponding to keys(). Same as `self.data`.

        Returns
        -------
        data: list of list of values
            The data of the table. First index is columns!
        """
        return self.data

    def items(self):
        """Generator over column names and corresponding data.

        Yields
        ------
        item: tuple
            Unique column specifications and the corresponding data.
        """
        for c in range(self.columns()):
            yield self.column_spec(c), self.data[c]
        
    def __len__(self):
        """The number of rows.
        
        Returns
        -------
        rows: int
            The number of rows contained in the table.
        """
        return self.rows()

    def __iter__(self):
        """Initialize iteration over data columns.
        """
        self.iter_counter = -1
        return self

    def __next__(self):
        """Next unique column key.

        Returns
        -------
        s: str
            Full specification of the column by all its section names and its header label.
        """
        self.iter_counter += 1
        if self.iter_counter >= self.columns():
            raise StopIteration
        else:
            return self.column_spec(self.iter_counter)

    def rows(self):
        """The number of rows.
        
        Returns
        -------
        rows: int
            The number of rows contained in the table.
        """
        return max(map(len, self.data)) if self.data else 0
    
    def columns(self):
        """The number of columns.
        
        Returns
        -------
        columns: int
            The number of columns contained in the table.
        """
        return len(self.header)

    def row(self, index):
        """A single row of the table as TableData.

        Parameters
        ----------
        index: int
            The index of the row to be returned.

        Returns
        -------
        data: TableData
            A TableData object with a single row.
        """
        data = TableData()
        sec_indices = [-1] * self.nsecs
        for c in range(self.columns()):
            data.append(*self.column_head(c))
            for l in range(self.nsecs):
                s, i = self.section(c, l+1)
                if i != sec_indices[l]:
                    data.header[-1].append(s)
                    sec_indices[l] = i
            data.data[-1] = [self.data[c][index]]
        data.nsecs = self.nsecs
        return data

    def row_list(self, index):
        """A single row of the table as list.

        Parameters
        ----------
        index: int
            The index of the row to be returned.

        Returns
        -------
        data: list
            A list with data values of each column of row `index`.
        """
        data = []
        for c in range(self.columns()):
            data.append(self.data[c][index])
        return data

    def row_data(self):
        """A generator for iterating over rows of the table.

        Yields
        ------
        data: list
            A list with data values of each column.
        """
        for r in range(self.rows()):
            yield self.row_list(r)

    def row_dict(self, index):
        """A single row of the table as dictionary.

        Parameters
        ----------
        index: int
            The index of the row to be returned.

        Returns
        -------
        data: dict
            A dictionary with column header as key and corresponding data value of row `index`
            as value.
        """
        data = {}
        for c in range(self.columns()):
            data[self.column_spec(c)] = self.data[c][index]
        return data

    def column(self, col):
        """A single column of the table.

        Parameters
        ----------
        col: None, int, or str
            The column to be returned.
            See self.index() for more information on how to specify a column.

        Returns
        -------
        table: TableData
            A TableData object with a single column.
        """
        data = TableData()
        c = self.index(col)
        data.append(*self.column_head(c))
        data.data = [self.data[c]]
        data.nsecs = 0
        return data

    def __call__(self, column):
        """A single column of the table as a ndarray.

        Parameters
        ----------
        column: None, int, or str
            The column to be returned.
            See self.index() for more information on how to specify a column.

        Returns
        -------
        data: 1-D ndarray
            Content of the specified column as a ndarray.
        """
        c = self.index(column)
        return np.asarray(self.data[c])

    def __setupkey(self, key):
        """Helper function that turns a key into row and column indices.

        Returns
        -------
        rows: list of int, slice, None
            Indices of selected rows.
        cols: list of int
            Indices of selected columns.

        Raises
        ------
        IndexError:
            If an invalid column was specified.
        """
        if type(key) is not tuple:
            if isinstance(key, str):
                cols = key
                rows = slice(0, self.rows(), 1)
            elif isinstance(key, slice) and isinstance(key.start, str) and isinstance(key.stop, str):
                cols = key
                rows = slice(0, self.rows(), 1)
            else:
                rows = key
                cols = range(self.columns())
        else:
            rows = key[0]
            cols = key[1]
        if isinstance(cols, slice):
            start = cols.start
            if start is not None:
                start = self.index(start)
                if start is None:
                    raise IndexError('"%s" is not a valid column index' % cols.start)
            stop = cols.stop
            if stop is not None:
                stop_str = isinstance(stop, str)
                stop = self.index(stop)
                if stop is None:
                    raise IndexError('"%s" is not a valid column index' % cols.stop)
                if stop_str:
                    stop += 1
            cols = slice(start, stop, cols.step)
            cols = range(self.columns())[cols]
        else:
            if not isinstance(cols, (list, tuple, np.ndarray, range)):
                cols = [cols]
            c = [self.index(inx) for inx in cols]
            if None in c:
                raise IndexError('"%s" is not a valid column index' % cols[c.index(None)])
            cols = c
        if isinstance(rows, np.ndarray) and rows.dtype == np.dtype(bool):
            rows = np.where(rows)[0]
            if len(rows) == 0:
                rows = None
        return rows, cols

    def __getitem__(self, key):
        """Data elements specified by slice.

        Parameters
        -----------
        key:
            First key specifies row, (optional) second key the column.
            Columns can be specified by index or name,
            see `index()` for details.
            A single key of strings selects columns by their names: `td[:, 'col'] == td['col']`
            If a stop column is specified by name,
            it is inclusively!

        Returns
        -------
        data:
            - A single data value if a single row and a single column is specified.
            - A ndarray of data elements if a single column is specified.
            - A TableData object for multiple columns.
            - None if no row is selected (e.g. by a logical index that nowhere is True)

        Raises
        ------
        IndexError:
            If an invalid column was specified.
        """
        rows, cols = self.__setupkey(key)
        if len(cols) == 1:
            if cols[0] >= self.columns():
                return None
            if rows is None:
                return None
            elif isinstance(rows, slice):
                return np.asarray(self.data[cols[0]][rows])
            elif isinstance(rows, (list, tuple, np.ndarray)):
                return np.asarray([self.data[cols[0]][r] for r in rows if r < len(self.data[cols[0]])])
            elif rows < len(self.data[cols[0]]):
                return self.data[cols[0]][rows]
            else:
                return None
        else:
            data = TableData()
            sec_indices = [-1] * self.nsecs
            for c in cols:
                data.append(*self.column_head(c, secs=True))
                if rows is None:
                    continue
                if isinstance(rows, (list, tuple, np.ndarray)):
                    for r in rows:
                        data.data[-1].append(self.data[c][r])
                else:
                    try:
                        if isinstance(self.data[c][rows],
                                      (list, tuple, np.ndarray)):
                            data.data[-1].extend(self.data[c][rows])
                        else:
                            data.data[-1].append(self.data[c][rows])
                    except IndexError:
                        data.data[-1].append(np.nan)
            data.nsecs = self.nsecs
            return data

    def __setitem__(self, key, value):
        """Assign values to data elements specified by slice.

        Parameters
        -----------
        key:
            First key specifies row, (optional) second one the column.
            Columns can be specified by index or name,
            see `index()` for details.
            A single key of strings selects columns by their names: `td[:, 'col'] == td['col']`
            If a stop column is specified by name,
            it is inclusively!
        value: TableData, list, ndarray, float, ...
            Value(s) used to assing to the table elements as specified by `key`.

        Raises
        ------
        IndexError:
            If an invalid column was specified.
        """
        rows, cols = self.__setupkey(key)
        if rows is None:
            return
        if isinstance(value, TableData):
            if isinstance(self.data[cols[0]][rows], (list, tuple, np.ndarray)):
                for k, c in enumerate(cols):
                    self.data[c][rows] = value.data[k]
            else:
                for k, c in enumerate(cols):
                    self.data[c][rows] = value.data[k][0]
        else:
            if len(cols) == 1:
                if isinstance(rows, (list, tuple, np.ndarray)):
                    if len(rows) == 1:
                        self.data[cols[0]][rows[0]] = value
                    elif isinstance(value, (list, tuple, np.ndarray)):
                        for k, r in enumerate(rows):
                            self.data[cols[0]][r] = value[k]
                    else:
                        for r in rows:
                            self.data[cols[0]][r] = value
                elif isinstance(value, (list, tuple, np.ndarray)):
                    self.data[cols[0]][rows] = value
                elif isinstance(rows, (np.integer, int)):
                    self.data[cols[0]][rows] = value
                else:
                    n = len(self.data[cols[0]][rows])
                    if n > 1:
                        self.data[cols[0]][rows] = [value]*n
                    else:
                        self.data[cols[0]][rows] = value
            else:
                if isinstance(self.data[0][rows], (list, tuple, np.ndarray)):
                    for k, c in enumerate(cols):
                        self.data[c][rows] = value[:,k]
                elif isinstance(value, (list, tuple, np.ndarray)):
                    for k, c in enumerate(cols):
                        self.data[c][rows] = value[k]
                else:
                    for k, c in enumerate(cols):
                        self.data[c][rows] = value

    def __delitem__(self, key):
        """Delete data elements or whole columns or rows.

        Parameters
        -----------
        key:
            First key specifies row, (optional) second one the column.
            Columns can be specified by index or name,
            see `index()` for details.
            A single key of strings selects columns by their names: `td[:, 'col'] == td['col']`
            If a stop column is specified by name,
            it is inclusively!
            If all rows are selected, then the specified columns are removed from the table.
            Otherwise only data values are removed.
            If all columns are selected than entire rows of data values are removed.
            Otherwise only data values in the specified rows are removed.

        Raises
        ------
        IndexError:
            If an invalid column was specified.
        """
        rows, cols = self.__setupkey(key)
        if rows is None:
            return
        row_indices = np.arange(self.rows(), dtype=int)[rows]
        if isinstance(row_indices, np.ndarray):
            if len(row_indices) == self.rows():
                # delete whole columns:
                self.remove(cols)
            elif len(row_indices) > 0:
                for r in reversed(sorted(row_indices)):
                    for c in cols:
                        if r < len(self.data[c]):
                            del self.data[c][r]
            self.__recompute_shape()
        else:
            for c in cols:
                del self.data[c][row_indices]
            self.__recompute_shape()

    def array(self, row=None):
        """The table data as a ndarray.

        Parameters
        ----------
        row: int or None
            If specified, a 1D ndarray of that row will be returned.

        Returns
        -------
        data: 2D or 1D ndarray
            If no row is specified, the data content of the entire table
            as a 2D ndarray (rows first).
            If a row is specified, a 1D ndarray of that row.
        """
        if row is None:
            return np.array(self.data).T
        else:
            return np.array([d[row] for d in self.data])

    def data_frame(self):
        """The table data as a pandas DataFrame.

        Returns
        -------
        data: pandas.DataFrame
            A pandas DataFrame of the whole table.
        """
        return pd.DataFrame(self.dict())

    def dicts(self, raw_values=True, missing=default_missing_str):
        """The table as a list of dictionaries.

        Parameters
        ----------
        raw_values: bool
            If True, use raw table values as values,
            else format the values and add unit string.
        missing: str
            String indicating non-existing data elements.

        Returns
        -------
        table: list of dict
            For each row of the table a dictionary with header as key.
        """
        table = []
        for row in range(self.rows()):
            data = {}
            for col in range(len(self.header)):
                if raw_values:
                    v = self.data[col][row];
                else:
                    if isinstance(self.data[col][row], (float, np.floating)) and m.isnan(self.data[col][row]):
                        v = missing
                    else:
                        u = ''
                        if not self.units[col] in '1-' and self.units[col] != 'a.u.':
                            u = self.units[col]
                        v = (self.formats[col] % self.data[col][row]) + u
                data[self.header[col][0]] = v
            table.append(data)
        return table

    def dict(self):
        """The table as a dictionary.

        Returns
        -------
        table: dict
            A dictionary with keys being the column headers and
            values the list of data elements of the corresponding column.
        """
        table = {k: v for k, v in self.items()}
        return table

    def _add_table_data(self, data, add_all):
        """Add data of a TableData.

        Parameters
        ----------
        data: TableData
            Table with the data to be added.
        add_all: bool
            If False, then only data of columns that already exist in
            the table are added to the table. If the table is empty or
            `add_all` is set to `True` then all data is added and if
            necessary new columns are appended to the table.
        """
        empty = False
        if self.shape[1] == 0:
            add_all = True
            empty = True
        maxr = self.rows()
        for k in data.keys():
            col = self.index(k)
            if empty or col is None:
                if not add_all:
                    continue
                self.append(*data.column_head(k, secs=True),
                            value=[np.nan]*maxr)
                col = len(self.data) - 1
            c = data.index(k)
            self.data[col].extend(data.data[c])
        self.__recompute_shape()

    def _add_dict(self, data, add_all):
        """Add data of a TableData.

        Parameters
        ----------
        data: dict
            Keys are column labels and values are single values or
            lists of values to be added to the corresponding table columns.
        add_all: bool
            If False, then only data of columns that already exist in
            the table are added to the table. If the table is empty or
            `add_all` is set to `True` then all data is added and if
            necessary new columns are appended to the table.

        """
        empty = False
        if self.shape[1] == 0:
            add_all = True
            empty = True
        maxr = self.rows()
        for key in data:
            new_key = key
            new_unit = ''
            if '/' in key:
                p = key.split('/')
                new_key = p[0].strip()
                new_unit = '/'.join(p[1:])
            col = self.index(new_key)
            if empty or col is None:
                if not add_all:
                    continue
                self.append(new_key, new_unit,
                            value=[np.nan]*maxr)
                col = len(self.data) - 1
            if isinstance(data[key], (list, tuple, np.ndarray)):
                self.data[col].extend(data[key])
            else:
                self.data[col].append(data[key])
        self.__recompute_shape()

    def add(self, data, column=None, add_all=False):
        """Add data elements to successive columns.

        The current column is set behid the added columns.

        Parameters
        ----------
        data: float, int, str, etc. or list thereof or list of list thereof or dict or list of dict or TableData
            Data values to be appended to successive columns:
            - A single value is simply appended to the specified
              column of the table.
            - A 1D-list of values is appended to successive columns of the table
              starting with the specified column.
            - The columns (second index) of a 2D-list of values are
              appended to successive columns of the table starting
              with the specified column.
            - Values of a dictionary or of a list of dictionaries are
              added to the columns specified by the keys. Dictionary
              values can also be lists of values. Their values are
              added to successive rows of the columns specified by the
              dictionary keys. Does not affect the current column.
            - All elements of a TableData are added to matching columns.
              Does not affect the current column.
        column: None, int, or str
            The first column to which the data should be appended,
            if `data` does not specify columns.
            If None, append to the current column.
            See self.index() for more information on how to specify a column.
        add_all: bool
            If the data are given as dictionaries or TableData, then
            only data of columns that already exist in the table are
            added to the table. If the table is empty or `add_all` is
            set to `True` then all data is added and if necessary new
            columns are appended to the table.
        """
        if self.shape[1] == 0:
            add_all = True
        column = self.index(column)
        if column is None:
            column = self.setcol
        if isinstance(data, TableData):
            self._add_table_data(data, add_all)
        elif isinstance(data, (list, tuple, np.ndarray)) and not \
             (isinstance(data, np.ndarray) and len(data.shape) == 0):
            if len(data) > 0 and \
               isinstance(data[0], (list, tuple, np.ndarray)) and not \
               (isinstance(data[0], np.ndarray) and len(data[0].shape) == 0):
                # 2D list, rows first:
                for row in data:
                    for i, val in enumerate(row):
                        self.data[column + i].append(val)
                self.setcol = column + len(data[0])
            elif len(data) > 0 and isinstance(data[0], dict):
                # list of dictionaries:
                for d in data:
                    self._add_dict(d, add_all)
            else:
                # 1D list:
                for val in data:
                    self.data[column].append(val)
                    column += 1
                self.setcol = column
        elif isinstance(data, dict):
            # dictionary with values:
            self._add_dict(data, add_all)
        else:
            # single value:
            self.data[column].append(data)
            self.setcol = column + 1
        if self.setcol >= len(self.data):
            self.setcol = 0
        self.__recompute_shape()

    def append_data_column(self, data, column=None):
        """Append data elements to a column.

        The current column is incremented by one.

        Parameters
        ----------
        data: float, int, str, etc. or list thereof
            Data values to be appended to a column.
        column: None, int, or str
            The column to which the data should be appended.
            If None, append to the current column.
            See self.index() for more information on how to specify a column.
        """
        column = self.index(column)
        if column is None:
            column = self.setcol
        if isinstance(data, (list, tuple, np.ndarray)):
            self.data[column].extend(data)
            column += 1
            self.setcol = column
        else:
            self.data[column].append(data)
            self.setcol = column+1
        if self.setcol >= len(self.data):
            self.setcol = 0
        self.__recompute_shape()

    def set_column(self, column):
        """Set the column where to add data.

        Parameters
        ----------
        column: int or str
            The column to which data elements should be appended.
            See self.index() for more information on how to specify a column.

        Raises
        ------
        IndexError:
            If an invalid column was specified.
        """
        col = self.index(column)
        if col is None:
            if isinstance(column, (np.integer, int)):
                column = '%d' % column
            raise IndexError('column ' + column + ' not found or invalid')
        self.setcol = col
        return col

    def fill_data(self):
        """Fill up all columns with missing data to have the same number of
        data elements.
        """
        # maximum rows:
        maxr = self.rows()
        # fill up:
        for c in range(len(self.data)):
            while len(self.data[c]) < maxr:
                self.data[c].append(np.nan)
        self.setcol = 0
        self.__recompute_shape()

    def clear_data(self):
        """Clear content of the table but keep header.
        """
        for c in range(len(self.data)):
            self.data[c] = []
        self.setcol = 0
        self.__recompute_shape()

    def clear(self):
        """Clear the table of any content and header information.
        """
        self.ndim = 2
        self.size = 0
        self.shape = (0, 0)
        self.nsecs = 0
        self.header = []
        self.units = []
        self.formats = []
        self.descriptions = []
        self.data = []
        self.hidden = []
        self.setcol = 0
        self.addcol = 0
                
    def sort(self, columns, reverse=False):
        """Sort the table rows in place.

        Parameters
        ----------
        columns: int or str or list of int or str
            A column specifier or a list of column specifiers of the columns
            to be sorted.
        reverse: boolean
            If `True` sort in descending order.

        Raises
        ------
        IndexError:
            If an invalid column was specified.
        """
        # fix columns:
        if not isinstance(columns, (list, tuple, np.ndarray)):
            columns = [ columns ]
        if not columns:
            return
        cols = []
        for col in columns:
            c = self.index(col)
            if c is None:
                if isinstance(col, (np.integer, int)):
                    col = '%d' % col
                raise IndexError('sort column ' + col + ' not found')
                continue
            cols.append(c)
        # get sorted row indices:
        row_inx = range(self.rows())
        row_inx = sorted(row_inx, key=lambda x : [float('-inf') if self.data[c][x] is np.nan \
                         or self.data[c][x] != self.data[c][x] \
                         else self.data[c][x] for c in cols], reverse=reverse)
        # sort table according to indices:
        for c in range(self.columns()):
            self.data[c] = [self.data[c][r] for r in row_inx]

    def key_value(self, row, col, missing=default_missing_str):
        """A data element returned as a key-value pair.

        Parameters
        ----------
        row: int
            Specifies the row from which the data element should be retrieved.
        col: None, int, or str
            A specification of a column.
            See self.index() for more information on how to specify a column.
        missing: str
            String indicating non-existing data elements.

        Returns
        -------
        key: str
            Header label of the column
        value: str
            A textual representation of the data element according to the format
            of the column, followed by the unit of the column.
        """
        col = self.index(col)
        if col is None:
            return ''
        if isinstance(self.data[col][row], (float, np.floating)) and m.isnan(self.data[col][row]):
            v = missing
        else:
            u = ''
            if not self.units[col] in '1-' and self.units[col] != 'a.u.':
                u = self.units[col]
            v = (self.formats[col] % self.data[col][row]) + u
        return self.header[col][0], v

    def _aggregate(self, funcs, columns=None, label=None,
                   numbers_only=False, remove_nans=False, single_row=False,
                   keep_columns=None):
        """Apply functions to columns.

        Parameter
        ---------
        funcs: function, list of function, dict
            Functions that are applied to columns of the table.
            - a single function that is applied to the `columns`.
              The results are named according to the function's `__name__`.
            - a list or tuple of functions.
              The results are named according to the functions' `__name__`.
            - a dictionary. The results are named after the provided keys,
              the functions are given by the values.
              If the function returns more than one value, then the
              corresponding key in the dictionary needs to be a tuple
              (not a list!) of names for each of the returned values.
            Functions in lists or dictionaries can be just a plain
            function, like `max` or `np.mean`. In case a function
            needs further arguments, then you need to supply a tuple
            with the first elements being the function, the second
            element being another tuple holding positional arguments,
            and an optional third argument holding a dictionary for
            key-word arguments.
        columns: None, int or str or list of int or str
            Columns of the table on which functions are applied.
            If None apply functions on all columns.
        label: str or list of str
            Column label and optional section names of the first
            column with the function labels (if `single_row` is `False`).
        numbers_only: bool
            If True, skip columns that do not contain numbers.
        remove_nans: bool
            If True, remove nans before passing column values to function.
        single_row: bool
            If False, add for each function a row to the table.
            If True, add function values in a single row.
        keep_columns: None, int or str or list of int or str
            Columns of the table from which to simply keep the first value.
            Only if single_row is True. Usefull for grouped tables.
            Order of columns and keep_columns are kept from the original table.

        Returns
        -------
        dest: TableData
            A new table with the column headers specified by `columns`.
            A first column is inserted with the function labels.
            The functions are applied to all columns specified by `columns`
            and their return values are written into the new table.

        """
        # standardize functions dictionary:
        if not isinstance(funcs, (list, tuple, dict)):
            funcs = [funcs]
        if isinstance(funcs, (list, tuple)):
            fs = {}
            for f in funcs:
                fs[f.__name__] = f
            funcs = fs
        fs = {}
        for k in funcs:
            kk = k
            if not isinstance(k, tuple):
                kk = (k,)
            v = funcs[k]
            if not isinstance(v, tuple):
                v = (funcs[k], (), {})
            elif len(v) < 3:
                v = (v[0], v[1], {})
            fs[kk] = v
        funcs = fs
        # standardize columns:
        if columns is None:
            columns = list(range(self.shape[1]))
        if not isinstance(columns, (list, tuple, np.ndarray)):
            columns = [columns]
        if numbers_only:
            cols = []
            for c in columns:
                c = self.index(c)
                if len(self.data[c]) > 0 and \
                   isinstance(self.data[c][0], (float, int, np.floating, np.integer)):
                    cols.append(c)
            columns = cols
        if label is None:
            label = 'property'
        dest = TableData()
        if single_row:
            if keep_columns is None:
                keep_columns = []
            elif not isinstance(keep_columns, (list, tuple)):
                keep_columns = [keep_columns]
            keep_columns = [self.index(c) for c in keep_columns]
            columns = [self.index(c) for c in columns]
            columns = [c for c in columns if not c in keep_columns]
            keep = np.zeros(len(keep_columns) + len(columns), dtype=bool)
            keep[:len(keep_columns)] = True
            columns = keep_columns + columns
            idx = np.argsort(columns)
            for i in idx:
                c = columns[i]
                if keep[i]:
                    name, unit, format, descr = self.column_head(c, secs=True)
                    dest.append(name + ['-'], unit, format, descr,
                                value=self.data[c][0])
                else:
                    name, unit, format, descr = self.column_head(c, secs=True)
                    values = self[:, c]
                    if remove_nans:
                        values = values[np.isfinite(values)]
                    for k in funcs:
                        v = funcs[k][0](values, *funcs[k][1], **funcs[k][2])
                        if len(k) == 1:
                            dest.append(name + [k[0]], unit, format, descr,
                                        value=v)
                        else:
                            for j in range(len(k)):
                                dest.append(name + [k[j]], unit, format, descr,
                                            value=v[j])
            dest.fill_data()
        else:
            dest.append(label, '', '%-s')
            for c in columns:
                dest.append(*self.column_head(c, secs=True))
            for k in funcs:
                for j in range(len(k)):
                    dest.add(k[j], 0)
                for i, c in enumerate(columns):
                    values = self[:, c]
                    if remove_nans:
                        values = values[np.isfinite(values)]
                    v = funcs[k][0](values, *funcs[k][1], **funcs[k][2])
                    if len(k) == 1:
                        dest.add(v, i + 1)
                    else:
                        for j in range(len(k)):
                            dest.add(v[j], i + 1)
                dest.fill_data()
        return dest

    def aggregate(self, funcs, columns=None, label=None,
                  numbers_only=False, remove_nans=False,
                  single_row=False, by=None):
        """Apply functions to columns.

        Parameter
        ---------
        funcs: function, list of function, dict
            Functions that are applied to columns of the table.
            - a single function that is applied to the `columns`.
              The results are named according to the function's `__name__`.
            - a list or tuple of functions.
              The results are named according to the functions' `__name__`.
            - a dictionary. The results are named after the provided keys,
              the functions are given by the values.
              If the function returns more than one value, then the
              corresponding key in the dictionary needs to be a tuple
              (not a list!) of names for each of the returned values.
            Functions in lists or dictionaries can be just a plain
            function, like `max` or `np.mean`. In case a function
            needs further arguments, then you need to supply a tuple
            with the first elements being the function, the second
            element being another tuple holding positional arguments,
            and an optional third argument holding a dictionary for
            key-word arguments.
        columns: None, int or str or list of int or str
            Columns of the table on which functions are applied.
            If None apply functions on all columns.
        label: str or list of str
            Column label and optional section names of the first
            column with the function labels (if `single_row` is `False`).
        numbers_only: bool
            If True, skip columns that do not contain numbers.
        remove_nans: bool
            If True, remove nans before passing column values to function.
        single_row: bool
            If False, add for each function a row to the table.
            If True, add function values in a single row.
        by: None, int or str or list of int or str
            Group the table by the specified columns and apply the functions
            to each resulting sub-table separately.

        Returns
        -------
        dest: TableData
            A new table with the column headers specified by `columns`.
            A first column is inserted with the function labels
            (if not `single_row`).
            The functions are applied to all columns specified by `columns`
            and their return values are written into the new table.
        """
        if by is not None:
            # aggregate on grouped table:
            if not isinstance(by, (list, tuple)):
                by = [by]
            if len(by) > 0:
                gd = TableData()
                for name, values in self.groupby(*by):
                    ad = values._aggregate(funcs, columns, label,
                                           numbers_only=numbers_only,
                                           remove_nans=remove_nans,
                                           single_row=True, keep_columns=by)
                    gd.add(ad)
                return gd
        # aggregate on whole table:
        return self._aggregate(funcs, columns, label,
                               numbers_only=numbers_only,
                               remove_nans=remove_nans,
                               single_row=single_row,
                               keep_columns=None)

    def statistics(self, columns=None, label=None,
                   remove_nans=False, single_row=False, by=None):
        """Descriptive statistics of each column.
        
        Parameter
        ---------
        columns: None, int or str or list of int or str
            Columns of the table on which statistics should be computed.
            If None apply functions on all columns.
        label: str or list of str
            Column label and optional section names of the first
            column with the function labels (if `single_row` is `False`).
        remove_nans: bool
            If True, remove nans before passing column values to function.
        single_row: bool
            If False, add for each function a row to the table.
            If True, add function values in a single row.
        by: None, int or str or list of int or str
            Group the table by the specified columns and compute statistics
            to each resulting sub-table separately.

        Returns
        -------
        dest: TableData
            A new table with the column headers specified by `columns`.
            For each column that contains numbers some basic
            descriptive statistics is computed.
        """
        if label is None:
            label = 'statistics'
        funcs = {'mean': np.mean,
                 'std': np.std,
                 'min': np.min,
                 ('quartile1', 'median', 'quartile2'):
                     (np.quantile, ([0.25, 0.5, 0.75],)),
                 'max': np.max,
                 'count': len}
        ds = self.aggregate(funcs, columns, label,
                            numbers_only=True,
                            remove_nans=remove_nans,
                            single_row=single_row, by=by)
        if by is not None:
            if not isinstance(by, (list, tuple)):
                by = [by]
            if len(by) > 0:
                single_row = True
        c0 = 0
        if not single_row:
            ds.set_format(0, '%-10s')
            c0 = 1
        for c in range(c0, ds.shape[1]):
            f = ds.formats[c]
            if single_row and ds.label(c) == 'count':
                ds.set_unit(c, '')
                ds.set_format(c, '%d')
            elif f[-1] in 'fge':
                i0 = f.find('.')
                if i0 > 0:
                    p = int(f[i0 + 1:-1])
                    f = f'{f[:i0 + 1]}{p + 1}{f[-1]}'
                ds.set_format(c, f)
            else:
                ds.set_format(c, '%.1f')
        return ds
        
    def groupby(self, *columns):
        """ Iterate through unique values of a column.

        Parameter
        ---------
        columns: int or str
            One or several columns used to group the data.
            See self.index() for more information on how to specify a column.

        Yields
        ------
        values: float or str or tuple of float or str
            The values of the specified columns.
        data: TableData
            The sub table where the specified columns equals `values`.
        """
        # check column indices and values:
        cols = []
        vals = []
        for col in columns:
            c = self.index(col)
            if c is None:
                raise StopIteration
            cols.append(c)
            vals.append(np.unique(self.data[c]))
        for values in product(*vals):
            mask = np.ones(len(self), dtype=bool)
            for c, v in zip(cols, values):
                mask &= self[:, c] == v
            if len(cols) == 1:
                yield values[0], self[mask]
            else:
                yield values, self[mask]

    def hide(self, column):
        """Hide a column or a range of columns.

        Hidden columns will not be printed out by the write() function.

        Parameters
        ----------
        column: int or str
            The column to be hidden.
            See self.index() for more information on how to specify a column.
        """
        c0, c1 = self.find_col(column)
        if c0 is not None:
            for c in range(c0, c1):
                self.hidden[c] = True

    def hide_all(self):
        """Hide all columns.

        Hidden columns will not be printed out by the write() function.
        """
        for c in range(len(self.hidden)):
            self.hidden[c] = True

    def hide_empty_columns(self, missing=default_missing_inputs):
        """Hide all columns that do not contain data.

        Hidden columns will not be printed out by the write() function.

        Parameters
        ----------
        missing: list of str
            Strings indicating missing data.
        """
        for c in range(len(self.data)):
            # check for empty column:
            isempty = True
            for v in self.data[c]:
                if isinstance(v, (float, np.floating)):
                    if not m.isnan(v):
                        isempty = False
                        break
                else:
                    if not v in missing:
                        isempty = False
                        break
            if isempty:
                self.hidden[c] = True

    def show(self, column):
        """Show a column or a range of columns.

        Undoes hiding of a column.

        Parameters
        ----------
        column: int or str
            The column to be shown.
            See self.index() for more information on how to specify a column.
        """
        c0, c1 = self.find_col(column)
        if c0 is not None:
            for c in range(c0, c1):
                self.hidden[c] = False

    def write(self, fh=sys.stdout, table_format=None, delimiter=None,
              unit_style=None, column_numbers=None, sections=None,
              align_columns=None, shrink_width=True,
              missing=default_missing_str, center_columns=False,
              latex_unit_package=None, latex_label_command='',
              latex_merge_std=False, descriptions_name='-description.md',
              section_headings=None, maxc=80):
        """Write the table to a file or stream.

        Parameters
        ----------
        fh: filename or stream
            If not a stream, the file with path `fh` is opened.
            If `fh` does not have an extension,
            the `table_format` is appended as an extension.
            Otherwise `fh` is used as a stream for writing.
        table_format: None or str
            The format to be used for output.
            One of 'out', 'dat', 'ascii', 'csv', 'rtai', 'md', 'tex', 'html'.
            If None or 'auto' then the format is set to the extension of the
            filename given by `fh`.
            If `fh` is a stream the format is set to 'dat'.
        delimiter: str
            String or character separating columns, if supported by the
            `table_format`.
            If None or 'auto' use the default for the specified `table_format`.
        unit_style: None or str
            - None or 'auto': use default of the specified `table_format`.
            - 'row': write an extra row to the table header specifying the
              units of the columns.
            - 'header': add the units to the column headers.
            - 'none': do not specify the units.
        column_numbers: str or None
            Add a row specifying the column index:
            - 'index': indices are integers, first column is 0.
            - 'num': indices are integers, first column is 1.
            - 'aa': use 'a', 'b', 'c', ..., 'z', 'aa', 'ab', ... for indexing
            - 'aa': use 'A', 'B', 'C', ..., 'Z', 'AA', 'AB', ... for indexing
            - None or 'none': do not add a row with column indices
            TableData.column_numbering is a list with the supported styles.
        sections: None or int
            Number of section levels to be printed.
            If `None` or 'auto' use default of selected `table_format`.
        align_columns: boolean
            - `True`: set width of column formats to make them align.
            - `False`: set width of column formats to 0 - no unnecessary spaces.
            - None or 'auto': Use default of the selected `table_format`.
        shrink_width: boolean
            If `True` disregard width specified by the format strings,
            such that columns can become narrower.
        missing: str
            Indicate missing data by this string.
        center_columns: boolean
            If True center all columns (markdown, html, and latex).
        latex_unit_package: None or 'siunitx' or 'SIunit'
            Translate units for the specified LaTeX package.
            If None set sub- and superscripts in text mode.
            If 'siunitx', also use `S` columns for numbers to align
            them on the decimal point.
        latex_label_command: str
            LaTeX command for formatting header labels.
            E.g. 'textbf' for making the header labels bold.
        latex_merge_std: str
            Merge header of columns with standard deviations with
            previous column (LaTeX tables only), but separate them
            with $\\pm$. Valid labels for standrad deviations are
            listed in `TableData.stdev_labels`.
        descriptions_name: None or str
            If not None and if `fh` is a file path, then write the column
            descriptions to a file with the same name as `fh`, but with
            `descriptions_name` appended.
        section_headings: None or int
            How to write treat header sections in the column descriptions.
            If set, set header sections as headings with the top-level
            section at the level as specified. 0 is the top level.
            If False, just produce a nested list.
        maxc: int
            Maximum character count for each line in the column descriptions.
        
        Returns
        -------
        file_name: str or None
            The full name of the file into which the data were written.

        Supported file formats
        ----------------------
        
        ## `dat`: data text file
        ``` plain
        # info           reaction     
        # size   weight  delay  jitter
        # m      kg      ms     mm    
           2.34     123   98.7      23
          56.70    3457   54.3      45
           8.90      43   67.9     345
        ```

        ## `ascii`: ascii-art table
        ``` plain
        |---------------------------------|
        | info           | reaction       |
        | size  | weight | delay | jitter |
        | m     | kg     | ms    | mm     |
        |-------|--------|-------|--------|
        |  2.34 |    123 |  98.7 |     23 |
        | 56.70 |   3457 |  54.3 |     45 |
        |  8.90 |     43 |  67.9 |    345 |
        |---------------------------------|
        ```

        ## `csv`: comma separated values
        ``` plain
        size/m,weight/kg,delay/ms,jitter/mm
        2.34,123,98.7,23
        56.70,3457,54.3,45
        8.90,43,67.9,345
        ```

        ## `rtai`: rtai-style table
        ``` plain
        RTH| info         | reaction     
        RTH| size | weight| delay| jitter
        RTH| m    | kg    | ms   | mm    
        RTD|  2.34|    123|  98.7|     23
        RTD| 56.70|   3457|  54.3|     45
        RTD|  8.90|     43|  67.9|    345
        ```

        ## `md`: markdown
        ``` plain
        | size/m | weight/kg | delay/ms | jitter/mm |
        |------:|-------:|------:|-------:|
        |  2.34 |    123 |  98.7 |     23 |
        | 56.70 |   3457 |  54.3 |     45 |
        |  8.90 |     43 |  67.9 |    345 |
        ```

        ## `tex`: latex tabular
        ``` tex
        \\begin{tabular}{rrrr}
          \\hline
          \\multicolumn{2}{l}{info} & \\multicolumn{2}{l}{reaction} \\
          \\multicolumn{1}{l}{size} & \\multicolumn{1}{l}{weight} & \\multicolumn{1}{l}{delay} & \\multicolumn{1}{l}{jitter} \\
          \\multicolumn{1}{l}{m} & \\multicolumn{1}{l}{kg} & \\multicolumn{1}{l}{ms} & \\multicolumn{1}{l}{mm} \\
          \\hline
          2.34 & 123 & 98.7 & 23 \\
          56.70 & 3457 & 54.3 & 45 \\
          8.90 & 43 & 67.9 & 345 \\
          \\hline
        \\end{tabular}
        ```

        ## `html`: html
        ``` html
        <table>
        <thead>
          <tr class="header">
            <th align="left" colspan="2">info</th>
            <th align="left" colspan="2">reaction</th>
          </tr>
          <tr class="header">
            <th align="left">size</th>
            <th align="left">weight</th>
            <th align="left">delay</th>
            <th align="left">jitter</th>
          </tr>
          <tr class="header">
            <th align="left">m</th>
            <th align="left">kg</th>
            <th align="left">ms</th>
            <th align="left">mm</th>
          </tr>
        </thead>
        <tbody>
          <tr class"odd">
            <td align="right">2.34</td>
            <td align="right">123</td>
            <td align="right">98.7</td>
            <td align="right">23</td>
          </tr>
          <tr class"even">
            <td align="right">56.70</td>
            <td align="right">3457</td>
            <td align="right">54.3</td>
            <td align="right">45</td>
          </tr>
          <tr class"odd">
            <td align="right">8.90</td>
            <td align="right">43</td>
            <td align="right">67.9</td>
            <td align="right">345</td>
          </tr>
        </tbody>
        </table>
        ```

        """
        # fix parameter:
        if table_format == 'auto':
            table_format = None
        if delimiter == 'auto':
            delimiter = None
        if unit_style == 'auto':
            unit_style = None
        if column_numbers == 'none':
            column_numbers = None
        if sections == 'auto':
            sections = None
        if align_columns == 'auto':
            align_columns = None
        # open file:
        own_file = False
        file_name = None
        if not hasattr(fh, 'write'):
            fh = Path(fh)
            ext = fh.suffix
            if table_format is None:
                if len(ext) > 1 and ext[1:] in self.ext_formats:
                    table_format = self.ext_formats[ext[1:]]
            elif not ext or not ext[1:].lower() in self.ext_formats:
                fh = fh.with_suffix('.' + self.extensions[table_format])
            file_name = fh
            try:
                fh = open(os.fspath(fh), 'w')
            except AttributeError:
                fh = open(str(fh), 'w')
            own_file = True
        if table_format is None:
            table_format = 'dat'
        # set style:        
        if table_format[0] == 'd':
            align_columns = True
            begin_str = ''
            end_str = ''
            header_start = '# '
            header_sep = '  '
            header_close = ''
            header_end = '\n'
            data_start = '  '
            data_sep = '  '
            data_close = ''
            data_end = '\n'
            top_line = False
            header_line = False
            bottom_line = False
            if delimiter is not None:
                header_sep = delimiter
                data_sep = delimiter
            if sections is None:
                sections = 1000
        elif table_format[0] == 'a':
            align_columns = True
            begin_str = ''
            end_str = ''
            header_start = '| '
            header_sep = ' | '
            header_close = ''
            header_end = ' |\n'
            data_start = '| '
            data_sep = ' | '
            data_close = ''
            data_end = ' |\n'
            top_line = True
            header_line = True
            bottom_line = True
            if delimiter is not None:
                header_sep = delimiter
                data_sep = delimiter
            if sections is None:
                sections = 1000
        elif table_format[0] == 'c':
            # csv according to http://www.ietf.org/rfc/rfc4180.txt :
            column_numbers=None
            if unit_style is None:
                unit_style = 'header'
            if align_columns is None:
                align_columns = False
            begin_str = ''
            end_str = ''
            header_start=''
            header_sep = ','
            header_close = ''
            header_end='\n'
            data_start=''
            data_sep = ','
            data_close = ''
            data_end='\n'
            top_line = False
            header_line = False
            bottom_line = False
            if delimiter is not None:
                header_sep = delimiter
                data_sep = delimiter
            if sections is None:
                sections = 0
        elif table_format[0] == 'r':
            align_columns = True
            begin_str = ''
            end_str = ''
            header_start = 'RTH| '
            header_sep = '| '
            header_close = ''
            header_end = '\n'
            data_start = 'RTD| '
            data_sep = '| '
            data_close = ''
            data_end = '\n'
            top_line = False
            header_line = False
            bottom_line = False
            if sections is None:
                sections = 1000
        elif table_format[0] == 'm':
            if unit_style is None or unit_style == 'row':
                unit_style = 'header'
            align_columns = True
            begin_str = ''
            end_str = ''
            header_start='| '
            header_sep = ' | '
            header_close = ''
            header_end=' |\n'
            data_start='| '
            data_sep = ' | '
            data_close = ''
            data_end=' |\n'
            top_line = False
            header_line = True
            bottom_line = False
            if sections is None:
                sections = 0
        elif table_format[0] == 'h':
            align_columns = False
            begin_str = '<table>\n<thead>\n'
            end_str = '</tbody>\n</table>\n'
            if center_columns:
                header_start='  <tr>\n    <th align="center"'
                header_sep = '</th>\n    <th align="center"'
            else:
                header_start='  <tr>\n    <th align="left"'
                header_sep = '</th>\n    <th align="left"'
            header_close = '>'
            header_end='</th>\n  </tr>\n'
            data_start='  <tr>\n    <td'
            data_sep = '</td>\n    <td'
            data_close = '>'
            data_end='</td>\n  </tr>\n'
            top_line = False
            header_line = False
            bottom_line = False
            if sections is None:
                sections = 1000
        elif table_format[0] == 't':
            if align_columns is None:
                align_columns = False
            begin_str = '\\begin{tabular}'
            end_str = '\\end{tabular}\n'
            header_start='  '
            header_sep = ' & '
            header_close = ''
            header_end=' \\\\\n'
            data_start='  '
            data_sep = ' & '
            data_close = ''
            data_end=' \\\\\n'
            top_line = True
            header_line = True
            bottom_line = True
            if sections is None:
                sections = 1000
        else:
            if align_columns is None:
                align_columns = True
            begin_str = ''
            end_str = ''
            header_start = ''
            header_sep = '  '
            header_close = ''
            header_end = '\n'
            data_start = ''
            data_sep = '  '
            data_close = ''
            data_end = '\n'
            top_line = False
            header_line = False
            bottom_line = False
            if sections is None:
                sections = 1000
        # check units:
        if unit_style is None:
            unit_style = 'row'
        have_units = False
        for u in self.units:
            if u and u != '1' and u != '-':
                have_units = True
                break
        if not have_units:
            unit_style = 'none'
        # find std columns:
        stdev_col = np.zeros(len(self.header), dtype=bool)
        for c in range(len(self.header) - 1):
            if self.header[c+1][0].lower() in self.stdev_labels and \
               not self.hidden[c+1]:
                stdev_col[c] = True
        # begin table:
        fh.write(begin_str)
        if table_format[0] == 't':
            fh.write('{')
            merged = False
            for h, f, s in zip(self.hidden, self.formats, stdev_col):
                if merged:
                    fh.write('l')
                    merged = False
                    continue
                if h:
                    continue
                if latex_merge_std and s:
                    fh.write('r@{$\\,\\pm\\,$}')
                    merged = True
                elif center_columns:
                    fh.write('c')
                elif f[1] == '-':
                    fh.write('l')
                else:
                    if latex_unit_package is not None and \
                       latex_unit_package.lower() == 'siunitx':
                        fh.write('S')
                    else:
                        fh.write('r')
            fh.write('}\n')
        # retrieve column formats and widths:
        widths = []
        widths_pos = []
        for c, f in enumerate(self.formats):
            w = 0
            # position of width specification:
            i0 = 1
            if len(f) > 1 and f[1] == '-' :
                i0 = 2
            i1 = f.find('.')
            if not shrink_width:
                if f[i0:i1]:
                    w = int(f[i0:i1])
            widths_pos.append((i0, i1))
            # adapt width to header label:
            hw = len(self.header[c][0])
            if unit_style == 'header' and self.units[c] and\
               self.units[c] != '1' and self.units[c] != '-':
                hw += 1 + len(self.units[c])
            if w < hw:
                w = hw
            # adapt width to data:
            if f[-1] == 's':
                for v in self.data[c]:
                    if isinstance(v, str) and w < len(v):
                        w = len(v)
            else:
                fs = f[:i0] + str(0) + f[i1:]
                for v in self.data[c]:
                    if v is None or (isinstance(v, (float, np.floating)) and m.isnan(v)):
                        s = missing
                    else:
                        try:
                            s = fs % v
                        except ValueError:
                            s = missing
                        except TypeError:
                            s = str(v)
                    if w < len(s):
                        w = len(s)
            widths.append(w)
        # adapt width to sections:
        sec_indices = [0] * self.nsecs
        sec_widths = [0] * self.nsecs
        sec_columns = [0] * self.nsecs
        for c in range(len(self.header)):
            w = widths[c]
            for l in range(min(self.nsecs, sections)):
                if 1+l < len(self.header[c]):
                    if c > 0 and sec_columns[l] > 0 and \
                       1+l < len(self.header[sec_indices[l]]) and \
                       len(self.header[sec_indices[l]][1+l]) > sec_widths[l]:
                        dw = len(self.header[sec_indices[l]][1+l]) - sec_widths[l]
                        nc = sec_columns[l]
                        ddw = np.zeros(nc, dtype=int) + dw // nc
                        ddw[:dw % nc] += 1
                        wk = 0
                        for ck in range(sec_indices[l], c):
                            if not self.hidden[ck]:
                                widths[ck] += ddw[wk]
                                wk += 1
                    sec_widths[l] = 0
                    sec_indices[l] = c
                if not self.hidden[c]:
                    if sec_widths[l] > 0:
                        sec_widths[l] += len(header_sep)
                    sec_widths[l] += w
                    sec_columns[l] += 1
        # set width of format string:
        formats = []
        for c, (f, w) in enumerate(zip(self.formats, widths)):
            formats.append(f[:widths_pos[c][0]] + str(w) + f[widths_pos[c][1]:])
        # top line:
        if top_line:
            if table_format[0] == 't':
                fh.write('  \\hline \\\\[-2ex]\n')
            else:
                first = True
                fh.write(header_start.replace(' ', '-'))
                for c in range(len(self.header)):
                    if self.hidden[c]:
                        continue
                    if not first:
                        fh.write('-'*len(header_sep))
                    first = False
                    fh.write(header_close)
                    w = widths[c]
                    fh.write(w*'-')
                fh.write(header_end.replace(' ', '-'))
        # section and column headers:
        nsec0 = self.nsecs - sections
        if nsec0 < 0:
            nsec0 = 0
        for ns in range(nsec0, self.nsecs+1):
            nsec = self.nsecs - ns
            first = True
            last = False
            merged = False
            fh.write(header_start)
            for c in range(len(self.header)):
                if nsec < len(self.header[c]):
                    # section width and column count:
                    sw = -len(header_sep)
                    columns = 0
                    if not self.hidden[c]:
                        sw = widths[c]
                        columns = 1
                    for k in range(c+1, len(self.header)):
                        if nsec < len(self.header[k]):
                            break
                        if self.hidden[k]:
                            continue
                        sw += len(header_sep) + widths[k]
                        columns += 1
                    else:
                        last = True
                        if len(header_end.strip()) == 0:
                            sw = 0  # last entry needs no width
                    if columns == 0:
                        continue
                    if not first and not merged:
                        fh.write(header_sep)
                    first = False
                    if table_format[0] == 'c':
                        sw -= len(header_sep)*(columns - 1)
                    elif table_format[0] == 'h':
                        if columns>1:
                            fh.write(' colspan="%d"' % columns)
                    elif table_format[0] == 't':
                        if merged:
                            merged = False
                            continue
                        if latex_merge_std and nsec == 0 and stdev_col[c]:
                            merged = True
                            fh.write('\\multicolumn{%d}{c}{' % (columns+1))
                        elif center_columns:
                            fh.write('\\multicolumn{%d}{c}{' % columns)
                        else:
                            fh.write('\\multicolumn{%d}{l}{' % columns)
                        if latex_label_command:
                            fh.write('\\%s{' % latex_label_command)
                    fh.write(header_close)
                    hs = self.header[c][nsec]
                    if nsec == 0 and unit_style == 'header':
                        if self.units[c] and self.units[c] != '1' and self.units[c] != '-':
                            hs += '/' + self.units[c]
                    if align_columns and not table_format[0] in 'th':
                        f = '%%-%ds' % sw
                        fh.write(f % hs)
                    else:
                        fh.write(hs)
                    if table_format[0] == 'c':
                        if not last:
                            fh.write(header_sep*(columns - 1))
                    elif table_format[0] == 't':
                        if latex_label_command:
                            fh.write('}')
                        fh.write('}')
            fh.write(header_end)
        # units:
        if unit_style == 'row':
            first = True
            merged = False
            fh.write(header_start)
            for c in range(len(self.header)):
                if self.hidden[c] or merged:
                    merged = False
                    continue
                if not first:
                    fh.write(header_sep)
                first = False
                fh.write(header_close)
                unit = self.units[c]
                if not unit:
                    unit = '-'
                if table_format[0] == 't':
                    if latex_merge_std and stdev_col[c]:
                        merged = True
                        fh.write('\\multicolumn{2}{c}{%s}' % latex_unit(unit, latex_unit_package))
                    elif center_columns:
                        fh.write('\\multicolumn{1}{c}{%s}' % latex_unit(unit, latex_unit_package))
                    else:
                        fh.write('\\multicolumn{1}{l}{%s}' % latex_unit(unit, latex_unit_package))
                else:
                    if align_columns and not table_format[0] in 'h':
                        f = '%%-%ds' % widths[c]
                        fh.write(f % unit)
                    else:
                        fh.write(unit)
            fh.write(header_end)
        # column numbers:
        if column_numbers is not None:
            first = True
            fh.write(header_start)
            for c in range(len(self.header)):
                if self.hidden[c]:
                    continue
                if not first:
                    fh.write(header_sep)
                first = False
                fh.write(header_close)
                i = c
                if column_numbers == 'num':
                    i = c+1
                aa = index2aa(c, 'a')
                if column_numbers == 'AA':
                    aa = index2aa(c, 'A')
                if table_format[0] == 't':
                    if column_numbers == 'num' or column_numbers == 'index':
                        fh.write('\\multicolumn{1}{l}{%d}' % i)
                    else:
                        fh.write('\\multicolumn{1}{l}{%s}' % aa)
                else:
                    if column_numbers == 'num' or column_numbers == 'index':
                        if align_columns:
                            f = '%%%dd' % widths[c]
                            fh.write(f % i)
                        else:
                            fh.write('%d' % i)
                    else:
                        if align_columns:
                            f = '%%-%ds' % widths[c]
                            fh.write(f % aa)
                        else:
                            fh.write(aa)
            fh.write(header_end)
        # header line:
        if header_line:
            if table_format[0] == 'm':
                fh.write('|')
                for c in range(len(self.header)):
                    if self.hidden[c]:
                        continue
                    w = widths[c]+2
                    if center_columns:
                        fh.write(':' + (w-2)*'-' + ':|')
                    elif formats[c][1] == '-':
                        fh.write(w*'-' + '|')
                    else:
                        fh.write((w - 1)*'-' + ':|')
                fh.write('\n')
            elif table_format[0] == 't':
                fh.write('  \\hline \\\\[-2ex]\n')
            else:
                first = True
                fh.write(header_start.replace(' ', '-'))
                for c in range(len(self.header)):
                    if self.hidden[c]:
                        continue
                    if not first:
                        fh.write(header_sep.replace(' ', '-'))
                    first = False
                    fh.write(header_close)
                    w = widths[c]
                    fh.write(w*'-')
                fh.write(header_end.replace(' ', '-'))
        # start table data:
        if table_format[0] == 'h':
            fh.write('</thead>\n<tbody>\n')
        # data:
        for k in range(self.rows()):
            first = True
            merged = False
            fh.write(data_start)
            for c, f in enumerate(formats):
                if self.hidden[c] or merged:
                    merged = False
                    continue
                if not first:
                    fh.write(data_sep)
                first = False
                if table_format[0] == 'h':
                    if center_columns:
                        fh.write(' align="center"')
                    elif f[1] == '-':
                        fh.write(' align="left"')
                    else:
                        fh.write(' align="right"')
                fh.write(data_close)
                if k >= len(self.data[c]) or self.data[c][k] is None or \
                   (isinstance(self.data[c][k], (float, np.floating)) and m.isnan(self.data[c][k])):
                    # missing data:
                    if table_format[0] == 't' and latex_merge_std and stdev_col[c]:
                        merged = True
                        fh.write('\\multicolumn{2}{c}{%s}' % missing)
                    elif align_columns:
                        if f[1] == '-':
                            fn = '%%-%ds' % widths[c]
                        else:
                            fn = '%%%ds' % widths[c]
                        fh.write(fn % missing)
                    else:
                        fh.write(missing)
                else:
                    # data value:
                    try:
                        ds = f % self.data[c][k]
                    except ValueError:
                        ds = missing
                    except TypeError:
                        ds = str(self.data[c][k])
                    if not align_columns:
                        ds = ds.strip()
                    fh.write(ds)
            fh.write(data_end)
        # bottom line:
        if bottom_line:
            if table_format[0] == 't':
                fh.write('  \\hline\n')
            else:
                first = True
                fh.write(header_start.replace(' ', '-'))
                for c in range(len(self.header)):
                    if self.hidden[c]:
                        continue
                    if not first:
                        fh.write('-'*len(header_sep))
                    first = False
                    fh.write(header_close)
                    w = widths[c]
                    fh.write(w*'-')
                fh.write(header_end.replace(' ', '-'))
        # end table:
        fh.write(end_str)
        # close file:
        if own_file:
            fh.close()
        # write descriptions:
        if file_name is not None and descriptions_name:
            write_descriptions = False
            for c in range(len(self.descriptions)):
                if self.descriptions[c]:
                    write_descriptions = True
            if write_descriptions:
                descr_path = file_name.with_name(file_name.stem +
                                                 descriptions_name)
                self.write_descriptions(descr_path, table_format=None,
                                        sections=sections,
                                        section_headings=section_headings,
                                        latex_unit_package=latex_unit_package,
                                        maxc=maxc)
        # return file name:
        return file_name


    def write_file_stream(self, basename, file_name, **kwargs):
        """Write table to file or stream and return appropriate file name.

        Parameters
        ----------
        basename: str or stream
            If str, path and basename of file.
            `file_name` and an extension are appended.
            If stream, write table data into this stream.
        file_name: str
            Name of file that is appended to a base path or `basename`.
        kwargs:
            Arguments passed on to `TableData.write()`.
            In particular, 'table_format' is used to set the file extension
            that is appended to the returned `file_name`.

        Returns
        -------
        file_name: str
            Path and full name of the written file in case of `basename`
            being a string. Otherwise, the file name and extension that
            should be appended to a base path.
        """
        if hasattr(basename, 'write'):
            table_format = kwargs.get('table_format', None)
            if table_format is None or table_format == 'auto':
                table_format = 'csv'
            file_name += '.' + TableData.extensions[table_format]
            self.write(basename, **kwargs)
            return file_name
        else:
            file_name = self.write(basename + file_name, **kwargs)
            return file_name

    def __str__(self):
        """Write table to a string.
        """
        stream = StringIO()
        self.write(stream, table_format='out')
        return stream.getvalue()
                
    def write_descriptions(self, fh=sys.stdout, table_format=None,
                           sections=None, section_headings=None,
                           latex_unit_package=None, maxc=80):
        """Write column descriptions of the table to a file or stream.

        Parameters
        ----------
        fh: filename or stream
            If not a stream, the file with path `fh` is opened.
            If `fh` does not have an extension,
            the `table_format` is appended as an extension.
            Otherwise `fh` is used as a stream for writing.
        table_format: None or str
            The format to be used for output.
            One of 'md', 'tex', or 'html'.
            If None or 'auto' then the format is set to the extension
            of the filename given by `fh`.
            If `fh` is a stream the format is set to 'md'.
        sections: None or int
            Number of section levels to be printed.
            If `None` or 'auto' use default of selected `table_format`.
        section_headings: None or int
            If set, set header sections as headings with the top-level
            section at the level as specified. 0 is the top level.
            If False, just produce a nested list.
        latex_unit_package: None or 'siunitx' or 'SIunit'
            Translate units for the specified LaTeX package.
            If None set sub- and superscripts in text mode.
            If 'siunitx', also use `S` columns for numbers to align
            them on the decimal point.
        maxc: int
            Maximum character count for each line.
        """
        # fix parameter:
        if table_format == 'auto':
            table_format = None
        if sections is None:
            sections = 1000
        nsecs = min(self.nsecs, sections)
        # open file:
        own_file = False
        file_name = None
        if not hasattr(fh, 'write'):
            fh = Path(fh)
            ext = fh.suffix
            if table_format is None:
                if len(ext) > 1 and ext[1:] in self.ext_formats:
                    table_format = self.ext_formats[ext[1:]]
            elif not ext or not ext[1:].lower() in self.ext_formats:
                fh = fh.with_suffix('.' + self.extensions[table_format])
            file_name = fh
            try:
                fh = open(os.fspath(fh), 'w')
            except AttributeError:
                fh = open(str(fh), 'w')
            own_file = True
        if table_format is None:
            table_format = 'md'
        # write descriptions:
        headers = ['']*(1 + nsecs)
        prev_headers = ['']*(1 + nsecs)
        if table_format == 'md':
            for c in range(len(self.header)):
                headers[:len(self.header[c])] = self.header[c]
                if not self.hidden[c]:
                    changed = False
                    for k in reversed(range(nsecs)):
                        if changed or prev_headers[k + 1] != headers[k + 1]:
                            changed = True
                            if section_headings is None:
                                fh.write(f'{" "*2*(nsecs - k - 1)}- '
                                         f'{headers[k + 1]}\n')
                            else:
                                level = nsecs - k - 1 + section_headings + 1
                                fh.write(f'\n{"#"*level} {headers[k + 1]}\n')
                            prev_headers[k + 1] = headers[k + 1]
                    indent = 2*nsecs if section_headings is None else 0
                    fh.write(f'{" "*indent}- **{headers[0]}**')
                    if self.units[c]:
                        fh.write(f' [{self.units[c]}]')
                    fh.write('  \n')
                    break_text(fh, self.descriptions[c], maxc,
                               indent=indent + 2)
                    prev_headers[0] = headers[0]
        elif table_format == 'html':
            level = -1
            for c in range(len(self.header)):
                headers[:len(self.header[c])] = self.header[c]
                if not self.hidden[c]:
                    changed = False
                    for k in reversed(range(nsecs)):
                        if changed or prev_headers[k + 1] != headers[k + 1]:
                            new_level = nsecs - k - 1
                            if not changed:
                                if section_headings is None:
                                    while level > new_level:
                                        fh.write(f'{" "*2*level}</ul>\n')
                                        level -= 1
                                elif level >= 0:
                                    fh.write(f'{" "*2*level}</ul>\n')
                                    level -= 1
                            changed = True
                            if section_headings is None:
                                while level < new_level:
                                    level += 1
                                    fh.write(f'{" "*2*level}<ul>\n')
                                fh.write(f'{" "*2*(level + 1)}<li><b>{headers[k + 1]}</b></li>\n')
                            else:
                                fh.write(f'\n<h{new_level + section_headings + 1}>{headers[k + 1]}</h{new_level + section_headings + 1}>\n')
                            prev_headers[k + 1] = headers[k + 1]
                    if changed:
                        level += 1
                        fh.write(f'{" "*2*level}<ul>\n')
                        
                    fh.write(f'{" "*2*(level + 1)}<li><b>{headers[0]}</b>')
                    if self.units[c]:
                        fh.write(f'[{self.units[c]}]')
                    fh.write('<br>\n')
                    break_text(fh, self.descriptions[c], maxc,
                               indent=2*(level + 1))
                    fh.write(f'{" "*2*(level + 1)}</li>\n')
                    prev_headers[0] = headers[0]
            while level >= 0:
                fh.write(f'{" "*2*level}</ul>\n')
                level -= 1
        elif table_format == 'tex':
            headings = [r'\section', r'\subsection', r'\subsubsection',
                        r'\paragraph', r'\subparagraph']
            level = -1
            for c in range(len(self.header)):
                headers[:len(self.header[c])] = self.header[c]
                if not self.hidden[c]:
                    changed = False
                    for k in reversed(range(nsecs)):
                        if changed or prev_headers[k + 1] != headers[k + 1]:
                            new_level = nsecs - k - 1
                            if not changed:
                                if section_headings is None:
                                    while level > new_level:
                                        fh.write(f'{" "*2*level}\\end{{enumerate}}\n')
                                        level -= 1
                                elif level >= 0:
                                    fh.write(f'{" "*2*level}\\end{{enumerate}}\n')
                                    level -= 1
                            changed = True
                            if section_headings is None:
                                while level < new_level:
                                    level += 1
                                    fh.write(f'{" "*2*level}\\begin{{enumerate}}\n')
                                fh.write(f'{" "*2*(level + 1)}\\item \\textbf{{{headers[k + 1]}}}\n')
                            else:
                                fh.write(f'\n{headings[new_level + section_headings]}{{{headers[k + 1]}}}\n')
                            prev_headers[k + 1] = headers[k + 1]
                    if changed:
                        level += 1
                        fh.write(f'{" "*2*level}\\begin{{enumerate}}\n')
                    fh.write(f'{" "*2*(level + 1)}\\item \\textbf{{{headers[0]}}}')
                    if self.units[c]:
                        fh.write(f' [{latex_unit(self.units[c],
                                                 latex_unit_package)}]')
                    fh.write('\n')
                    break_text(fh, self.descriptions[c], maxc,
                               indent=2*(level + 1))
                    prev_headers[0] = headers[0]
            while level >= 0:
                fh.write(f'{" "*2*level}\\end{{enumerate}}\n')
                level -= 1
        else:
            raise ValueError(f'File format "{table_format}" not supported for writing column descriptions')
        # close file:
        if own_file:
            fh.close()
        # return file name:
        return file_name
        
    def load(self, fh, missing=default_missing_inputs, sep=None, stop=None):
        """Load table from file or stream.

        File type and properties are automatically inferred.

        Parameters
        ----------
        fh: str, Path, or stream
            If not a stream, the file with path `fh` is opened for reading.
        missing: str or list of str
            Missing data are indicated by this string and
            are translated to np.nan.
        sep: str or None
            Column separator.
        stop: str or None
            If a line matches `stop`, stop reading the file.  `stop`
            can be an empty string to stop reading at the first empty
            line.

        Raises
        ------
        FileNotFoundError:
            If `fh` is a path that does not exist.

        """

        def read_key_line(line, sep, table_format):
            if sep is None:
                cols, indices = zip(*[(m.group(0), m.start()) for m in re.finditer(r'( ?[\S]+)+(?=[ ][ ]+|\Z)', line.strip())])
            elif table_format == 'csv':
                cols, indices = zip(*[(c.strip(), i) for i, c in enumerate(line.strip().split(sep)) if c.strip()])
            else:
                seps = r'[^'+re.escape(sep)+']+'
                cols, indices = zip(*[(m.group(0), m.start()) for m in re.finditer(seps, line.strip())])
            colss = []
            indicess = []
            if table_format == 'tex':
                i = 0
                for c in cols:
                    if 'multicolumn' in c:
                        fields = c.split('{')
                        n = int(fields[1].strip().rstrip('}').rstrip())
                        colss.append(fields[3].strip().rstrip('}').rstrip())
                        indicess.append(i)
                        i += n
                    else:
                        colss.append(c.strip())
                        indicess.append(i)
                        i += 1
            else:
                for k, (c, i) in enumerate(zip(cols, indices)):
                    if table_format != 'csv':
                        if k == 0:
                            c = c.lstrip('|')
                        if k == len(cols) - 1:
                            c = c.rstrip('|')
                    cs = c.strip()
                    if len(cs) >= 2 and cs[0] == '"' and cs[-1] == '"':
                        cs = cs.strip('"')
                    colss.append(cs)
                    indicess.append(i)
            return colss, indicess

        def read_data_line(line, sep, post, precd, alld, numc, exped,
                           fixed, strf, missing, nans):
            # read line:
            cols = []
            if sep is None:
                cols = [m.group(0) for m in re.finditer(r'\S+', line.strip())]
            else:
                if sep.isspace():
                    seps = r'[^'+re.escape(sep)+']+'
                    cols = [m.group(0) for m in re.finditer(seps, line.strip())]
                else:
                    cols = line.split(sep)
                    if len(cols) > 0 and len(cols[0]) == 0:
                        cols = cols[1:]
                    if len(cols) > 0 and len(cols[-1]) == 0:
                        cols = cols[:-1]
                if len(cols) > 0:
                    cols[0] = cols[0].lstrip('|').lstrip()
                    cols[-1] = cols[-1].rstrip('|').rstrip()
            cols = [c.strip() for c in cols if c != '|']
            # read columns:
            for k, c in enumerate(cols):
                try:
                    v = float(c)
                    ad = 0
                    ve = c.split('e')
                    if len(ve) <= 1:
                        exped[k] = False
                    else:
                        ad = len(ve[1])+1
                    vc = ve[0].split('.')
                    ad += len(vc[0])
                    prec = len(vc[0].lstrip('-').lstrip('+').lstrip('0')) 
                    if len(vc) == 2:
                        if numc[k] and post[k] != len(vc[1]):
                            fixed[k] = False
                        if post[k] < len(vc[1]):
                            post[k] = len(vc[1])
                        ad += len(vc[1])+1
                        prec += len(vc[1].rstrip('0'))
                    if precd[k] < prec:
                        precd[k] = prec
                    if alld[k] < ad:
                        alld[k] = ad
                    numc[k] = True
                except ValueError:
                    if c in missing:
                        v = np.nan
                        nans[k] = c
                    elif len(c) == 0 and not strf[k]:
                        v = np.nan
                    else:
                        strf[k] = True
                        if alld[k] < len(c):
                            alld[k] = len(c)
                        if len(c) >= 2 and c[0] == '"' and c[-1] == '"':
                            v = c.strip('"')
                        else:
                            v = c
                self.add(v, k)
            self.fill_data()

        # initialize:
        if isinstance(missing, str):
            missing = [missing]
        self.data = []
        self.ndim = 2
        self.shape = (0, 0)
        self.header = []
        self.nsecs = 0
        self.units = []
        self.formats = []
        self.descriptions = []
        self.hidden = []
        self.setcol = 0
        self.addcol = 0
        # open file:
        own_file = False
        if not hasattr(fh, 'readline'):
            try:
                fh = open(os.fspath(fh), 'r')
            except AttributeError:
                fh = open(str(fh), 'r')
            own_file = True
        # read inital lines of file:
        key = []
        data = []
        target = data
        comment = False
        table_format='dat'        
        for line in fh:
            line = line.rstrip()
            if line == stop:
                break;
            if line:
                if r'\begin{tabular' in line:
                    table_format='tex'
                    target = key
                    continue
                if table_format == 'tex':
                    if r'\end{tabular' in line:
                        break
                    if r'\hline' in line:
                        if key:
                            target = data
                        continue
                    line = line.rstrip(r'\\')
                if line[0] == '#':
                    comment = True
                    table_format='dat'        
                    target = key
                    line = line.lstrip('#')
                elif comment:
                    target = data
                if line[0:3] == 'RTH':
                    target = key
                    line = line[3:]
                    table_format='rtai'
                elif line[0:3] == 'RTD':
                    target = data
                    line = line[3:]
                    table_format='rtai'        
                if (line[0:3] == '|--' or line[0:3] == '|:-') and \
                   (line[-3:] == '--|' or line[-3:] == '-:|'):
                    if not data and not key:
                        table_format='ascii'
                        target = key
                        continue
                    elif not key:
                        table_format='md'
                        key = data
                        data = []
                        target = data
                        continue
                    elif not data:
                        target = data
                        continue
                    else:
                        break
                target.append(line)
            else:
                break
            if len(data) > 5:
                break
        # find column separator of data and number of columns:
        col_seps = ['|', ',', ';', ':', '\t', '&', None]
        if sep is not None:
            col_seps = [sep]
        colstd = np.zeros(len(col_seps))
        colnum = np.zeros(len(col_seps), dtype=int)
        for k, sep in enumerate(col_seps):
            cols = []
            s = 5 if len(data) >= 8 else len(data) - 3
            if s < 0 or key:
                s = 0
            for line in data[s:]:
                cs = line.strip().split(sep)
                if not cs[0]:
                    cs = cs[1:]
                if cs and not cs[-1]:
                    cs = cs[:-1]
                cols.append(len(cs))
            colstd[k] = np.std(cols)
            colnum[k] = np.median(cols)
        if np.max(colnum) < 2:
            sep = None
            colnum = 1
        else:
            ci = np.where(np.array(colnum) > 1.5)[0]
            ci = ci[np.argmin(colstd[ci])]
            sep = col_seps[ci]
            colnum = int(colnum[ci])
        # fix key:
        if not key and sep is not None and sep in ',;:\t|':
            table_format = 'csv'
        # read key:
        key_cols = []
        key_indices = []
        for line in key:
            cols, indices = read_key_line(line, sep, table_format)
            key_cols.append(cols)
            key_indices.append(indices)
        if not key_cols:
            # no obviously marked table key:
            key_num = 0
            for line in data:
                cols, indices = read_key_line(line, sep, table_format)
                numbers = 0
                for c in cols:
                    try:
                        v = float(c)
                        numbers += 1
                    except ValueError:
                        break
                if numbers == 0:
                    key_cols.append(cols)
                    key_indices.append(indices)
                    key_num += 1
                else:
                    break
            if len(key_cols) == len(data):
                key_num = 1
                key_cols = key_cols[:1]
                key_indices = key_indices[:1]
                colnum = len(key_cols[0])
            data = data[key_num:]
        kr = len(key_cols) - 1
        # check for key with column indices:
        if kr >= 0:
            cols = key_cols[kr]
            numrow = True
            try:
                pv = int(cols[0])
                for c in cols[1:]:
                    v = int(c)
                    if v != pv+1:
                        numrow = False
                        break
                    pv = v
            except ValueError:
                try:
                    pv = aa2index(cols[0])
                    for c in cols[1:]:
                        v = aa2index(c)
                        if v != pv+1:
                            numrow = False
                            break
                        pv = v
                except ValueError:
                    numrow = False
            if numrow:
                kr -= 1
        # check for unit line:
        units = None
        if kr > 0 and len(key_cols[kr]) == len(key_cols[kr - 1]):
            units = key_cols[kr]
            kr -= 1
        # column labels:
        if kr >= 0:
            if units is None:
                # units may be part of the label:
                labels = []
                units = []
                for c in key_cols[kr]:
                    if c[-1] == ')':
                        lu = c[:-1].split('(')
                        if len(lu) >= 2:
                            labels.append(lu[0].strip())
                            units.append('('.join(lu[1:]).strip())
                            continue
                    lu = c.split('/')
                    if len(lu) >= 2:
                        labels.append(lu[0].strip())
                        units.append('/'.join(lu[1:]).strip())
                    else:
                        labels.append(c)
                        units.append('')
            else:
                labels = key_cols[kr]
            indices = key_indices[kr]
            # init table columns:
            for k in range(colnum):
                self.append(labels[k], units[k], '%g')
        # read in sections:
        while kr > 0:
            kr -= 1
            for sec_label, sec_inx in zip(key_cols[kr], key_indices[kr]):
                col_inx = indices.index(sec_inx)
                self.header[col_inx].append(sec_label)
                if self.nsecs < len(self.header[col_inx]) - 1:
                    self.nsecs = len(self.header[col_inx]) - 1
        # read data:
        post = np.zeros(colnum, dtype=int)
        precd = np.zeros(colnum, dtype=int)
        alld = np.zeros(colnum, dtype=int)
        numc = [False] * colnum
        exped = [True] * colnum
        fixed = [True] * colnum
        strf = [False] * colnum
        nans = [None] * colnum   # for each column the missing string that was encountered.
        for line in data:
            read_data_line(line, sep, post, precd, alld, numc, exped, fixed,
                           strf, missing, nans)
        # read remaining data:
        for line in fh:
            line = line.rstrip()
            if line == stop:
                break;
            if table_format == 'tex':
                if r'\end{tabular' in line or r'\hline' in line:
                    break
                line = line.rstrip(r'\\')
            if (line[0:3] == '|--' or line[0:3] == '|:-') and \
                (line[-3:] == '--|' or line[-3:] == '-:|'):
                break
            if line[0:3] == 'RTD':
                line = line[3:]
            read_data_line(line, sep, post, precd, alld, numc, exped, fixed,
                           strf, missing, nans)
        # set formats:
        for k in range(len(alld)):
            if strf[k]:
                self.set_format(k, '%%-%ds' % alld[k])
                # make sure all elements are strings:
                for i in range(len(self.data[k])):
                    if self.data[k][i] is np.nan:
                        self.data[k][i] = nans[k]
                    else:
                        self.data[k][i] = str(self.data[k][i])
            elif exped[k]:
                self.set_format(k, '%%%d.%de' % (alld[k], post[k]))
            elif fixed[k]:
                self.set_format(k, '%%%d.%df' % (alld[k], post[k]))
            else:
                self.set_format(k, '%%%d.%dg' % (alld[k], precd[k]))
        # close file:
        if own_file:
            fh.close()

    
def add_write_table_config(cfg, table_format=None, delimiter=None,
                           unit_style=None, column_numbers=None,
                           sections=None, align_columns=None,
                           shrink_width=True, missing='-',
                           center_columns=False,
                           latex_label_command='',
                           latex_merge_std=False):
    """Add parameter specifying how to write a table to a file as a new
section to a configuration.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
    """

    cfg.add_section('File format for storing analysis results:')
    cfg.add('fileFormat', table_format or 'auto', '', 'Default file format used to store analysis results.\nOne of %s.' % ', '.join(TableData.formats))
    cfg.add('fileDelimiter', delimiter or 'auto', '', 'String used to separate columns or "auto".')
    cfg.add('fileUnitStyle', unit_style or 'auto', '', 'Add units as extra row ("row"), add units to header label separated by "/" ("header"), do not print out units ("none"), or "auto".')
    cfg.add('fileColumnNumbers', column_numbers or 'none', '', 'Add line with column indices ("index", "num", "aa", "AA", or "none")')
    cfg.add('fileSections', sections or 'auto', '', 'Maximum number of section levels or "auto"')
    cfg.add('fileAlignColumns', align_columns or 'auto', '', 'If True, write all data of a column using the same width, if False write the data without any white space, or "auto".')
    cfg.add('fileShrinkColumnWidth', shrink_width, '', 'Allow to make columns narrower than specified by the corresponding format strings.')
    cfg.add('fileMissing', missing, '', 'String used to indicate missing data values.')
    cfg.add('fileCenterColumns', center_columns, '', 'Center content of all columns instead of left align columns of strings and right align numbers (markdown, html, and latex).')
    cfg.add('fileLaTeXLabelCommand', latex_label_command, '', 'LaTeX command name for formatting column labels of the table header.')
    cfg.add('fileLaTeXMergeStd', latex_merge_std, '', 'Merge header of columns with standard deviations with previous column (LaTeX tables only).')


def write_table_args(cfg):
    """Translates a configuration to the respective parameter names for
writing a table to a file.
    
    The return value can then be passed as key-word arguments to TableData.write().

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.

    Returns
    -------
    a: dict
        Dictionary with names of arguments of the `TableData.write` function
        and their values as supplied by `cfg`.
    """
    d = cfg.map({'table_format': 'fileFormat',
                 'delimiter': 'fileDelimiter',
                 'unit_style': 'fileUnitStyle',
                 'column_numbers': 'fileColumnNumbers',
                 'sections': 'fileSections',
                 'align_columns': 'fileAlignColumns',
                 'shrink_width': 'fileShrinkColumnWidth',
                 'missing': 'fileMissing',
                 'center_columns': 'fileCenterColumns',
                 'latex_label_command': 'fileLaTeXLabelCommand',
                 'latex_merge_std': 'fileLaTeXMergeStd'})
    if 'sections' in d:
        if d['sections'] != 'auto':
            d['sections'] = int(d['sections'])
    return d


def latex_unit(unit, unit_package=None):
    """Translate unit string into LaTeX code.
    
    Parameters
    ----------
    unit: str
        String denoting a unit.
    unit_package: None or 'siunitx' or 'SIunit'
        Translate unit string for the specified LaTeX package.
        If None set sub- and superscripts in text mode.
        
    Returns
    -------
    unit: str
        Unit string as valid LaTeX code.
    """
    si_prefixes = {'y': '\\yocto',
                  'z': '\\zepto',
                  'a': '\\atto',
                  'f': '\\femto',
                  'p': '\\pico',
                  'n': '\\nano',
                  'u': '\\micro',
                  'm': '\\milli',
                  'c': '\\centi',
                  'd': '\\deci',
                  'h': '\\hecto',
                  'k': '\\kilo',
                  'M': '\\mega',
                  'G': '\\giga',
                  'T': '\\tera',
                  'P': '\\peta',
                  'E': '\\exa',
                  'Z': '\\zetta',
                  'Y': '\\yotta' }
    si_units = {'m': '\\metre',
               'g': '\\gram',
               's': '\\second',
               'A': '\\ampere',
               'K': '\\kelvin',
               'mol': '\\mole',
               'M': '\\mole',
               'cd': '\\candela',
               'Hz': '\\hertz',
               'N': '\\newton',
               'Pa': '\\pascal',
               'J': '\\joule',
               'W': '\\watt',
               'C': '\\coulomb',
               'V': '\\volt',
               'F': '\\farad',
               'O': '\\ohm',
               'S': '\\siemens',
               'Wb': '\\weber',
               'T': '\\tesla',
               'H': '\\henry',
               'C': '\\celsius',
               'lm': '\\lumen',
               'lx': '\\lux',
               'Bq': '\\becquerel',
               'Gv': '\\gray',
               'Sv': '\\sievert'}
    other_units = {"'": '\\arcminute',
               "''": '\\arcsecond',
               'a': '\\are',
               'd': '\\dday',
               'eV': '\\electronvolt',
               'ha': '\\hectare',
               'h': '\\hour',
               'L': '\\liter',
               'l': '\\litre',
               'min': '\\minute',
               'Np': '\\neper',
               'rad': '\\rad',
               't': '\\ton',
               '%': '\\%'}
    unit_powers = {'^2': '\\squared',
              '^3': '\\cubed',
              '/': '\\per',
              '^-1': '\\power{}{-1}',
              '^-2': '\\rpsquared',
              '^-3': '\\rpcubed'}
    if unit_package is None:
        # without any unit package:
        units = ''
        k = 0
        while k < len(unit):
            if unit[k] == '^':
                j = k + 1
                while j < len(unit) and (unit[j] == '-' or unit[j].isdigit()):
                    j += 1
                units = units + '$^{\\text{' + unit[k + 1:j] + '}}$'
                k = j
            elif unit[k] == '_':
                j = k + 1
                while j < len(unit) and not unit[j].isspace():
                    j += 1
                units = units + '$_{\\text{' + unit[k + 1:j] + '}}$'
                k = j
            else:
                units = units + unit[k]
                k += 1
    elif unit_package.lower() in ['siunit', 'siunitx']:
        # use SIunit package:
        if '\\' in unit:   # this string is already translated!
            return unit
        units = ''
        j = len(unit)
        while j >= 0:
            for k in range(-3, 0):
                if j+k < 0:
                    continue
                uss = unit[j+k:j]
                if uss in unit_powers:
                    units = unit_powers[uss] + units
                    break
                elif uss in other_units:
                    units = other_units[uss] + units
                    break
                elif uss in si_units:
                    units = si_units[uss] + units
                    j = j+k
                    k = 0
                    if j - 1 >= 0:
                        uss = unit[j - 1:j]
                        if uss in si_prefixes:
                            units = si_prefixes[uss] + units
                            k = -1
                    break
            else:
                k = -1
                units = unit[j+k:j] + units
            j = j + k
        if unit_package.lower() == 'siunitx':
            units = '\\unit{' + units + '}'
    else:
        raise ValueError(f'latex_unit(): invalid unit_package={unit_package}!')
    return units


def break_text(stream, text, maxc=80, indent=0):
    """Write text to stream and break lines at maximum character count.

    Parameters
    ----------
    stream: io
        Stream into which the text is written.
    text: str
        The text to be written to the stream.
    maxc: int
        Maximum character count for each line.
    indent: int
        Number of characters each line is indented.
    """
    nc = 0
    nw = 0
    stream.write(' '*indent)
    nc += indent
    for word in text.split():
        if nc + len(word) > maxc:
            stream.write('\n')
            nc = 0
            nw = 0
            stream.write(' '*indent)
            nc += indent
        if nw > 0:
            stream.write(' ')
            nc += 1
        stream.write(word)
        nc += len(word)
        nw += 1
    stream.write('\n')


def index2aa(n, a='a'):
    """Convert an integer into an alphabetical representation.

    The integer number is converted into 'a', 'b', 'c', ..., 'z',
    'aa', 'ab', 'ac', ..., 'az', 'ba', 'bb', ...

    Inspired by https://stackoverflow.com/a/37604105

    Parameters
    ----------
    n: int
        An integer to be converted into alphabetical representation.
    a: str ('a' or 'A')
        Use upper or lower case characters.

    Returns
    -------
    ns: str
        Alphabetical represtnation of an integer.
    """
    d, m = divmod(n, 26)
    bm = chr(ord(a)+m)
    return index2aa(d - 1, a) + bm if d else bm


def aa2index(s):
    """Convert an alphabetical representation to an index.

    The alphabetical representation 'a', 'b', 'c', ..., 'z',
    'aa', 'ab', 'ac', ..., 'az', 'ba', 'bb', ...
    is converted to an index starting with 0.

    Parameters
    ----------
    s: str
        Alphabetical representation of an index.

    Returns
    -------
    index: int
        The corresponding index.

    Raises
    ------
    ValueError:
        Invalid character in input string.
    """
    index = 0
    maxc = ord('z') - ord('a') + 1
    for c in s.lower():
        index *= maxc
        if ord(c) < ord('a') or ord(c) > ord('z'):
            raise ValueError('invalid character "%s" in string.' % c)
        index += ord(c) - ord('a') + 1
    return index - 1

        
class IndentStream(object):
    """Filter an output stream and start each newline with a number of
    spaces.
    """
    def __init__(self, stream, indent=4):
        self.stream = stream
        self.indent = indent
        self.pending = True

    def __getattr__(self, attr_name):
        return getattr(self.stream, attr_name)

    def write(self, data):
        if not data:
            return
        if self.pending:
            self.stream.write(' '*self.indent)
            self.pending = False
        substr = data.rstrip('\n')
        rn = len(data) - len(substr)
        if len(substr) > 0:
            self.stream.write(substr.replace('\n', '\n'+' '*self.indent))
        if rn > 0:
            self.stream.write('\n'*rn)
            self.pending = True

    def flush(self):
        self.stream.flush()


def main():
    # setup a table:
    df = TableData()
    df.append(["data", "specimen", "ID"], "", "%-s", value=list('ABCBAACB'))
    df.append("size", "m", "%6.2f", value=[2.34, 56.7, 8.9])
    df.append("full weight", "kg", "%.0f", value=122.8)
    df.append_section("all measures")
    df.append("speed", "m/s", "%.3g", value=98.7)
    df.append("median jitter", "mm", "%.1f", value=23)
    df.append("size", "g", "%.2e", value=1.234)
    df.set_descriptions({'ID': 'A unique identifier of a snake.',
                         'size': 'The total length of each snake.',
                         'full weight': 'Weight of each snake',
                         'speed': 'Maximum speed the snake can climb a tree.',
                         'median jitter': 'The jitter around a given path the snake should follow.',
                         'all measures>size': 'Weight of mouse the snake has eaten before.',
                         })
    df.add(np.nan, 2)  # single value
    df.add([0.543, 45, 1.235e2]) # remaining row
    df.add([43.21, 6789.1, 3405, 1.235e-4], 2) # next row
    a = 0.5*np.arange(1, 6)*np.random.randn(5, 5) + 10.0 + np.arange(5)
    df.add(a.T, 1) # rest of table
    #df[3:6,'weight'] = [11.0]*3
    df.insert('median jitter', 's.d.', 'm/s', '%.3g',
              'Standard deviation of all speeds',
              value=2*np.random.rand(df.rows()))
    
    # write out in all formats:
    for tf in TableData.formats:
        print('    - `%s`: %s' % (tf, TableData.descriptions[tf]))
        print('      ```')
        iout = IndentStream(sys.stdout, 4+2)
        df.write(iout, table_format=tf, latex_unit_package='siunitx',
                 latex_merge_std=True)
        print('      ```')
        print()

    # write descriptions:
    for tf in ['md', 'html', 'tex']:
        df.write_descriptions(table_format=tf, maxc=40)
        print()

    # aggregate demos:
    print(df)
    print(df.aggregate(np.mean, numbers_only=True))
    print(df.aggregate(dict(count=len, maximum=np.max), numbers_only=True))
    print(df.aggregate([np.mean, len, max],
                       ['size', 'full weight', 'speed'], 'statistics',
                       remove_nans=True, single_row=False))
    print(df.aggregate({('25%', '50%', '75%'):
                        (np.quantile, ([0.25, 0.6, 0.75],))},
                       numbers_only=True))
    
    print(df.statistics(single_row=False))
    print(df.statistics(single_row=True, remove_nans=True))
    print(df.statistics(remove_nans=True, by='ID'))

    # groupby demo:
    for name, values in df.groupby('ID'):
        print(name)
        print(values)
    print()

    # aggregrate on groups demo:
    print(df.aggregate(np.mean, by='ID'))
    print()

    # write descriptions:
    df.write_descriptions(table_format='md', section_headings=0)
    print()
    
        
if __name__ == "__main__":
    main()
