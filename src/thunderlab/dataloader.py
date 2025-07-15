"""Load time-series data from files.

```
data, rate, unit, amax = load_data('data/file.wav')
```

The function `data_loader()` loads the whole time-series from the file
as a numpy array of floats.  First dimension is frames, second is
channels. In contrast to the `audioio.load_audio()` function, the
values of the data array are not restricted between -1 and 1. They can
assume any value wihin the range `-amax` to `+amax` with the returned
`unit`.

```
data = DataLoader('data/file.wav', 60.0)
```
or
```
with DataLoader('data/file.wav', 60.0) as data:
```
Create an `DataLoader` object that loads chuncks of 60 seconds long data
on demand. `data` can be used like a read-only numpy array of floats.


## Supported file formats

- python pickle files
- numpy .npz files
- matlab .mat files
- audio files via [`audioio`](https://github.com/bendalab/audioio) package
- LabView .scandat files
- relacs trace*.raw files (https://www.relacs.net)
- fishgrid traces-*.raw files (https://github.com/bendalab/fishgrid)


## Metadata

Many file formats allow to store metadata that further describe the
stored time series data. We handle them as nested dictionary of key-value
pairs. Load them with the `metadata()` function:
```
metadata = metadata('data/file.mat')
```

## Markers

Some file formats also allow to store markers that mark specific
positions in the time series data. Load marker positions and spans (in
the 2-D array `locs`) and label and text strings (in the 2-D array
`labels`) with the `markers()` function:
```
locs, labels = markers('data.wav')
```

## Aditional, format specific functions

- `extract_container_metadata()`: extract metadata from dictionary loaded from a container file.
- `relacs_samplerate_unit()`: retrieve sampling rate and unit from a relacs stimuli.dat file.
- `relacs_header()`: read key-value pairs from relacs *.dat file headers.
- `fishgrid_grids()`: retrieve grid sizes from a fishgrid.cfg file.
- `fishgrid_spacings()`: spacing between grid electrodes.

"""

import gc
import os
import sys
import glob
import gzip
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
from pathlib import Path
from datetime import timedelta
from audioio import load_audio, AudioLoader, unflatten_metadata
from audioio import get_number_unit, get_number, get_int, get_bool, get_gain
from audioio import default_starttime_keys, default_gain_keys
from audioio import get_datetime, flatten_metadata, add_metadata, set_starttime
from audioio import metadata as metadata_audioio
from audioio import markers as markers_audioio


def relacs_samplerate_unit(filepath, channel=0):
    """Retrieve sampling rate and unit from a relacs stimuli.dat file.

    Parameters
    ----------
    filepath: str
        Path to a relacs data directory, or a file in a relacs data directory.
    channel: int
        Channel (trace) number, if `filepath` does not specify a
        trace-*.raw file.

    Returns
    -------
    samplerate: float
        Sampling rate in Hertz
    unit: str
        Unit of the trace, can be empty if not found

    Raises
    ------
    IOError/FileNotFoundError:
        If the stimuli.dat file does not exist.
    ValueError:
        stimuli.dat file does not contain sampling rate.
    """
    trace = channel + 1
    relacs_dir = filepath
    # check for relacs data directory:
    if not os.path.isdir(filepath):
        relacs_dir = os.path.dirname(filepath)
        bn = os.path.basename(filepath).lower()
        i = bn.find('.raw')
        if len(bn) > 5 and bn[0:5] == 'trace' and i > 6:
            trace = int(bn[6:i])

    # retreive sampling rate and unit from stimuli.dat file:
    samplerate = None
    sampleinterval = None
    unit = ""

    lines = []
    stimuli_file = os.path.join(relacs_dir, 'stimuli.dat')
    if os.path.isfile(stimuli_file + '.gz'):
        stimuli_file += '.gz'
    if stimuli_file[-3:] == '.gz':
        with gzip.open(stimuli_file, 'r', encoding='latin-1') as sf:
            for line in sf:
                line = line.strip()
                if len(line) == 0 or line[0] != '#':
                    break
                lines.append(line)
    else:
        with open(stimuli_file, 'r', encoding='latin-1') as sf:
            for line in sf:
                line = line.strip()
                if len(line) == 0 or line[0] != '#':
                    break
                lines.append(line)
        
    for line in lines:
        if "unit%d" % trace in line:
            unit = line.split(':')[1].strip()
        if "sampling rate%d" % trace in line:
            value = line.split(':')[1].strip()
            samplerate = float(value.replace('Hz',''))
        elif "sample interval%d" % trace in line:
            value = line.split(':')[1].strip()
            sampleinterval = float(value.replace('ms',''))

    if samplerate is not None:
        return samplerate, unit
    if sampleinterval is not None:
        return 1000/sampleinterval, unit
    raise ValueError(f'could not retrieve sampling rate from {stimuli_file}')


def relacs_header(filepath, store_empty=False, first_only=False,
                  lower_keys=False, flat=False,
                  add_sections=False):
    """Read key-value pairs from a relacs *.dat file header.

    Parameters
    ----------
    filepath: str
        A relacs *.dat file, can be also a zipped .gz file.
    store_empty: bool
        If `False` do not add meta data with empty values.
    first_only: bool
        If `False` only store the first element of a list.
    lower_keys: bool
        Make all keys lower case.
    flat: bool
        Do not make a nested dictionary.
        Use this option also to read in very old relacs metadata with
        ragged left alignment.
    add_sections: bool
        If `True`, prepend keys with sections names separated by
        '.' to make them unique.

    Returns
    -------
    data: dict
        Nested dictionary with key-value pairs of the file header.
        
    Raises
    ------
    IOError/FileNotFoundError:
        If `filepath` cannot be opened.
    """
    # read in header from file:
    lines = []
    if os.path.isfile(filepath + '.gz'):
        filepath += '.gz'
    if filepath[-3:] == '.gz':
        with gzip.open(filepath, 'r', encoding='latin-1') as sf:
            for line in sf:
                line = line.strip()
                if len(line) == 0 or line[0] != '#':
                    break
                lines.append(line)
    else:
        with open(filepath, 'r', encoding='latin-1') as sf:
            for line in sf:
                line = line.strip()
                if len(line) == 0 or line[0] != '#':
                    break
                lines.append(line)
    # parse:
    data = {}
    cdatas = [data]
    sections = ['']
    ident_offs = None
    ident = None
    for line in lines:
        words = line.split(':')
        value = ':'.join(words[1:]).strip() if len(words) > 1 else ''
        if len(words) >= 1:
            key = words[0].strip('#')
            # get section level:
            level = 0
            if not flat or len(value) == 0:
                nident = len(key) - len(key.lstrip())
                if ident_offs is None:
                    ident_offs = nident
                elif ident is None:
                    if nident > ident_offs:
                        ident = nident - ident_offs
                        level = 1
                else:
                    level = (nident - ident_offs)//ident
                # close sections:
                if not flat:
                    while len(cdatas) > level + 1:
                        cdatas[-1][sections.pop()] = cdatas.pop()
                else:
                    while len(sections) > level + 1:
                        sections.pop()
            # key:
            key = key.strip().strip('"')
            if lower_keys:
                key = key.lower()
            skey = key
            if add_sections:
                key = '.'.join(sections[1:] + [key])
            if len(value) == 0:
                # new sub-section:
                if flat:
                    if store_empty:
                        cdatas[-1][key] = None
                else:
                    cdatas.append({})
                sections.append(skey)
            else:
                # key-value pair:
                value = value.strip('"')
                if len(value) > 0 or value != '-' or store_empty:
                    if len(value) > 0 and value[0] == '[' and value[-1] == ']':
                        value = [v.strip() for v in value.lstrip('[').rstrip(']').split(',')]
                        if first_only:
                            value = value[0]
                    cdatas[-1][key] = value
    while len(cdatas) > 1:
        cdatas[-1][sections.pop()] = cdatas.pop()
    return data


def check_relacs(filepath):
    """Check for valid relacs file.

    Parameters
    ----------
    filepath: str
        Path to a relacs data directory, or a file in a relacs data directory.

    Returns
    -------
    is_relacs: boolean
      `True` if `filepath` is a valid relacs directory or is a file therein.
    """
    # relacs data directory:
    relacs_dir = filepath
    if not os.path.isdir(filepath):
        relacs_dir = os.path.dirname(filepath)
    # check for a valid relacs data directory:
    has_stimuli = False
    has_trace = False
    for fname in ['stimuli.dat', 'stimuli.dat.gz']:
        if os.path.isfile(os.path.join(relacs_dir, fname)):
            has_stimuli = True
    for fname in ['trace-1.raw', 'trace-1.raw.gz']:
        if os.path.isfile(os.path.join(relacs_dir, fname)):
            has_trace = True
    return has_stimuli and has_trace

    
def relacs_trace_files(filepath):
    """Expand file path for relacs data to appropriate trace*.raw file names.

    Parameters
    ----------
    filepath: str
        Path to a relacs data directory, or a file in a relacs data directory.
        
    Returns
    -------
    trace_filepaths: list of str
        List of relacs trace*.raw files.
    """
    relacs_dir = filepath
    if not os.path.isdir(filepath):
        relacs_dir = os.path.dirname(filepath)
    trace_filepaths = []
    for k in range(10000):
        fname = os.path.join(relacs_dir, f'trace-{k+1}.raw')
        if os.path.isfile(fname):
            trace_filepaths.append(fname)
        elif os.path.isfile(fname + '.gz'):
            trace_filepaths.append(fname + '.gz')
        else:
            break
    return trace_filepaths

        
def load_relacs(filepath, amax=1.0):
    """Load traces that have been recorded with relacs (https://github.com/relacs/relacs).

    Parameters
    ----------
    filepath: str
        Path to a relacs data directory, or a file in a relacs data directory.
    amax: float
        The amplitude range of the data.

    Returns
    -------
    data: 2-D array
        All data traces as an 2-D numpy array, even for single channel data.
        First dimension is time, second is channel.
    rate: float
        Sampling rate of the data in Hz
    unit: str
        Unit of the data
    amax: float
        Maximum amplitude of data range.

    Raises
    ------
    FileNotFoundError:
        Invalid or non existing relacs files.
    ValueError:
        - Invalid name for relacs trace-*.raw file.
        - Sampling rates of traces differ.
        - Unit of traces differ.
    """
    trace_filepaths = relacs_trace_files(filepath)
    if len(trace_filepaths) == 0:
        raise FileNotFoundError(f'no relacs files found')
    # load trace*.raw files:
    nchannels = len(trace_filepaths)
    data = None
    nrows = 0
    rate = None
    unit = ''
    for c, path in enumerate(sorted(trace_filepaths)):
        if path[-3:] == '.gz':
            with gzip.open(path, 'rb') as sf:
                x = np.frombuffer(sf.read(), dtype=np.float32)
        else:
            x = np.fromfile(path, np.float32)
        if data is None:
            nrows = len(x)
            data = np.zeros((nrows, nchannels))
        n = min(len(x), nrows)
        data[:n,c] = x[:n]
        # retrieve sampling rate and unit:
        crate, us = relacs_samplerate_unit(path, c)
        if rate is None:
            rate = crate
        elif crate != rate:
            raise ValueError('sampling rates of traces differ')
        if len(unit) == 0:
            unit = us
        elif us != unit:
            raise ValueError('unit of traces differ')
    return data, rate, unit, amax


def metadata_relacs(filepath, store_empty=False, first_only=False,
                    lower_keys=False, flat=False, add_sections=False):
    """ Read meta-data of a relacs data set.

    Parameters
    ----------
    filepath: str
        A relacs data directory or a file therein.
    store_empty: bool
        If `False` do not add meta data with empty values.
    first_only: bool
        If `False` only store the first element of a list.
    lower_keys: bool
        Make all keys lower case.
    flat: bool
        Do not make a nested dictionary.
        Use this option also to read in very old relacs metadata with
        ragged left alignment.
    add_sections: bool
        If `True`, prepend keys with sections names separated by
        '.' to make them unique.

    Returns
    -------
    data: nested dict
        Nested dictionary with key-value pairs of the meta data.
    """
    relacs_dir = filepath
    if not os.path.isdir(filepath):
        relacs_dir = os.path.dirname(filepath)
    info_path = os.path.join(relacs_dir, 'info.dat')
    if not os.path.exists(info_path):
        return dict(), []
    data = relacs_header(info_path, store_empty, first_only,
                         lower_keys, flat, add_sections)
    return data


def fishgrid_spacings(metadata, unit='m'):
    """Spacing between grid electrodes.

    Parameters
    ----------
    metadata: dict
        Fishgrid metadata obtained from `metadata_fishgrid()`.
    unit: str
        Unit in which to return the spacings.

    Returns
    -------
    grid_dist: list of tuple of float
        For each grid the distances between rows and columns in `unit`.
    """
    grids_dist = []
    for k in range(4):
        row_dist = get_number(metadata, unit, f'RowDistance{k+1}', default=0)
        col_dist = get_number(metadata, unit, f'ColumnDistance{k+1}', default=0)
        rows = get_int(metadata, f'Rows{k+1}', default=0)
        cols = get_int(metadata, f'Columns{k+1}', default=0)
        if get_bool(metadata, f'Used{k+1}', default=False) or \
           cols > 0 and rows > 0:
            grids_dist.append((row_dist, col_dist))
    return grids_dist


def fishgrid_grids(metadata):
    """Retrieve grid sizes from a fishgrid.cfg file.

    Parameters
    ----------
    metadata: dict
        Fishgrid metadata obtained from `metadata_fishgrid()`.

    Returns
    -------
    grids: list of tuple of int
        For each grid the number of rows and columns.
    """
    grids = []
    for k in range(4):
        rows = get_int(metadata, f'Rows{k+1}', default=0)
        cols = get_int(metadata, f'Columns{k+1}', default=0)
        if get_bool(metadata, f'Used{k+1}', default=False) or \
           cols > 0 and rows > 0:
            grids.append((rows, cols))
    return grids


def check_fishgrid(filepath):
    """Check for valid fishgrid file (https://github.com/bendalab/fishgrid).

    Parameters
    ----------
    filepath: str
        Path to a fishgrid data directory or a file in a fishgrid
        data directory.

    Returns
    -------
    is_fishgrid: bool
        `True` if `filepath` is a valid fishgrid data directory or
        a file therein.
    """
    # fishgrid data directory:
    fishgrid_dir = filepath
    if not os.path.isdir(filepath):
        fishgrid_dir = os.path.dirname(filepath)
    # check for a valid fishgrid data directory:
    return (os.path.isfile(os.path.join(fishgrid_dir, 'fishgrid.cfg')) and
            (os.path.isfile(os.path.join(fishgrid_dir, 'traces-grid1.raw')) or
             os.path.isfile(os.path.join(fishgrid_dir, 'traces.raw'))))

    
def fishgrid_trace_files(filepath):
    """Expand file paths for fishgrid data to appropriate traces*.raw file names.

    Parameters
    ----------
    filepath: str
        Path to a fishgrid data directory, or a file therein.
        
    Returns
    -------
    trace_filepaths: list of str
        List of fishgrid traces*.raw files.
    """
    # find grids:
    fishgrid_dir = filepath
    if not os.path.isdir(fishgrid_dir):
        fishgrid_dir = os.path.dirname(filepath)
    trace_filepaths = []
    for k in range(10000):
        file = os.path.join(fishgrid_dir, f'traces-grid{k+1}.raw')
        if os.path.isfile(file):
            trace_filepaths.append(file)
        else:
            break
    if len(trace_filepaths) == 0:
        file = os.path.join(fishgrid_dir, f'traces.raw')
        if os.path.isfile(file):
            trace_filepaths.append(file)
    return trace_filepaths

        
def load_fishgrid(filepath):
    """Load traces that have been recorded with fishgrid (https://github.com/bendalab/fishgrid).

    Parameters
    ----------
    filepath: str
        Path to a fishgrid data directory, or a file therein.

    Returns
    -------
    data: 2-D array
        All data traces as an 2-D numpy array, even for single channel data.
        First dimension is time, second is channel.
    rate: float
        Sampling rate of the data in Hz.
    unit: str
        Unit of the data.
    amax: float
        Maximum amplitude of data range.

    Raises
    ------
    FileNotFoundError:
        Invalid or non existing fishgrid files.
    """
    trace_filepaths = fishgrid_trace_files(filepath)
    if len(trace_filepaths) == 0:
        raise FileNotFoundError(f'no fishgrid files found')
    md = metadata_fishgrid(filepath)
    grids = fishgrid_grids(md)
    grid_sizes = [r*c for r, c in grids]
                
    # load traces-grid*.raw files:
    grid_channels = []
    nchannels = 0
    for g, path in enumerate(trace_filepaths):
        grid_channels.append(grid_sizes[g])
        nchannels += grid_sizes[g]
    data = None
    nrows = 0
    c = 0
    rate = get_number(md, 'Hz', 'AISampleRate')
    for path, channels in zip(trace_filepaths, grid_channels):
        x = np.fromfile(path, np.float32).reshape((-1, channels))
        if data is None:
            nrows = len(x)
            data = np.zeros((nrows, nchannels))
        n = min(len(x), nrows)
        data[:n,c:c+channels] = x[:n,:]
        c += channels
    amax, unit = get_number_unit(md, 'AIMaxVolt')
    return data, rate, unit, amax


# add fishgrid keys:
default_starttime_keys.append(['StartDate', 'StartTime'])
default_gain_keys.insert(0, 'AIMaxVolt')


def metadata_fishgrid(filepath):
    """ Read meta-data of a fishgrid data set.

    Parameters
    ----------
    filepath: str
        A fishgrid data directory or a file therein.

    Returns
    -------
    data: nested dict
        Nested dictionary with key-value pairs of the meta data.
    """
    fishgrid_dir = filepath
    if not os.path.isdir(fishgrid_dir):
        fishgrid_dir = os.path.dirname(filepath)
    path = os.path.join(fishgrid_dir, 'fishgrid.cfg')
    # read in header from file:
    lines = []
    if os.path.isfile(path + '.gz'):
        path += '.gz'
    if not os.path.exists(path):
        return {}
    if path[-3:] == '.gz':
        with gzip.open(path, 'r', encoding='latin-1') as sf:
            for line in sf:
                lines.append(line)
    else:
        with open(path, 'r', encoding='latin-1') as sf:
            for line in sf:
                lines.append(line)
    # parse:
    data = {}
    cdatas = [data]
    ident_offs = None
    ident = None
    old_style = False
    grid_n = False
    for line in lines:
        if len(line.strip()) == 0:
            continue
        if line[0] == '*':
            key = line[1:].strip()
            data[key] = {}
            cdatas = [data, data[key]]
        elif '----' in line:
            old_style = True
            key = line.strip().strip(' -').replace('&', '')
            if key.upper() == 'SETUP':
                key = 'Grid 1'
            grid_n = False
            if key[:4].lower() == 'grid':
                grid_n = key[5]
            cdatas = cdatas[:2]
            cdatas[1][key] = {}
            cdatas.append(cdatas[1][key])
        else:
            words = line.split(':')
            key = words[0].strip().strip('"')
            value = None
            if len(words) > 1 and (len(words[1].strip()) > 0 or old_style):
                value = ':'.join(words[1:]).strip().strip('"')
            if old_style:
                if value is None:
                    cdatas = cdatas[:3]
                    cdatas[2][key] = {}
                    cdatas.append(cdatas[2][key])            
                else:
                    if grid_n and key[-1] != grid_n:
                        key = key + grid_n
                    cdatas[-1][key] = value
            else:
                # get section level:
                level = 0
                nident = len(line) - len(line.lstrip())
                if ident_offs is None:
                    ident_offs = nident
                elif ident is None:
                    if nident > ident_offs:
                        ident = nident - ident_offs
                        level = 1
                else:
                    level = (nident - ident_offs)//ident
                # close sections:
                cdatas = cdatas[:2 + level]
                if value is None:
                    # new section:
                    cdatas[-1][key] = {}
                    cdatas.append(cdatas[-1][key])
                else:
                    # key-value pair:
                    cdatas[-1][key] = value.replace(r'\n', '\n')
    # remove unused grids:
    fgm = data.get('FishGrid', {})
    for i in range(4):
        gs = f'Grid {i+1}'
        if gs in fgm:
            gm = fgm[gs]
            us = f'Used{i+1}'
            if us in gm and gm[us].upper() == 'FALSE':
                del fgm[gs]
    return data


def markers_fishgrid(filepath):
    """ Read markers of a fishgrid data set.

    Parameters
    ----------
    filepath: str
        A fishgrid data directory or a file therein.

    Returns
    -------
    locs: 2-D array of ints
        Marker positions (first column) and spans (second column)
        for each marker (rows).
    labels: 2-D array of string objects
        Labels (first column) and texts (second column)
        for each marker (rows).
    """
    def add_marker():
        if 'index1' in marker:
            index1 = int(marker['index1'])//nchannels
        else:
            index1 = int(marker['index'])//nchannels
        span1 = int(marker.get('span1', 0))//nchannels
        locs.append([index1, span1])
        ls = marker.get('label', 'M')
        cs = marker.get('comment', '')
        labels.append([ls, cs])
        
    fishgrid_dir = filepath
    if not os.path.isdir(fishgrid_dir):
        fishgrid_dir = os.path.dirname(filepath)
    path = os.path.join(fishgrid_dir, 'timestamps.dat')
    if not os.path.isfile(path):
        return np.zeros((0, 2), dtype=int), np.zeros((0, 2), dtype=object)
    # get number of channels:
    md = metadata_fishgrid(path.replace('timestamps.dat', 'fishgrid.cfg'))
    grids = fishgrid_grids(md)
    nchannels = np.prod(grids[0])
    # read timestamps:
    locs = []
    labels = []
    marker = {}
    with open(path, 'r') as sf:
        for line in sf:
            if len(line.strip()) == 0:
                add_marker()
                marker = {}
            else:
                words = line.split(':')
                if len(words) > 1:
                    v = words[1].strip()
                    v = v.strip('"')
                    marker[words[0].strip().lower()] = v
    if len(marker) > 0:
        add_marker()
    if len(locs) > 2:
        return np.array(locs[1:-1]), np.array(labels[1:-1])
    else:
        return np.zeros((0, 2), dtype=int), np.zeros((0, 2), dtype=object)
        

def check_container(filepath):
    """Check if file is a generic container file.

    Supported file formats are:

    - python pickle files (.pkl)
    - numpy files (.npz)
    - matlab files (.mat)

    Parameters
    ----------
    filepath: str
        Path of the file to check.
    
    Returns
    -------
    is_container: bool
        `True`, if `filepath` is a supported container format.
    """
    ext = os.path.splitext(filepath)[1]
    return ext.lower() in ('.pkl', '.npz', '.mat')


def extract_container_data(data_dict, datakey=None,
                           samplekey=['rate', 'Fs', 'fs'],
                           timekey=['time'], amplkey=['amax'], unitkey='unit',
                           amax=1.0, unit='a.u.'):
    """Extract data from dictionary loaded from a container file.

    Parameters
    ----------
    data_dict: dict
        Dictionary of the data items contained in the container.
    datakey: None, str, or list of str
        Name of the variable holding the data.  If `None` take the
        variable that is an 2D array and has the largest number of
        elements.
    samplekey: str or list of str
        Name of the variable holding the sampling rate.
    timekey: str or list of str
        Name of the variable holding sampling times.
        If no sampling rate is available, the sampling rate is retrieved
        from the sampling times.
    amplkey: str or list of str
        Name of the variable holding the amplitude range of the data.
    unitkey: str
        Name of the variable holding the unit of the data.
    amax: None or float
        If specified and no amplitude range has been found in `data_dict`,
        then this is the amplitude range of the data.
    unit: None or str
        If specified and no unit has been found in `data_dict`,
        then return this as the unit of the data.

    Returns
    -------
    data: 2-D array of floats
        All data traces as an 2-D numpy array, even for single channel data.
        First dimension is time, second is channel.
    rate: float
        Sampling rate of the data in Hz.
    unit: str
        Unit of the data.
    amax: float
        Maximum amplitude of data range in `unit`.

    Raises
    ------
    ValueError:
        Invalid key requested.
    """
    # extract format data:
    if not isinstance(samplekey, (list, tuple, np.ndarray)):
        samplekey = (samplekey,)
    if not isinstance(timekey, (list, tuple, np.ndarray)):
        timekey = (timekey,)
    if not isinstance(amplkey, (list, tuple, np.ndarray)):
        amplkey = (amplkey,)
    rate = 0.0
    for skey in samplekey:
        if skey in data_dict:
            rate = float(data_dict[skey])
            break
    if rate == 0.0:
        for tkey in timekey:
            if tkey in data_dict:
                rate = 1.0/(data_dict[tkey][1] - data_dict[tkey][0])
                break
    if rate == 0.0:
        raise ValueError(f"invalid keys {', '.join(samplekey)} and {', '.join(timekey)} for requesting sampling rate or sampling times")
    for akey in amplkey:
        if akey in data_dict:
            amax = float(data_dict[akey])
            break
    if unitkey in data_dict:
        unit = data_dict[unitkey]
    # get data array:
    raw_data = np.array([])
    if datakey:
        # try data keys:
        if not isinstance(datakey, (list, tuple, np.ndarray)):
            datakey = (datakey,)
        for dkey in datakey:
            if dkey in data_dict:
                raw_data = data_dict[dkey]
                break
        if len(raw_data) == 0:
            raise ValueError(f"invalid key(s) {', '.join(datakey)} for requesting data")
    else:
        # find largest 2D array:
        for d in data_dict:
            if hasattr(data_dict[d], 'shape'):
                if 1 <= len(data_dict[d].shape) <= 2 and \
                   np.max(data_dict[d].shape) > np.max(raw_data.shape):
                    raw_data = data_dict[d]
    if len(raw_data) == 0:
        raise ValueError('no data found')
    # make 2D:
    if len(raw_data.shape) == 1:
        raw_data = raw_data.reshape(-1, 1)
    # transpose if necessary:
    if np.argmax(raw_data.shape) > 0:
        raw_data = raw_data.T
    # recode:
    if raw_data.dtype == np.dtype('int16'):
        data = raw_data.astype('float32')
        data *= amax/2**15
    elif raw_data.dtype == np.dtype('int32'):
        data = raw_data.astype(float)
        data *= amax/2**31
    elif raw_data.dtype == np.dtype('int64'):
        data = raw_data.astype(float)
        data *= amax/2**63
    else:
        data = raw_data
    return data, rate, unit, amax


def load_container(filepath, datakey=None,
                   samplekey=['rate', 'Fs', 'fs'],
                   timekey=['time'], amplkey=['amax'], unitkey='unit',
                   amax=1.0, unit='a.u.'):
    """Load data from a generic container file.

    Supported file formats are:

    - python pickle files (.pkl)
    - numpy files (.npz)
    - matlab files (.mat)

    Parameters
    ----------
    filepath: str
        Path of the file to load.
    datakey: None, str, or list of str
        Name of the variable holding the data.  If `None` take the
        variable that is an 2D array and has the largest number of
        elements.
    samplekey: str or list of str
        Name of the variable holding the sampling rate.
    timekey: str or list of str
        Name of the variable holding sampling times.
        If no sampling rate is available, the sampling rate is retrieved
        from the sampling times.
    amplkey: str
        Name of the variable holding the amplitude range of the data.
    unitkey: str
        Name of the variable holding the unit of the data.
        If `unitkey` is not a valid key, then return `unitkey` as the `unit`.
    amax: None or float
        If specified and no amplitude range has been found in the data
        container, then this is the amplitude range of the data.
    unit: None or str
        If specified and no unit has been found in the data container,
        then return this as the unit of the data.

    Returns
    -------
    data: 2-D array of floats
        All data traces as an 2-D numpy array, even for single channel data.
        First dimension is time, second is channel.
    rate: float
        Sampling rate of the data in Hz.
    unit: str
        Unit of the data.
    amax: float
        Maximum amplitude of data range.

    Raises
    ------
    ValueError:
        Invalid key requested.
    """
    # load data:
    data_dict = {}
    ext = os.path.splitext(filepath)[1]
    if ext == '.pkl':
        import pickle
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
    elif ext == '.npz':
        data_dict = np.load(filepath)
    elif ext == '.mat':
        from scipy.io import loadmat
        data_dict = loadmat(filepath, squeeze_me=True)
    return extract_container_data(data_dict, datakey, samplekey,
                                  timekey, amplkey, unitkey, amax, unit)


def extract_container_metadata(data_dict, metadatakey=['metadata', 'info']):
    """ Extract metadata from dictionary loaded from a container file.

    Parameters
    ----------
    data_dict: dict
        Dictionary of the data items contained in the container.
    metadatakey: str or list of str
        Name of the variable holding the metadata.

    Returns
    -------
    metadata: nested dict
        Nested dictionary with key-value pairs of the meta data.
    """
    if not isinstance(metadatakey, (list, tuple, np.ndarray)):
        metadatakey = (metadatakey,)
    # get single metadata dictionary:
    for mkey in metadatakey:
        if mkey in data_dict:
            return data_dict[mkey]
    # collect all keys starting with metadatakey:
    metadata = {}
    for mkey in metadatakey:
        mkey += '__'
        for dkey in data_dict:
            if dkey[:len(mkey)] == mkey:
                v = data_dict[dkey]
                if hasattr(v, 'size') and v.ndim == 0:
                    v = v.item()
                metadata[dkey[len(mkey):]] = v
        if len(metadata) > 0:
            return unflatten_metadata(metadata, sep='__')
    return metadata


def metadata_container(filepath, metadatakey=['metadata', 'info']):
    """ Read meta-data of a container file.

    Parameters
    ----------
    filepath: str
        A container file.
    metadatakey: str or list of str
        Name of the variable holding the metadata.

    Returns
    -------
    metadata: nested dict
        Nested dictionary with key-value pairs of the meta data.
    """
    data_dict = {}
    ext = os.path.splitext(filepath)[1]
    if ext == '.pkl':
        import pickle
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
    elif ext == '.npz':
        data_dict = np.load(filepath)
    elif ext == '.mat':
        from scipy.io import loadmat
        data_dict = loadmat(filepath, squeeze_me=True)
    return extract_container_metadata(data_dict, metadatakey)


def extract_container_markers(data_dict, poskey=['positions'],
                              spanskey=['spans'], labelskey=['labels'],
                              descrkey=['descriptions']):
    """ Extract markers from dictionary loaded from a container file.

    Parameters
    ----------
    data_dict: dict
        Dictionary of the data items contained in the container.
    poskey: str or list of str
        Name of the variable holding positions of markers.
    spanskey: str or list of str
        Name of the variable holding spans of markers.
    labelskey: str or list of str
        Name of the variable holding labels of markers.
    descrkey: str or list of str
        Name of the variable holding descriptions of markers.

    Returns
    -------
    locs: 2-D array of ints
        Marker positions (first column) and spans (second column)
        for each marker (rows).
    labels: 2-D array of string objects
        Labels (first column) and texts (second column)
        for each marker (rows).
    """
    if not isinstance(poskey, (list, tuple, np.ndarray)):
        poskey = (poskey,)
    if not isinstance(spanskey, (list, tuple, np.ndarray)):
        spanskey = (spanskey,)
    if not isinstance(labelskey, (list, tuple, np.ndarray)):
        labelskey = (labelskey,)
    if not isinstance(descrkey, (list, tuple, np.ndarray)):
        descrkey = (descrkey,)
    locs = np.zeros((0, 2), dtype=int)
    for pkey in poskey:
        if pkey in data_dict:
            locs = np.zeros((len(data_dict[pkey]), 2), dtype=int)
            locs[:,0] = data_dict[pkey]
            break
    for skey in spanskey:
        if skey in data_dict:
            locs[:,1] = data_dict[skey]
            break
    labels = np.zeros((0, 2), dtype=object)
    for lkey in labelskey:
        if lkey in data_dict:
            labels = np.zeros((len(data_dict[lkey]), 2), dtype=object)
            labels[:,0] = data_dict[lkey]
            break
    for dkey in descrkey:
        if dkey in data_dict:
            labels[:,1] = data_dict[dkey]
            break
    return locs, labels


def markers_container(filepath, poskey=['positions'],
                      spanskey=['spans'], labelskey=['labels'],
                      descrkey=['descriptions']):
    """ Read markers of a container file.

    Parameters
    ----------
    filepath: str
        A container file.
    poskey: str or list of str
        Name of the variable holding positions of markers.
    spanskey: str or list of str
        Name of the variable holding spans of markers.
    labelskey: str or list of str
        Name of the variable holding labels of markers.
    descrkey: str or list of str
        Name of the variable holding descriptions of markers.

    Returns
    -------
    locs: 2-D array of ints
        Marker positions (first column) and spans (second column)
        for each marker (rows).
    labels: 2-D array of string objects
        Labels (first column) and texts (second column)
        for each marker (rows).
    """
    data_dict = {}
    ext = os.path.splitext(filepath)[1]
    if ext == '.pkl':
        import pickle
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
    elif ext == '.npz':
        data_dict = np.load(filepath)
    elif ext == '.mat':
        from scipy.io import loadmat
        data_dict = loadmat(filepath, squeeze_me=True)
    return extract_container_markers(data_dict, poskey, spanskey,
                                     labelskey, descrkey)


def check_raw(filepath):
    """Check if file is a raw file.

    The following extensions are interpreted as raw files:

    - raw files (*.raw)
    - LabView scandata (*.scandat)

    Parameters
    ----------
    filepath: str
        Path of the file to check.
    
    Returns
    -------
    is_raw: bool
        `True`, if `filepath` is a raw format.
    """
    ext = os.path.splitext(filepath)[1]
    return ext.lower() in ('.raw', '.scandat', '.mat')


def load_raw(filepath, rate=44000, channels=1, dtype=np.float32,
             amax=1.0, unit='a.u.'):
    """Load data from a raw file.

    Raw files just contain the data and absolutely no metadata, not
    even the smapling rate, number of channels, etc.
    Supported file formats are:

    - raw files (*.raw)
    - LabView scandata (*.scandat)

    Parameters
    ----------
    filepath: str
        Path of the file to load.
    rate: float
        Sampling rate of the data in Hertz.
    channels: int
        Number of channels multiplexed in the data.
    dtype: str or numpy.dtype
        The data type stored in the file.
    amax: float
        The amplitude range of the data.
    unit: str
        The unit of the data.

    Returns
    -------
    data: 2-D array of floats
        All data traces as an 2-D numpy array, even for single channel data.
        First dimension is time, second is channel.
    rate: float
        Sampling rate of the data in Hz.
    unit: str
        Unit of the data.
    amax: float
        Maximum amplitude of data range.

    """
    raw_data = np.fromfile(filepath, dtype=dtype).reshape(-1, channels)
    # recode:
    if dtype == np.dtype('int16'):
        data = raw_data.astype('float32')
        data *= amax/2**15
    elif dtype == np.dtype('int32'):
        data = raw_data.astype(float)
        data *= amax/2**31
    elif dtype == np.dtype('int64'):
        data = raw_data.astype(float)
        data *= amax/2**63
    else:
        data = raw_data
    return data, rate, unit, amax
    

def load_audioio(filepath, verbose=0, gainkey=default_gain_keys, sep='.',
                 amax=1.0, unit='a.u.'):
    """Load data from an audio file.

    See the
    [`load_audio()`](https://bendalab.github.io/audioio/api/audioloader.html#audioio.audioloader.load_audio)
    function of the [`audioio`](https://github.com/bendalab/audioio)
    package for more infos.

    Parameters
    ----------
    filepath: str
        Path of the file to load.
    verbose: int
        If > 0 show detailed error/warning messages.
    gainkey: str or list of str
        Key in the file's metadata that holds some gain information.
        If found, the data will be multiplied with the gain,
        and if available, the corresponding unit is returned.
        See the [audioio.get_gain()](https://bendalab.github.io/audioio/api/audiometadata.html#audioio.audiometadata.get_gain) function for details.
    sep: str
        String that separates section names in `gainkey`.
    amax: float
        If specified and no gain has been found in the metadata,
        then use this as the amplitude range.
    unit: str
        If specified and no gain has been found in the metadata,
        then return this as the unit of the data.

    Returns
    -------
    data: 2-D array of floats
        All data traces as an 2-D numpy array, even for single channel data.
        First dimension is time, second is channel.
    rate: float
        Sampling rate of the data in Hz.
    unit: str
        Unit of the data if found in the metadata (see `gainkey`),
        otherwise `unit`.
    amax: float
        Maximum amplitude of data range.
    """
    # get gain:
    md = metadata_audioio(filepath)
    amax, unit = get_gain(md, gainkey, sep, amax, unit)
    # load data:
    data, rate = load_audio(filepath, verbose)
    if amax != 1.0:
        data *= amax
    return data, rate, unit, amax


data_loader_funcs = (
    ('relacs', check_relacs, load_relacs, metadata_relacs, None),
    ('fishgrid', check_fishgrid, load_fishgrid, metadata_fishgrid, markers_fishgrid),
    ('container', check_container, load_container, metadata_container, markers_container),
    ('raw', check_raw, load_raw, None, None),
    ('audioio', None, load_audioio, metadata_audioio, markers_audioio),
    )
"""List of implemented load functions.

Each element of the list is a tuple with the data format's name, its
check and its load function.

"""    

    
def load_data(filepath, verbose=0, **kwargs):
    """Load time-series data from a file.

    Parameters
    ----------
    filepath: str
        Path and name of the file to load.
    verbose: int
        If > 0 show detailed error/warning messages.
    **kwargs: dict
        Further keyword arguments that are passed on to the 
        format specific loading functions.
        For example:
        - `amax`: the amplitude range of the data.
        - 'unit': the unit of the data.

    Returns
    -------
    data: 2-D array
        All data traces as an 2-D numpy array, even for single channel data.
        First dimension is time, second is channel.
    rate: float
        Sampling rate of the data in Hz.
    unit: str
        Unit of the data.
    amax: float
        Maximum amplitude of data range.

    Raises
    ------
    ValueError:
        `filepath` is empty string.
    """
    if len(filepath) == 0:
        raise ValueError('input argument filepath is empty string.')
    # load data:
    for name, check_file, load_file, _, _ in  data_loader_funcs:
        if check_file is None or check_file(filepath):
            data, rate, unit, amax = load_file(filepath, **kwargs)
            if verbose > 0:
                print(f'loaded {name} data from file "{filepath}"')
                if verbose > 1:
                    print(f'  sampling rate: {rate:g} Hz')
                    print(f'  channels     : {data.shape[1]}')
                    print(f'  frames       : {len(data)}')
                    print(f'  range        : {amax:g}{unit}')
            return data, rate, unit, amax
    return np.zeros((0, 1)), 0.0, '', 1.0


def metadata(filepath, **kwargs):
    """ Read meta-data from a data file.

    Parameters
    ----------
    filepath: str
        The full path and name of the file to load. For some file
        formats several files can be provided in a list.
    **kwargs: dict
        Further keyword arguments that are passed on to the 
        format specific loading functions.

    Returns
    -------
    meta_data: nested dict
        Meta data contained in the file.  Keys of the nested
        dictionaries are always strings.  If the corresponding
        values are dictionaries, then the key is the section name
        of the metadata contained in the dictionary. All other
        types of values are values for the respective key. In
        particular they are strings, or list of strings. But other
        simple types like ints or floats are also allowed.

    Raises
    ------
    ValueError:
        `filepath` is empty string.
    """
    if len(filepath) == 0:
        raise ValueError('input argument filepath is empty string.')
    # load metadata:
    for _, check_file, _, metadata_file, _ in  data_loader_funcs:
        if check_file is None or check_file(filepath):
            if metadata_file is not None:
                return metadata_file(filepath, **kwargs)
    return {}


def markers(filepath):
    """ Read markers of a data file.

    Parameters
    ----------
    filepath: str or file handle
        The data file.

    Returns
    -------
    locs: 2-D array of ints
        Marker positions (first column) and spans (second column)
        for each marker (rows).
    labels: 2-D array of string objects
        Labels (first column) and texts (second column)
        for each marker (rows).

    Raises
    ------
    ValueError:
        `filepath` is empty string.
    """
    if len(filepath) == 0:
        raise ValueError('input argument filepath is empty string.')
    # load markers:
    for _, check_file, _, _, markers_file in  data_loader_funcs:
        if check_file is None or check_file(filepath):
            if markers_file is not None:
                return markers_file(filepath)
    return np.zeros((0, 2), dtype=int), np.zeros((0, 2), dtype=object)


class DataLoader(AudioLoader):
    """Buffered reading of time-series data for random access of the data in the file.
    
    This allows for reading very large data files that do not fit into
    memory.  A `DataLoader` instance can be used like a huge
    read-only numpy array, i.e.
    ```
    data = DataLoader('path/to/data/file.dat')
    x = data[10000:20000,0]
    ```
    The first index specifies the frame, the second one the channel.

    `DataLoader` first determines the format of the data file and then
    opens the file (first line). It then reads data from the file as
    necessary for the requested data (second line).

    Supported file formats are

    - audio files via `audioio` package
    - python pickle files
    - numpy .npz files
    - matlab .mat files
    - relacs trace*.raw files (www.relacs.net)
    - fishgrid traces-*.raw files

    Reading sequentially through the file is always possible. If
    previous data are requested, then the file is read from the
    beginning. This might slow down access to previous data
    considerably. Use the `backsize` argument to the open functions to
    make sure some data are loaded before the requested frame. Then a
    subsequent access to the data within `backsize` seconds before that
    frame can still be handled without the need to reread the file
    from the beginning.

    Usage:
    ------
    ```
    import thunderlab.dataloader as dl
    with dl.DataLoader(filepath, 60.0, 10.0) as data:
        # do something with the content of the file:
        x = data[0:10000,0]
        y = data[10000:20000,0]
        z = x + y
    ```
    
    Normal open and close:
    ```
    data = dl.DataLoader(filepath, 60.0)
    x = data[:,:]  # read the whole file
    data.close()
    ```    
    that is the same as:
    ```
    data = dl.DataLoader()
    data.open(filepath, 60.0)
    ```
    
    Parameters
    ----------
    filepath: str
        Name of the file.
    buffersize: float
        Size of internal buffer in seconds.
    backsize: float
        Part of the buffer to be loaded before the requested start index in seconds.
    verbose: int
        If larger than zero show detailed error/warning messages.
    meta_kwargs: dict
        Keyword arguments that are passed on to the _load_metadata() function.

    Attributes
    ----------
    rate: float
        The sampling rate of the data in Hertz.
    channels: int
        The number of channels that are read in.
    frames: int
        The number of frames in the file.
    format: str or None
        Format of the audio file.
    encoding: str or None
        Encoding/subtype of the audio file.
    shape: tuple
        Number of frames and channels of the data.
    ndim: int
        Number of dimensions: always 2 (frames and channels).
    unit: str
        Unit of the data.
    ampl_min: float
        Minimum amplitude the file format supports.
    ampl_max: float
        Maximum amplitude the file format supports.

    Methods
    -------

    - `len()`: the number of frames
    - `open()`: open a data file.
    - `open_*()`: open a data file of a specific format.
    - `close()`: close the file.
    - `metadata()`: metadata of the file.
    - `markers()`: markers of the file.
    - `set_unwrap()`: Set parameters for unwrapping clipped data.

    """

    def __init__(self, filepath=None, buffersize=10.0, backsize=0.0,
                 verbose=0, **meta_kwargs):
        super().__init__(None, buffersize, backsize,
                         verbose, **meta_kwargs)
        if filepath is not None:
            self.open(filepath, buffersize, backsize, verbose, **meta_kwargs)

    def __getitem__(self, key):
        return super(DataLoader, self).__getitem__(key)
 
    def __next__(self):
        return super(DataLoader, self).__next__()

    
    # relacs interface:        
    def open_relacs(self, filepath, buffersize=10.0, backsize=0.0,
                    verbose=0, amax=1.0):
        """Open relacs data files (www.relacs.net) for reading.

        Parameters
        ----------
        filepath: str
            Path to a relacs data directory or a file therein.
        buffersize: float
            Size of internal buffer in seconds.
        backsize: float
            Part of the buffer to be loaded before the requested start index in seconds.
        verbose: int
            If > 0 show detailed error/warning messages.
        amax: float
            The amplitude range of the data.

        Raises
        ------
        FileNotFoundError:
            Invalid or non existing fishgrid files.
        ValueError:
            .gz files not supported.
        """
        self.verbose = verbose

        # open trace files:
        self.trace_filepaths = relacs_trace_files(filepath)
        if len(self.trace_filepaths) == 0:
            raise FileNotFoundError(f'no relacs files found')
        self.sf = []
        self.frames = None
        self.rate = None
        self.unit = ''
        self.filepath = filepath
        self.file_paths = [self.filepath]
        self.file_indices = [0]
        for path in self.trace_filepaths:
            if path[-3:] == '.gz':
                raise ValueError('.gz files not supported')
            sf = open(path, 'rb')
            self.sf.append(sf)
            if verbose > 0:
                print(f'open_relacs(filepath) with filepath={path}')
            # file size:
            sf.seek(0, os.SEEK_END)
            frames = sf.tell()//4
            if self.frames is None:
                self.frames = frames
            elif self.frames != frames:
                diff = self.frames - frames
                if diff > 1 or diff < -2:
                    raise ValueError('number of frames of traces differ')
                elif diff >= 0:
                    self.frames = frames
            sf.seek(0)
            # retrieve sampling rate and unit:
            rate, us = relacs_samplerate_unit(path)
            if self.rate is None:
                self.rate = rate
            elif rate != self.rate:
                raise ValueError('sampling rates of traces differ')
            if len(self.unit) == 0:
                self.unit = us
            elif us != self.unit:
                raise ValueError('unit of traces differ')
        self.channels = len(self.sf)
        self.shape = (self.frames, self.channels)
        self.size = self.frames * self.channels
        self.ndim = len(self.shape)
        self.format = 'RELACS'
        self.encoding = 'FLOAT'
        self.bufferframes = int(buffersize*self.rate)
        self.backframes = int(backsize*self.rate)
        self.init_buffer()
        self.offset = 0
        self.close = self._close_relacs
        self.load_audio_buffer = self._load_buffer_relacs
        self.basename = self._basename_relacs
        self.ampl_min = -amax
        self.ampl_max = +amax
        self._load_metadata = self._metadata_relacs
        # TODO: load markers:
        self._locs = np.zeros((0, 2), dtype=int)
        self._labels = np.zeros((0, 2), dtype=object)
        self._load_markers = None
        return self

    def _close_relacs(self):
        """Close the relacs data files.
        """
        for file in self.sf:
            file.close()
        self.sf = []

    def _load_buffer_relacs(self, r_offset, r_size, buffer):
        """Load new data from relacs data file.

        Parameters
        ----------
        r_offset: int
           First frame to be read from file.
        r_size: int
           Number of frames to be read from file.
        buffer: ndarray
           Buffer where to store the loaded data.
        """
        if len(self.sf) == 0 and len(self.trace_filepaths) > 0:
            for path in self.trace_filepaths:
                self.sf.append(open(path, 'rb'))
        for i, file in enumerate(self.sf):
            file.seek(r_offset*4)
            data = file.read(r_size*4)
            buffer[:, i] = np.frombuffer(data, dtype=np.float32)
        

    def _metadata_relacs(self, store_empty=False, first_only=False):
        """ Load meta-data of a relacs data set.
        """
        path = os.path.dirname(self.filepath)
        info_path = os.path.join(path, 'info.dat')
        if not os.path.exists(info_path):
            return {}
        return relacs_header(info_path, store_empty, first_only)

    def _basename_relacs(self, path=None):
        """ Base name of the relacs data files.

        Parameters
        ----------
        path: str or None
            Path of a relacs data file (*.raw, info.dat, or just the directory).
            If `None`, use `self.filepath`.

        Returns
        -------
        s: str
            The base name, i.e. the name of the directory containing the
            relacs data files.

        """
        if path is None:
            path = self.filepath
        path = Path(path)
        if path.is_dir():
            return path.name
        else:
            return path.parent.name

    
    # fishgrid interface:        
    def open_fishgrid(self, filepath, buffersize=10.0, backsize=0.0,
                      verbose=0):
        """Open fishgrid data files (https://github.com/bendalab/fishgrid) for reading.

        Parameters
        ----------
        filepath: str
            Path to a fishgrid data directory, or a file therein.
        buffersize: float
            Size of internal buffer in seconds.
        backsize: float
            Part of the buffer to be loaded before the requested start index in seconds.
        verbose: int
            If > 0 show detailed error/warning messages.

        Raises
        ------
        FileNotFoundError:
            Invalid or non existing fishgrid files.
        """
        self.verbose = verbose

        self.trace_filepaths = fishgrid_trace_files(filepath)
        if len(self.trace_filepaths) == 0:
            raise FileNotFoundError(f'no fishgrid files found')
        self.filepath = filepath
        self.file_paths = [self.filepath]
        self.file_indices = [0]
        self._load_metadata = metadata_fishgrid
        self._load_markers = markers_fishgrid

        # open grid files:
        grids = fishgrid_grids(self.metadata())
        grid_sizes = [r*c for r,c in grids]
        self.channels = 0
        for g, path in enumerate(self.trace_filepaths):
            self.channels += grid_sizes[g]
        self.sf = []
        self.grid_channels = []
        self.grid_offs = []
        offs = 0
        self.frames = None
        self.rate = get_number(self.metadata(), 'Hz', 'AISampleRate')
        v, self.unit = get_number_unit(self.metadata(), 'AIMaxVolt')
        if v is not None:
            self.ampl_min = -v
            self.ampl_max = +v
            
        for g, path in enumerate(self.trace_filepaths):
            sf = open(path, 'rb')
            self.sf.append(sf)
            if verbose > 0:
                print(f'open_fishgrid(filepath) with filepath={path}')
            # grid channels:
            self.grid_channels.append(grid_sizes[g])
            self.grid_offs.append(offs)
            offs += grid_sizes[g]
            # file size:
            sf.seek(0, os.SEEK_END)
            frames = sf.tell()//4//grid_sizes[g]
            if self.frames is None:
                self.frames = frames
            elif self.frames != frames:
                diff = self.frames - frames
                if diff > 1 or diff < -2:
                    raise ValueError('number of frames of traces differ')
                elif diff >= 0:
                    self.frames = frames
            sf.seek(0)
        self.shape = (self.frames, self.channels)
        self.size = self.frames * self.channels
        self.ndim = len(self.shape)
        self.format = 'FISHGRID'
        self.encoding = 'FLOAT'
        self.bufferframes = int(buffersize*self.rate)
        self.backframes = int(backsize*self.rate)
        self.init_buffer()
        self.offset = 0
        self.close = self._close_fishgrid
        self.load_audio_buffer = self._load_buffer_fishgrid
        self.basename = self._basename_fishgrid
        return self

    def _close_fishgrid(self):
        """Close the fishgrid data files.
        """
        for file in self.sf:
            file.close()
        self.sf = []

    def _load_buffer_fishgrid(self, r_offset, r_size, buffer):
        """Load new data from relacs data file.

        Parameters
        ----------
        r_offset: int
           First frame to be read from file.
        r_size: int
           Number of frames to be read from file.
        buffer: ndarray
           Buffer where to store the loaded data.
        """
        if len(self.sf) == 0 and len(self.trace_filepaths) > 0:
            for path in self.trace_filepaths:
                self.sf.append(open(path, 'rb'))
        for file, gchannels, goffset in zip(self.sf, self.grid_channels, self.grid_offs):
            file.seek(r_offset*4*gchannels)
            data = file.read(r_size*4*gchannels)
            buffer[:, goffset:goffset+gchannels] = np.frombuffer(data, dtype=np.float32).reshape((-1, gchannels))

    def _basename_fishgrid(self, path=None):
        """ Base name of the fishgrid data files.

        Parameters
        ----------
        path: str or None
            Path of a fishgrid data file
            (*.raw, fishgrid.cfg, or just the directory).
            If `None`, use `self.filepath`.

        Returns
        -------
        s: str
            The base name, i.e. the name of the directory containing the
            fishgrid data files.

        """
        if path is None:
            path = self.filepath
        path = Path(path)
        if path.is_dir():
            return path.name
        else:
            return path.parent.name



    # container interface:
    def open_container(self, filepath, buffersize=10.0,
                       backsize=0.0, verbose=0, datakey=None,
                       samplekey=['rate', 'Fs', 'fs'],
                       timekey=['time'], amplkey=['amax'], unitkey='unit',
                       metadatakey=['metadata', 'info'],
                       poskey=['positions'],
                       spanskey=['spans'], labelskey=['labels'],
                       descrkey=['descriptions'],
                       amax=1.0, unit='a.u.'):
        """Open generic container file.

        Supported file formats are:

        - python pickle files (.pkl)
        - numpy files (.npz)
        - matlab files (.mat)

        Parameters
        ----------
        filepath: str
            Path to a container file.
        buffersize: float
            Size of internal buffer in seconds.
        backsize: float
            Part of the buffer to be loaded before the requested start index in seconds.
        verbose: int
            If > 0 show detailed error/warning messages.
        datakey: None, str, or list of str
            Name of the variable holding the data.  If `None` take the
            variable that is an 2D array and has the largest number of
            elements.
        samplekey: str or list of str
            Name of the variable holding the sampling rate.
        timekey: str or list of str
            Name of the variable holding sampling times.
            If no sampling rate is available, the sampling rate is retrieved
            from the sampling times.
        amplkey: str or list of str
            Name of the variable holding the amplitude range of the data.
        unitkey: str
            Name of the variable holding the unit of the data.
        metadatakey: str or list of str
            Name of the variable holding the metadata.
        poskey: str or list of str
            Name of the variable holding positions of markers.
        spanskey: str or list of str
            Name of the variable holding spans of markers.
        labelskey: str or list of str
            Name of the variable holding labels of markers.
        descrkey: str or list of str
            Name of the variable holding descriptions of markers.
        amax: None or float
            If specified and no amplitude range has been found in the data
            container, then this is the amplitude range of the data.
        unit: None or str
            If specified and no unit has been found in the data container,
            then return this as the unit of the data.

        Raises
        ------
        ValueError:
            Invalid key requested.
        """
        self.verbose = verbose
        data_dict = {}
        ext = os.path.splitext(filepath)[1]
        if ext == '.pkl':
            import pickle
            with open(filepath, 'rb') as f:
                data_dict = pickle.load(f)
            self.format = 'PKL'
        elif ext == '.npz':
            data_dict = np.load(filepath)
            self.format = 'NPZ'
        elif ext == '.mat':
            from scipy.io import loadmat
            data_dict = loadmat(filepath, squeeze_me=True)
            self.format = 'MAT'
        self.buffer, self.rate, self.unit, amax = \
            extract_container_data(data_dict, datakey, samplekey,
                                   timekey, amplkey, unitkey, amax, unit)
        self.filepath = filepath
        self.file_paths = [self.filepath]
        self.file_indices = [0]
        self.channels = self.buffer.shape[1]
        self.frames = self.buffer.shape[0]
        self.shape = self.buffer.shape
        self.ndim = self.buffer.ndim
        self.size = self.buffer.size
        self.encoding = self.numpy_encodings[self.buffer.dtype]
        self.ampl_min = -amax
        self.ampl_max = +amax
        self.offset = 0
        self.buffer_changed = np.zeros(self.channels, dtype=bool)
        self.bufferframes = self.frames
        self.backsize = 0
        self.close = self._close_container
        self.load_audio_buffer = self._load_buffer_container
        self._metadata = extract_container_metadata(data_dict, metadatakey)
        self._load_metadata = None
        self._locs, self._labels = extract_container_markers(data_dict,
                                                             poskey,
                                                             spanskey,
                                                             labelskey,
                                                             descrkey)
        self._load_markers = None

    def _close_container(self):
        """Close container. """
        pass

    def _load_buffer_container(self, r_offset, r_size, buffer):
        """Load new data from container."""
        buffer[:, :] = self.buffer[r_offset:r_offset + r_size, :]


    # raw data interface:
    def open_raw(self, filepath, buffersize=10.0, backsize=0.0,
                 verbose=0, rate=44000, channels=1, dtype=np.float32,
                 amax=1.0, unit='a.u.'):
        """Load data from a raw file.

        Raw files just contain the data and absolutely no metadata, not
        even the smapling rate, number of channels, etc.
        Supported file formats are:

        - raw files (*.raw)
        - LabView scandata (*.scandat)

        Parameters
        ----------
        filepath: str
            Path of the file to load.
        buffersize: float
            Size of internal buffer in seconds.
        backsize: float
            Part of the buffer to be loaded before the requested start index in seconds.
        verbose: int
            If > 0 show detailed error/warning messages.
        rate: float
            Sampling rate of the data in Hertz.
        channels: int
            Number of channels multiplexed in the data.
        dtype: str or numpy.dtype
            The data type stored in the file.
        amax: float
            The amplitude range of the data.
        unit: str
            The unit of the data.
        """
        self.verbose = verbose
        self.filepath = filepath
        self.file_paths = [self.filepath]
        self.file_indices = [0]
        self.sf = open(self.filepath, 'rb')
        if verbose > 0:
            print(f'open_raw(filepath) with filepath={filepath}')
        self.dtype = np.dtype(dtype)
        self.rate = float(rate)
        # file size:
        self.sf.seek(0, os.SEEK_END)
        self.frames = self.sf.tell()//self.dtype.itemsize
        self.sf.seek(0)
        self.channels = int(channels)
        self.shape = (self.frames, self.channels)
        self.ndim = len(self.shape)
        self.size = self.frames*self.channels
        self.format = 'RAW'
        self.encoding = self.numpy_encodings.get(self.dtype, 'UNKNOWN')
        self.unit = unit
        self.ampl_max = float(amax)
        self.ampl_min = -self.ampl_max
        self.offset = 0
        self.bufferframes = int(buffersize*self.rate)
        self.backframes = int(backsize*self.rate)
        self.init_buffer()
        self.close = self._close_raw
        self.load_audio_buffer = self._load_buffer_raw
        self._metadata = None
        self._load_metadata = None
        self._locs = None
        self._labels = None
        self._load_markers = None

    def _close_raw(self):
        """Close raw file. """
        self.sf.close()
        self.sf = None

    def _load_buffer_raw(self, r_offset, r_size, buffer):
        """Load new data from container."""
        if self.sf is None:
            self.sf = open(self.filepath, 'rb')
        self.sf.seek(r_offset*self.dtype.itemsize)
        raw_data = self.sf.read(r_size*self.dtype.itemsize)
        raw_data = np.frombuffer(raw_data, dtype=self.dtype)
        raw_data = raw_data.reshape(-1, self.channels)
        # recode:
        if self.dtype == np.dtype('int16'):
            data = raw_data.astype('float32')
            data *= self.ampl_max/2**15
        elif self.dtype == np.dtype('int32'):
            data = raw_data.astype(float)
            data *= self.ampl_max/2**31
        elif self.dtype == np.dtype('int64'):
            data = raw_data.astype(float)
            data *= self.ampl_max/2**63
        else:
            data = raw_data
        buffer[:, :] = data

    
    # audioio interface:        
    def open_audioio(self, filepath, buffersize=10.0, backsize=0.0,
                     verbose=0, gainkey=default_gain_keys, sep='.',
                     amax=None, unit='a.u.'):
        """Open an audio file.

        See the [audioio](https://github.com/bendalab/audioio) package
        for details.

        Parameters
        ----------
        filepath: str
            Path to an audio file.
        buffersize: float
            Size of internal buffer in seconds.
        backsize: float
            Part of the buffer to be loaded before the requested start index
            in seconds.
        verbose: int
            If > 0 show detailed error/warning messages.
        gainkey: str or list of str
            Key in the file's metadata that holds some gain information.
            If found, the data will be multiplied with the gain,
            and if available, the corresponding unit is returned.
            See the [audioio.get_gain()](https://bendalab.github.io/audioio/api/audiometadata.html#audioio.audiometadata.get_gain) function for details.
        sep: str
            String that separates section names in `gainkey`.
        amax: None or float
            If specified and no gain has been found in the metadata,
            then use this as the amplitude range.
        unit: None or str
            If specified and no gain has been found in the metadata,
            then this is the unit of the data.

        """
        self.verbose = verbose
        super(DataLoader, self).open(filepath, buffersize, backsize, verbose)
        md = self.metadata()
        fac, unit = get_gain(md, gainkey, sep, amax, unit)
        if fac is None:
            self.gain_fac = 1.0 
        else:
            self.gain_fac = fac
            self._load_buffer_audio_org = self.load_audio_buffer
            self.load_audio_buffer = self._load_buffer_audioio
        self.ampl_min *= self.gain_fac
        self.ampl_max *= self.gain_fac
        self.unit = unit
        return self
    
    def _load_buffer_audioio(self, r_offset, r_size, buffer):
        """Load and scale new data from an audio file.

        Parameters
        ----------
        r_offset: int
           First frame to be read from file.
        r_size: int
           Number of frames to be read from file.
        buffer: ndarray
           Buffer where to store the loaded data.
        """
        self._load_buffer_audio_org(r_offset, r_size, buffer)
        buffer *= self.gain_fac


    # open multiple files as one:
    def open_multiple(self, filepaths, buffersize=10.0, backsize=0.0,
                      verbose=0, rate=None, channels=None,
                      unit=None, amax=None, end_indices=None):
        """Open multiple files as a single concatenated array.

        Parameters
        ----------
        filepaths: list of str
            List of file names of audio files.
        buffersize: float
            Size of internal buffer in seconds.
        backsize: float
            Part of the buffer to be loaded before the requested start index in seconds.
        verbose: int
            If larger than zero show detailed error/warning messages.
        rate: float
            If provided, do a minimal initialization (no checking)
            using the provided sampling rate (in Hertz), channels,
            unit, maximum amplitude, and end_indices.
        channels: int
            If provided, do a minimal initialization (no checking)
            using the provided rate, number of channels,
            unit, maximum amplitude, and end_indices.
        unit: str
            If provided, do a minimal initialization (no checking)
            using the provided rate, number of channels,
            unit, maximum amplitude, and end_indices.
        amax: float
            If provided, do a minimal initialization (no checking)
            using the provided rate, number of channels,
            unit, maximum amplitude amax, and end_indices.
        end_indices: sequence of int
            If provided, do a minimal initialization (no checking)
            using the provided rate, channels,
            unit, maximum amplitude, and end_indices.

        Raises
        ------
        TypeError
            `filepaths` must be a sequence.
        ValueError
            Empty `filepaths`.
        FileNotFoundError
            `filepaths` does not contain a single valid file.

        """
        if not isinstance(filepaths, (list, tuple, np.ndarray)):
            raise TypeError('input argument filepaths is not a sequence!')
        if len(filepaths) == 0:
            raise ValueError('input argument filepaths is empy sequence!')
        self.buffersize = buffersize
        self.backsize = backsize
        self.filepath = None
        self.file_paths = []
        self.open_files = []
        self.open_loaders = []
        self.data_files = []
        self.collect_counter = 0
        self.frames = 0
        self.start_indices = []
        self.end_indices = []
        self.start_time = None
        start_time = None
        self._metadata = {}
        self._locs = np.zeros((0, 2), dtype=int)
        self._labels = np.zeros((0, 2), dtype=object)
        if end_indices is not None:
            self.filepath = filepaths[0]
            self.file_paths = filepaths
            self.data_files = [None] * len(filepaths)
            self.frames = end_indices[-1]
            self.start_indices = [0] + list(end_indices[:-1])
            self.end_indices = end_indices
            self.format = None
            self.encoding = None
            self.rate = rate
            self.channels = channels
            self.unit = unit
            self.ampl_max = amax
            self.ampl_min = -amax
        else:
            for filepath in filepaths:
                try:
                    a = DataLoader(filepath, buffersize, backsize, verbose)
                except Exception as e:
                    if verbose > 0:
                        print(e)
                    continue
                # collect metadata:
                md = a.metadata()
                fmd = flatten_metadata(md, True)
                add_metadata(self._metadata, fmd)
                if self.filepath is None:
                    # first file:
                    self.filepath = a.filepath
                    self.format = a.format
                    self.encoding = a.encoding
                    self.rate = a.rate
                    self.channels = a.channels
                    self.unit = a.unit
                    self.ampl_max = a.ampl_max
                    self.ampl_min = a.ampl_min
                    self.start_time = get_datetime(md)
                    start_time = self.start_time
                else:
                    # check channels, rate, and amplitudes:
                    error_str = None
                    if a.channels != self.channels:
                        error_str = f'number of channels differs: ' \
                                    f'{a.channels} in {a.filepath} versus ' \
                                    f'{self.channels} in {self.filepath}'
                    if a.rate != self.rate:
                        error_str = f'sampling rates differ: ' \
                                    f'{a.rate} in {a.filepath} versus ' \
                                    f'{self.rate} in {self.filepath}'
                    if a.ampl_min != self.ampl_min:
                        error_str = f'minimum amplitudes differ: ' \
                                    f'{a.ampl_min} in {a.filepath} versus ' \
                                    f'{self.ampl_min} in {self.filepath}'
                    if a.ampl_max != self.ampl_max:
                        error_Str = f'maximum amplitudes differ: ' \
                                    f'{a.ampl_max} in {a.filepath} versus ' \
                                    f'{self.ampl_max} in {self.filepath}'
                    # check start time of recording:
                    stime = get_datetime(md)
                    if start_time is None or stime is None or \
                       abs(start_time - stime) > timedelta(seconds=1):
                        error_str = f'start time does not indicate continuous recording: ' \
                                    f'expected {start_time} instead of ' \
                                    f'{stime} in {a.filepath}'
                    if error_str is not None:
                        if verbose > 0:
                            print(error_str)
                        a.close()
                        del a
                        break
                # markers:
                locs, labels = a.markers()
                locs[:,0] += self.frames
                self._locs = np.vstack((self._locs, locs))
                self._labels = np.vstack((self._labels, labels))
                # indices:
                self.start_indices.append(self.frames)
                self.frames += a.frames
                self.end_indices.append(self.frames)
                if start_time is not None:
                    start_time += timedelta(seconds=a.frames/a.rate)
                # add file to lists:
                self.file_paths.append(filepath)
                if len(self.open_files) < AudioLoader.max_open_files:
                    self.open_files.append(a)
                else:
                    a.close()
                if len(self.open_loaders) < AudioLoader.max_open_loaders:
                    self.data_files.append(a)
                    self.open_loaders.append(a)
                else:
                    a.close()
                    del a
                    self.data_files.append(None)
            if len(self.data_files) == 0:
                raise FileNotFoundError('input argument filepaths does not contain any valid audio file!')
            # set startime from first file:
            if self.start_time is not None:
                set_starttime(self._metadata, self.start_time)
        # setup infrastructure:
        self.file_indices = self.start_indices
        self.start_indices = np.array(self.start_indices)
        self.end_indices = np.array(self.end_indices)
        self.shape = (self.frames, self.channels)
        self.bufferframes = int(buffersize*self.rate)
        self.backframes = int(backsize*self.rate)
        self.init_buffer()
        self.close = self._close_multiple
        self.load_audio_buffer = self._load_buffer_multiple
        self._load_metadata = None
        self._load_markers = None
        return self

    def _close_multiple(self):
        """Close all the data files. """
        self.open_files = []
        self.open_loaders = []
        if hasattr(self, 'data_files'):
            for a in self.data_files:
                if a is not None:
                    a.close()
        self.data_files = []
        self.filepath = None
        self.file_paths = []
        self.file_indices = []
        self.start_indices = []
        self.end_indices = []
        del self.data_files
        del self.open_files
        del self.open_loaders
        del self.start_indices
        del self.end_indices

    def _load_buffer_multiple(self, r_offset, r_size, buffer):
        """Load new data from the underlying files.

        Parameters
        ----------
        r_offset: int
           First frame to be read from file.
        r_size: int
           Number of frames to be read from file.
        buffer: ndarray
           Buffer where to store the loaded data.
        """
        offs = r_offset
        size = r_size
        boffs = 0
        ai = np.searchsorted(self.end_indices, offs, side='right')
        while size > 0:
            if self.data_files[ai] is None:
                a = DataLoader(self.file_paths[ai],
                                self.buffersize, self.backsize, 0)
                self.data_files[ai] = a
                self.open_loaders.append(a)
                self.open_files.append(a)
                if len(self.open_files) > AudioLoader.max_open_files:
                    a0 = self.open_files.pop(0)
                    a0.close()
                if len(self.open_loaders) > AudioLoader.max_open_loaders:
                    a0 = self.open_loaders.pop(0)
                    self.data_files[self.data_files.index(a0)] = None
                    a0.close()
                    del a0
                    self.collect_counter += 1
                    if self.collect_counter > AudioLoader.max_open_loaders//2:
                        gc.collect()   # takes time!
                        self.collect_counter = 0
            else:
                self.open_loaders.pop(self.open_loaders.index(self.data_files[ai]))
                self.open_loaders.append(self.data_files[ai])
            ai0 = offs - self.start_indices[ai]
            ai1 = offs + size
            if ai1 > self.end_indices[ai]:
                ai1 = self.end_indices[ai]
            ai1 -= self.start_indices[ai]
            n = ai1 - ai0
            self.data_files[ai].load_audio_buffer(ai0, n,
                                                  buffer[boffs:boffs + n,:])
            if self.data_files[ai] in self.open_files:
                self.open_files.pop(self.open_files.index(self.data_files[ai]))
            self.open_files.append(self.data_files[ai])
            if len(self.open_files) > AudioLoader.max_open_files:
                self.open_files[0].close()
                self.open_files.pop(0)
            boffs += n
            offs += n
            size -= n
            ai += 1

        
    def open(self, filepath, buffersize=10.0, backsize=0.0,
             verbose=0, **kwargs):
        """Open file with time-series data for reading.

        Parameters
        ----------
        filepath: str or list of str
            Name of the file or list of many file names that should be
            made accessible as a single array.
        buffersize: float
            Size of internal buffer in seconds.
        backsize: float
            Part of the buffer to be loaded before the requested start index
            in seconds.
        verbose: int
            If > 0 show detailed error/warning messages.
        **kwargs: dict
            Further keyword arguments that are passed on to the 
            format specific opening functions.
            For example:
            - `amax`: the amplitude range of the data.
            - 'unit': the unit of the data.

        Raises
        ------
        ValueError:
            `filepath` is empty string.
        """
        # list of implemented open functions:
        data_open_funcs = (
            ('relacs', check_relacs, self.open_relacs, 1),
            ('fishgrid', check_fishgrid, self.open_fishgrid, 1),
            ('container', check_container, self.open_container, 1),
            ('raw', check_raw, self.open_raw, 1),
            ('audioio', None, self.open_audioio, 0),
            )
        
        self.buffer = np.array([])
        self.rate = 0.0
        if not filepath:
            raise ValueError('input argument filepath is empty string.')
        if isinstance(filepath, (list, tuple, np.ndarray)):
            if len(filepath) > 1:
                self.open_multiple(filepath, buffersize, backsize,
                                   verbose, **kwargs)
                if len(self.file_paths) > 1:
                    return self
                filepath = self.file_paths[0]
                self.close()
            else:
                filepath = filepath[0]
        # open data:
        for name, check_file, open_file, v in  data_open_funcs:
            if check_file is None or check_file(filepath):
                open_file(filepath, buffersize, backsize, verbose, **kwargs)
                if v*verbose > 1:
                    if self.format is not None:
                        print(f'  format       : {self.format}')
                    if self.encoding is not None:
                        print(f'  encoding     : {self.encoding}')
                    print(f'  sampling rate: {self.rate} Hz')
                    print(f'  channels     : {self.channels}')
                    print(f'  frames       : {self.frames}')
                    print(f'  range        : {self.ampl_max:g}{self.unit}')
                break
        return self


def demo(filepath, plot=False):
    print("try load_data:")
    data, rate, unit, amax = load_data(filepath, verbose=2)
    if plot:
        fig, ax = plt.subplots()
        time = np.arange(len(data))/rate
        for c in range(data.shape[1]):
            ax.plot(time, data[:,c])
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'[{unit}]')
        if amax is not None and np.isfinite(amax):
            ax.set_ylim(-amax, +amax)
        plt.show()
        return

    print('')
    print("try DataLoader:")
    with DataLoader(filepath, 2.0, 1.0, 1) as data:
        print('sampling rate: %g' % data.rate)
        print('frames       : %d %d' % (len(data), data.shape[0]))
        nframes = int(1.0 * data.rate)
        # forward:
        for i in range(0, len(data), nframes):
            print('forward %d-%d' % (i, i + nframes))
            x = data[i:i + nframes, 0]
            if plot:
                fig, ax = plt.subplots()
                ax.plot((i + np.arange(len(x)))/data.rate, x)
                ax.set_xlabel('Time [s]')
                ax.set_ylabel(f'[{data.unit}]')
                plt.show()
        # and backwards:
        for i in reversed(range(0, len(data), nframes)):
            print('backward %d-%d' % (i, i + nframes))
            x = data[i:i + nframes, 0]
            if plot:
                fig, ax = plt.subplots()
                ax.plot((i + np.arange(len(x)))/data.rate, x)
                ax.set_xlabel('Time [s]')
                ax.set_ylabel(f'[{data.unit}]')
                plt.show()
                
    
def main(*cargs):
    """Call demo with command line arguments.

    Parameters
    ----------
    cargs: list of str
        Command line arguments as provided by sys.argv[1:]
    """
    import argparse
    parser = argparse.ArgumentParser(description=
                                     'Checking thunderlab.dataloader module.')
    parser.add_argument('-p', dest='plot', action='store_true',
                        help='plot loaded data')
    parser.add_argument('file', nargs=1, default='', type=str,
                        help='name of data file')
    args = parser.parse_args(cargs)
    demo(args.file[0], args.plot)
    

if __name__ == "__main__":
    main(*sys.argv[1:])
