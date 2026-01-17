"""Writing numpy arrays of floats to data files.

- `write_data()`: write data into a file.
- `available_formats()`: supported data and audio file formats.
- `available_encodings()`: encodings of a data file format.
- `format_from_extension()`: deduce data file format from file extension.
- `recode_array()`: recode array of floats.
- `insert_container_metadata()`: insert flattened metadata to data dictionary for a container file format.
"""

import sys
import datetime as dt

from pathlib import Path
from copy import deepcopy
from audioio import find_key, add_metadata, move_metadata
from audioio import get_datetime, default_gain_keys

data_modules = {}
"""Dictionary with availability of various modules needed for writing data.
Keys are the module names, values are booleans.
"""

try:
    import pickle
    data_modules['pickle'] = True
except ImportError:
    data_modules['pickle'] = False

try:
    import numpy as np
    data_modules['numpy'] = True
except ImportError:
    data_modules['numpy'] = False

try:
    import scipy.io as sio
    data_modules['scipy'] = True
except ImportError:
    data_modules['scipy'] = False

try:
    import audioio.audiowriter as aw
    import audioio.audiometadata as am
    from audioio import write_metadata_text, flatten_metadata
    data_modules['audioio'] = True
except ImportError:
    data_modules['audioio'] = False


def format_from_extension(filepath):
    """Deduce data file format from file extension.

    Parameters
    ----------
    filepath: str or Path or None
        Path and name of the data file.

    Returns
    -------
    format: str
        Data format deduced from file extension.
    """
    if filepath is None:
        return None
    filepath = Path(filepath)
    ext = filepath.suffix
    if not ext:
        return None
    if ext[0] == '.':
        ext = ext[1:]
    if not ext:
        return None
    ext = ext.upper()
    if data_modules['audioio']:
        ext = aw.format_from_extension(filepath)
    return ext


def recode_array(data, amax, encoding):
    """Recode array of floats.

    Parameters
    ----------
    data: array of floats
        Data array with values ranging between -1 and 1
    amax: float
        Maximum amplitude of data range.
    encoding: str
        Encoding, one of PCM_16, PCM_32, PCM_64, FLOAT or DOUBLE.

    Returns
    -------
    buffer: array
        The data recoded according to `encoding`.
    """
    
    encodings = {'PCM_16': (2, 'i2'),
                 'PCM_32': (4, 'i4'),
                 'PCM_64': (8, 'i8'),
                 'FLOAT': (4, 'f'),
                 'DOUBLE': (8, 'd')}

    if not encoding in encodings:
        return data
    dtype = encodings[encoding][1]
    if dtype[0] == 'i':
        sampwidth = encodings[encoding][0]
        factor = 2**(sampwidth*8-1)
        buffer = np.round(data/amax*factor).astype(dtype)
        buffer[data >= +amax] = factor - 1
        buffer[data <= -amax] = -(factor - 1)
    else:
        buffer = data.astype(dtype, copy=False)
    return buffer

    
def formats_relacs():
    """Data format of the relacs file format.

    Returns
    -------
    formats: list of str
        List of supported file formats as strings.
    """
    return ['RELACS']


def encodings_relacs(format=None):
    """Encodings of the relacs file format.

    Parameters
    ----------
    format: str
        The file format.

    Returns
    -------
    encodings: list of str
        List of supported encodings as strings.
    """
    if not format:
        format = 'RELACS'
    if format.upper() != 'RELACS':
        return []
    else:
        return ['FLOAT']
    
    
def write_relacs(filepath, data, rate, amax=1.0, unit=None,
                 metadata=None, locs=None, labels=None, format=None,
                 encoding=None):
    """Write data as relacs raw files.

    Parameters
    ----------
    filepath: str or Path
        Full path of folder where to write relacs files.
    data: 1-D or 2-D array of floats
        Array with the data (first index time, optional second index channel).
    rate: float
        Sampling rate of the data in Hertz.
    amax: float
        Maximum possible amplitude of the data in `unit`.
    unit: str
        Unit of the data.
    metadata: nested dict
        Additional metadata saved into `info.dat`.
    locs: None or 1-D or 2-D array of ints
        Marker positions (first column) and spans (optional second column)
        for each marker (rows).
    labels: None or 2-D array of string objects
        Labels (first column) and texts (optional second column)
        for each marker (rows).
    format: str or None
        File format, only None or 'RELACS' are supported.
    encoding: str or None
        Encoding of the data. Only None or 'FLOAT' are supported.

    Returns
    -------
    filepath: Path
        The actual folder used for writing the data.

    Raises
    ------
    ValueError
        File format or encoding not supported.
    """
    if format is None:
        format = 'RELACS'
    if format.upper() != 'RELACS':
        raise ValueError(f'file format {format} not supported by relacs file format')
    if encoding is None:
        encoding = 'FLOAT'
    if encoding.upper() != 'FLOAT':
        raise ValueError(f'file encoding {encoding} not supported by relacs file format')
    filepath = Path(filepath)
    if not filepath.exists():
        filepath.mkdir()
    # write data:
    if data.ndim == 1:
        with open(filepath / f'trace-1.raw', 'wb') as df:
            df.write(data.astype(np.float32).tobytes())
    else:
        for c in range(data.shape[1]):
            with open(filepath / f'trace-{c+1}.raw', 'wb') as df:
                df.write(data[:, c].astype(np.float32).tobytes())
    if unit is None:
        unit = 'V'
    # write data format:
    df = open(filepath / 'stimuli.dat', 'w')
    df.write('# analog input traces:\n')
    for c in range(data.shape[1] if data.ndim > 1 else 1):
        df.write(f'#     identifier{c+1}      : V-{c+1}\n')
        df.write(f'#     data file{c+1}       : trace-{{c+1}}.raw\n')
        df.write(f'#     sample interval{c+1} : {1000.0/rate:.4f}ms\n')
        df.write(f'#     sampling rate{c+1}   : {rate:.2f}Hz\n')
        df.write(f'#     unit{c+1}            : {unit}\n')
    df.write('# event lists:\n')
    df.write('#      event file1: stimulus-events.dat\n')
    df.write('#      event file2: restart-events.dat\n')
    df.write('#      event file3: recording-events.dat\n')
    df.close()
    # write empty event files:
    for events in ['Recording', 'Restart', 'Stimulus']:
        df = open(filepath / f'{events.lower()}-events.dat', 'w')
        df.write(f'# events: {events}\n\n')
        df.write('#Key\n')
        if events == 'Stimulus':
            df.write('# t    duration\n')
            df.write('# sec  s\n')
            df.write('#   1         2\n')
        else:
            df.write('# t\n')
            df.write('# sec\n')
            df.write('# 1\n')
            if events == 'Recording':
                df.write('  0.0\n')
        df.close()
    # write metadata:
    if metadata:
        write_metadata_text(filepath / 'info.dat',
                            metadata, prefix='# ')
    return filepath

    
def formats_fishgrid():
    """Data format of the fishgrid file format.

    Returns
    -------
    formats: list of str
        List of supported file formats as strings.
    """
    return ['FISHGRID']


def encodings_fishgrid(format=None):
    """Encodings of the fishgrid file format.

    Parameters
    ----------
    format: str
        The file format.

    Returns
    -------
    encodings: list of str
        List of supported encodings as strings.
    """
    if not format:
        format = 'FISHGRID'
    if format.upper() != 'FISHGRID':
        return []
    else:
        return ['FLOAT']
    
    
def write_fishgrid(filepath, data, rate, amax=1.0, unit=None,
                   metadata=None, locs=None, labels=None, format=None,
                   encoding=None):
    """Write data as fishgrid raw files.

    Parameters
    ----------
    filepath: str or Path
        Full path of the folder where to write fishgrid files.
    data: 1-D or 2-D array of floats
        Array with the data (first index time, optional second index channel).
    rate: float
        Sampling rate of the data in Hertz.
    amax: float
        Maximum possible amplitude of the data in `unit`.
    unit: str
        Unit of the data.
    metadata: nested dict
        Additional metadata saved into the `fishgrid.cfg`.
    locs: None or 1-D or 2-D array of ints
        Marker positions (first column) and spans (optional second column)
        for each marker (rows).
    labels: None or 2-D array of string objects
        Labels (first column) and texts (optional second column)
        for each marker (rows).
    format: str or None
        File format, only None or 'FISHGRID' are supported.
    encoding: str or None
        Encoding of the data. Only None or 'FLOAT' are supported.

    Returns
    -------
    filepath: Path
        The actual folder used for writing the data.

    Raises
    ------
    ValueError
        File format or encoding not supported.
    """
    def write_timestamp(df, count, index, span, rate, starttime,
                        label, comment):
        datetime = starttime + dt.timedelta(seconds=index/rate)
        df.write(f'    Num: {count}\n')
        df.write(f' Index1: {index}\n')
        #df.write(f' Index2: 0\n')
        #df.write(f' Index3: 0\n')
        #df.write(f' Index4: 0\n')
        if span > 0:
            df.write(f'  Span1: {span}\n')
        df.write(f'   Date: {datetime.date().isoformat()}\n')
        df.write(f'   Time: {datetime.time().isoformat(timespec="seconds")}\n')
        if label:
            df.write(f'  Label: {label}\n')
        df.write(f'Comment: {comment}\n')
        df.write('\n')
        
    if format is None:
        format = 'FISHGRID'
    if format.upper() != 'FISHGRID':
        raise ValueError(f'file format {format} not supported by fishgrid file format')
    if encoding is None:
        encoding = 'FLOAT'
    if encoding.upper() != 'FLOAT':
        raise ValueError(f'file encoding {encoding} not supported by fishgrid file format')
    filepath = Path(filepath)
    if not filepath.exists():
        filepath.mkdir()
    # write data:
    with open(filepath / 'traces-grid1.raw', 'wb') as df:
        df.write(data.astype(np.float32).tobytes())
    # write metadata:
    if unit is None:
        unit = 'mV'
    cfgfile = filepath / 'fishgrid.cfg'
    nchannels = data.shape[1] if data.ndim > 1 else 1
    ncols = int(np.ceil(np.sqrt(nchannels)))
    nrows = int(np.ceil(nchannels/ncols))
    if metadata is None:
        metadata = {}
    if 'FishGrid' in metadata:
        md = {}
        rmd = {}
        for k in metadata:
            if isinstance(metadata[k], dict):
                md[k] = deepcopy(metadata[k])
            else:
                rmd[k] = metadata[k]
        if len(rmd) > 0:
            m, k = find_key(md, 'FishGrid.Recording')
            if k in m:
                m[k].update(rmd)
            else:
                m[k] = rmd
    else:
        smd = deepcopy(metadata)
        gm = dict(Used1='true', Columns1=f'{ncols}', Rows1=f'{nrows}')
        hm = {'DAQ board': dict()}
        if not move_metadata(smd, hm, 'Amplifier'):
            am = {}
            move_metadata(smd, am, ['Amplifier.Name', 'AmplName'], 'AmplName')
            move_metadata(smd, am, ['Amplifier.Model', 'AmplModel'], 'AmplModel')
            move_metadata(smd, am, 'Amplifier.Type')
            move_metadata(smd, am, 'Gain')
            move_metadata(smd, am, 'HighpassCutoff')
            move_metadata(smd, am, 'LowpassCutoff')
            if len(am) > 0:
                hm['Amplifier'] = am
        md = dict(FishGrid={'Grid 1': gm, 'Hardware Settings': hm})
        move_metadata(smd, md['FishGrid'], 'Recording')
        gm = {}
        starttime = get_datetime(smd, remove=True)
        if not starttime is None:
            gm['StartDate'] = starttime.date().isoformat()
            gm['StartTime'] = starttime.time().isoformat(timespec='seconds')
        move_metadata(smd, gm, 'Location')
        move_metadata(smd, gm, 'Position')
        move_metadata(smd, gm, 'WaterTemperature')
        move_metadata(smd, gm, 'WaterConductivity')
        move_metadata(smd, gm, 'WaterpH')
        move_metadata(smd, gm, 'WaterOxygen')
        move_metadata(smd, gm, 'Temperature')
        move_metadata(smd, gm, 'Humidity')
        move_metadata(smd, gm, 'Pressure')
        move_metadata(smd, gm, 'Comment')
        move_metadata(smd, gm, 'Experimenter')
        if len(gm) > 0:
            if not 'Recording' in md['FishGrid']:
                md['FishGrid']['Recording'] = {}
            md['FishGrid']['Recording'].update({'General': gm})
        bm = {}
        move_metadata(smd, bm, 'DataTime')
        move_metadata(smd, bm, 'DataInterval')
        move_metadata(smd, bm, 'BufferTime')
        move_metadata(smd, bm, 'BufferInterval')
        if len(bm) > 0:
            if not 'Recording' in md['FishGrid']:
                md['FishGrid']['Recording'] = {}
            md['FishGrid']['Recording'].update({'Buffers and timing': bm})
        if smd:
            md['FishGrid']['Other'] = smd
    add_metadata(md,
                 [f'FishGrid.Hardware Settings.DAQ board.AISampleRate={0.001*rate:.3f}kHz',
                  f'FishGrid.Hardware Settings.DAQ board.AIMaxVolt={amax:g}{unit}'])
    with open(cfgfile, 'w') as df:
        for k in md:
            df.write(f'*{k}\n')
            write_metadata_text(df, md[k], prefix='  ')
    # write markers:
    filename = filepath / 'timestamps.dat'
    starttime = get_datetime(metadata, (('DateTimeOriginal',),
                                        ('OriginationDate', 'OriginationTime'),
                                        ('StartDate', 'StartTime'),
                                        ('Location_Time',)),
                             default=dt.datetime.fromtimestamp(0, dt.timezone.utc))
    with open(filename, 'w') as df:
        count = 0
        write_timestamp(df, count, 0, 0, rate, starttime,
                        '', 'begin of recording')
        count += 1
        if locs is not None:
            for i in range(len(locs)):
                label = ''
                comment = ''
                if labels is not None and len(labels) > i:
                    label = labels[i,0] if labels.ndim > 1 else labels[i]
                    comment = labels[i,1] if labels.ndim > 1 else ''
                index = locs[i,0] if locs.ndim > 1 else locs[i]
                span = locs[i,1] if locs.ndim > 1 else 0
                write_timestamp(df, count, index*nchannels,
                                span*nchannels, rate,
                                starttime, label, comment)
                count += 1
        write_timestamp(df, count, len(data)*nchannels, 0, rate,
                        starttime, '', 'end of recording')
    return filepath

    
def formats_pickle():
    """Data formats supported by pickle.dump().

    Returns
    -------
    formats: list of str
        List of supported file formats as strings.
    """
    if not data_modules['pickle']:
        return []
    else:
        return ['PKL']


def encodings_pickle(format=None):
    """Encodings of the pickle format.

    Parameters
    ----------
    format: str
        The file format.

    Returns
    -------
    encodings: list of str
        List of supported encodings as strings.
    """
    if not format:
        format = 'PKL'
    if format.upper() != 'PKL':
        return []
    else:
        return ['PCM_16', 'PCM_32', 'FLOAT', 'DOUBLE']

    
def write_pickle(filepath, data, rate, amax=1.0, unit=None,
                 metadata=None, locs=None, labels=None, format=None,
                 encoding=None):
    """Write data into python pickle file.
    
    Documentation
    -------------
    https://docs.python.org/3/library/pickle.html

    Parameters
    ----------
    filepath: str or Path
        Full path and name of the file to write.
    data: 1-D or 2-D array of floats
        Array with the data (first index time, optional second index channel).
        Stored under the key "data".
    rate: float
        Sampling rate of the data in Hertz.
        Stored under the key "rate".
    amax: float
        Maximum possible amplitude of the data in `unit`.
        Stored under the key "amax".
    unit: str
        Unit of the data.
        Stored under the key "unit".
    metadata: nested dict
        Additional metadata saved into the pickle.
        Stored under the key "metadata".
    locs: None or 1-D or 2-D array of ints
        Marker positions (first column) and spans (optional second column)
        for each marker (rows).
    labels: None or 2-D array of string objects
        Labels (first column) and texts (optional second column)
        for each marker (rows).
    format: str or None
        File format, only None or 'PKL' are supported.
    encoding: str or None
        Encoding of the data.

    Returns
    -------
    filepath: Path
        The actual file name used for writing the data.

    Raises
    ------
    ImportError
        The pickle module is not available.
    ValueError
        File format or encoding not supported.
    """
    if not data_modules['pickle']:
        raise ImportError
    if format is None:
        format = 'PKL'
    if format.upper() != 'PKL':
        raise ValueError(f'file format {format} not supported by pickle file format')
    filepath = Path(filepath)
    ext = filepath.suffix
    if len(ext) <= 1 or ext[1].upper() != 'P':
        filepath = filepath.with_suffix('.pkl')
    if encoding is None:
        encoding = 'DOUBLE'
    encoding = encoding.upper()
    if not encoding in encodings_pickle(format):
        raise ValueError(f'file encoding {encoding} not supported by pickle file format')
    buffer = recode_array(data, amax, encoding)
    ddict = dict(data=buffer, rate=rate)
    ddict['amax'] = amax
    if unit:
        ddict['unit'] = unit
    if metadata:
        ddict['metadata'] = metadata
    if locs is not None and len(locs) > 0:
        if locs.ndim == 1:
            ddict['positions'] = locs
        else:
            ddict['positions'] = locs[:,0]
            if locs.shape[1] > 1:
                ddict['spans'] = locs[:,1]
    if labels is not None and len(labels) > 0:
        if labels.ndim == 1:
            ddict['labels'] = labels
        else:
            ddict['labels'] = labels[:,0]
            if labels.shape[1] > 1:
                ddict['descriptions'] = labels[:,1]
    with open(filepath, 'wb') as df:
        pickle.dump(ddict, df)
    return filepath


def insert_container_metadata(metadata, data_dict, metadatakey='metadata'):
    """Insert flattened metadata to data dictionary for a container file format.

    Parameters
    ----------
    metadata: nested dict
        Nested dictionary with key-value pairs of the meta data.
    data_dict: dict
        Dictionary of the data items contained in the container to
        which the metadata should be added.
    metadatakey: str or list of str
        Name of the variable holding the metadata.
    """
    fmeta = flatten_metadata(metadata, True, sep='__')
    for k in list(fmeta):
        fmeta[metadatakey + '__' + k] = fmeta.pop(k)
    data_dict.update(fmeta)
    

def formats_numpy():
    """Data formats supported by numpy.savez().

    Returns
    -------
    formats: list of str
        List of supported file formats as strings.
    """
    if not data_modules['numpy']:
        return []
    else:
        return ['NPZ']


def encodings_numpy(format=None):
    """Encodings of the numpy file format.

    Parameters
    ----------
    format: str
        The file format.

    Returns
    -------
    encodings: list of str
        List of supported encodings as strings.
    """
    if not format:
        format = 'NPZ'
    if format.upper() !=  'NPZ':
        return []
    else:
        return ['PCM_16', 'PCM_32', 'FLOAT', 'DOUBLE']


def write_numpy(filepath, data, rate, amax=1.0, unit=None,
                metadata=None, locs=None, labels=None, format=None,
                encoding=None):
    """Write data into numpy npz file.
    
    Documentation
    -------------
    https://numpy.org/doc/stable/reference/generated/numpy.savez.html

    Parameters
    ----------
    filepath: str or Path
        Full path and name of the file to write.
    data: 1-D or 2-D array of floats
        Array with the data (first index time, optional second index channel).
        Stored under the key "data".
    rate: float
        Sampling rate of the data in Hertz.
        Stored under the key "rate".
    amax: float
        Maximum possible amplitude of the data in `unit`.
        Stored under the key "amax".
    unit: str
        Unit of the data.
        Stored under the key "unit".
    metadata: nested dict
        Additional metadata saved into the numpy file.
        Flattened dictionary entries stored under keys
        starting with "metadata__".
    locs: None or 1-D or 2-D array of ints
        Marker positions (first column) and spans (optional second column)
        for each marker (rows).
    labels: None or 2-D array of string objects
        Labels (first column) and texts (optional second column)
        for each marker (rows).
    format: str or None
        File format, only None or 'NPZ' are supported.
    encoding: str or None
        Encoding of the data.

    Returns
    -------
    filepath: Path
        The actual file name used for writing the data.

    Raises
    ------
    ImportError
        The numpy module is not available.
    ValueError
        File format or encoding not supported.
    """
    if not data_modules['numpy']:
        raise ImportError
    if format is None:
        format = 'NPZ'
    if format.upper() not in formats_numpy():
        raise ValueError(f'file format {format} not supported by numpy file format')
    filepath = Path(filepath)
    ext = filepath.suffix
    if len(ext) <= 1 or ext[1].upper() != 'N':
        filepath = filepath.with_suffix('.npz')
    if encoding is None:
        encoding = 'DOUBLE'
    encoding = encoding.upper()
    if not encoding in encodings_numpy(format):
        raise ValueError(f'file encoding {encoding} not supported by numpy file format')
    buffer = recode_array(data, amax, encoding)
    ddict = dict(data=buffer, rate=rate)
    ddict['amax'] = amax
    if unit:
        ddict['unit'] = unit
    if metadata:
        insert_container_metadata(metadata, ddict, 'metadata')
    if locs is not None and len(locs) > 0:
        if locs.ndim == 1:
            ddict['positions'] = locs
        else:
            ddict['positions'] = locs[:,0]
            if locs.shape[1] > 1:
                ddict['spans'] = locs[:,1]
    if labels is not None and len(labels) > 0:
        if labels.ndim == 1:
            maxc = np.max([len(l) for l in labels])
            ddict['labels'] = labels.astype(dtype=f'U{maxc}')
        else:
            maxc = np.max([len(l) for l in labels[:,0]])
            ddict['labels'] = labels[:,0].astype(dtype=f'U{maxc}')
            if labels.shape[1] > 1:
                maxc = np.max([len(l) for l in labels[:,1]])
                ddict['descriptions'] = labels[:,1].astype(dtype=f'U{maxc}')
    np.savez(filepath, **ddict)
    return filepath


def formats_mat():
    """Data formats supported by scipy.io.savemat().

    Returns
    -------
    formats: list of str
        List of supported file formats as strings.
    """
    if not data_modules['scipy']:
        return []
    else:
        return ['MAT']


def encodings_mat(format=None):
    """Encodings of the matlab format.

    Parameters
    ----------
    format: str
        The file format.

    Returns
    -------
    encodings: list of str
        List of supported encodings as strings.
    """
    if not format:
        format = 'MAT'
    if format.upper() != 'MAT':
        return []
    else:
        return ['PCM_16', 'PCM_32', 'FLOAT', 'DOUBLE']


def write_mat(filepath, data, rate, amax=1.0, unit=None,
              metadata=None, locs=None, labels=None, format=None,
              encoding=None):
    """Write data into matlab file.
    
    Documentation
    -------------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.savemat.html

    Parameters
    ----------
    filepath: str or Path
        Full path and name of the file to write.
    data: 1-D or 2-D array of floats
        Array with the data (first index time, optional second index channel).
        Stored under the key "data".
    rate: float
        Sampling rate of the data in Hertz.
        Stored under the key "rate".
    amax: float
        Maximum possible amplitude of the data in `unit`.
        Stored under the key "amax".
    unit: str
        Unit of the data.
        Stored under the key "unit".
    metadata: nested dict
        Additional metadata saved into the mat file.
        Stored under the key "metadata".
    locs: None or 1-D or 2-D array of ints
        Marker positions (first column) and spans (optional second column)
        for each marker (rows).
    labels: None or 2-D array of string objects
        Labels (first column) and texts (optional second column)
        for each marker (rows).
    format: str or None
        File format, only None or 'MAT' are supported.
    encoding: str or None
        Encoding of the data.

    Returns
    -------
    filepath: Path
        The actual file name used for writing the data.

    Raises
    ------
    ImportError
        The scipy.io module is not available.
    ValueError
        File format or encoding not supported.
    """
    if not data_modules['scipy']:
        raise ImportError
    if format is None:
        format = 'MAT'
    if format.upper() not in formats_mat():
        raise ValueError(f'file format {format} not supported by matlab file format')
    filepath = Path(filepath)
    ext = filepath.suffix
    if len(ext) <= 1 or ext[1].upper() != 'M':
        filepath = filepath.with_suffix('.mat')
    if encoding is None:
        encoding = 'DOUBLE'
    encoding = encoding.upper()
    if not encoding in encodings_mat(format):
        raise ValueError(f'file encoding {encoding} not supported by matlab file format')
    buffer = recode_array(data, amax, encoding)
    ddict = dict(data=buffer, rate=rate)
    ddict['amax'] = amax
    if unit:
        ddict['unit'] = unit
    if metadata:
        insert_container_metadata(metadata, ddict, 'metadata')
    if locs is not None and len(locs) > 0:
        if locs.ndim == 1:
            ddict['positions'] = locs
        else:
            ddict['positions'] = locs[:,0]
            if locs.shape[1] > 1:
                ddict['spans'] = locs[:,1]
    if labels is not None and len(labels) > 0:
        if labels.ndim == 1:
            maxc = np.max([len(l) for l in labels])
            ddict['labels'] = labels.astype(dtype=f'U{maxc}')
        else:
            maxc = np.max([len(l) for l in labels[:,0]])
            ddict['labels'] = labels[:,0].astype(dtype=f'U{maxc}')
            if labels.shape[1] > 1:
                maxc = np.max([len(l) for l in labels[:,1]])
                ddict['descriptions'] = labels[:,1].astype(dtype=f'U{maxc}')
    sio.savemat(filepath, ddict)
    return filepath


def formats_raw():
    """Data formats supported as raw formats.

    Returns
    -------
    formats: list of str
        List of supported file formats as strings.
    """
    return ['RAW']


def encodings_raw(format=None):
    """Encodings supported for raw file formats.

    Parameters
    ----------
    format: str
        The file format.

    Returns
    -------
    encodings: list of str
        List of supported encodings as strings.
    """
    if not format:
        format = 'RAW'
    if format.upper() != 'RAW':
        return []
    else:
        return ['PCM_16', 'PCM_32', 'FLOAT', 'DOUBLE']


def write_raw(filepath, data, rate, amax=1.0, unit=None,
              metadata=None, locs=None, labels=None, format=None,
              encoding=None):
    """Write data into raw file.

    Writes just the data without sampling rate, metadata and markers.

    Parameters
    ----------
    filepath: str or Path
        Full path and name of the file to write.
    data: 1-D or 2-D array of floats
        Array with the data (first index time, optional second index channel).
    rate: float
        Sampling rate of the data in Hertz.
    amax: float
        Maximum possible amplitude of the data in `unit`.
    unit: str
        Unit of the data.
    metadata: nested dict
        Additional metadata saved into the mat file.
    locs: None or 1-D or 2-D array of ints
        Marker positions (first column) and spans (optional second column)
        for each marker (rows).
    labels: None or 2-D array of string objects
        Labels (first column) and texts (optional second column)
        for each marker (rows).
    format: str or None
        File format, only None or 'RAW' are supported.
    encoding: str or None
        Encoding of the data.

    Returns
    -------
    filepath: Path
        The actual file name used for writing the data.

    Raises
    ------
    ValueError
        File format or encoding not supported.
    """
    if format is None:
        format = 'RAW'
    if format.upper() not in formats_raw():
        raise ValueError(f'file format {format} not supported by matlab file format')
    filepath = Path(filepath)
    ext = filepath.suffix
    if len(ext) <= 1 or ext[1].upper() != 'R':
        filepath = filepath.with_suffix('.raw')
    if encoding is None:
        encoding = 'DOUBLE'
    encoding = encoding.upper()
    if not encoding in encodings_raw(format):
        raise ValueError(f'file encoding {encoding} not supported by raw file format')
    buffer = recode_array(data, amax, encoding)
    with open(filepath, 'wb') as df:
        df.write(buffer.tobytes())
    return filepath


def formats_audioio():
    """Data formats supported by audioio.

    Returns
    -------
    formats: list of str
        List of supported file formats as strings.
    """
    if not data_modules['audioio']:
        return []
    else:
        return aw.available_formats()


def encodings_audio(format):
    """Encodings of any audio format.

    Parameters
    ----------
    format: str
        The file format.

    Returns
    -------
    encodings: list of str
        List of supported encodings as strings.
    """
    if not data_modules['audioio']:
        return []
    else:
        return aw.available_encodings(format)


def write_audioio(filepath, data, rate, amax=1.0, unit=None,
                  metadata=None, locs=None, labels=None, format=None,
                  encoding=None, gainkey=default_gain_keys, sep='.'):
    """Write data into audio file.

    If a gain setting is available in the metadata, then the data are divided
    by the gain before they are stored in the audio file.
    After this operation, the data values need to range between -1 and 1,
    in particular if the data are encoded as integers
    (i.e. PCM_16, PCM_32 and PCM_64).
    Note, that this function does not check for this requirement!
    
    Documentation
    -------------
    https://bendalab.github.io/audioio/

    Parameters
    ----------
    filepath: str or Path
        Full path and name of the file to write.
    data: 1-D or 2-D array of floats
        Array with the data (first index time, optional second index channel).
    rate: float
        Sampling rate of the data in Hertz.
    amax: float
        Maximum possible amplitude of the data in `unit`.
    unit: str
        Unit of the data. If supplied and a gain is found in the metadata it
        has to match the unit of the gain. If no gain is found in the metadata
        and metadata is not None, then a gain of one with this unit is added
        to the metadata using the first key in `gainkey`.
    metadata: nested dict
        Metadata saved into the audio file. If it contains a gain,
        the gain factor is used to divide the data down into a
        range between -1 and 1.
    locs: None or 1-D or 2-D array of ints
        Marker positions (first column) and spans (optional second column)
        for each marker (rows).
    labels: None or 2-D array of string objects
        Labels (first column) and texts (optional second column)
        for each marker (rows).
    format: str or None
        File format. If None deduce file format from filepath.
        See `available_formats()` for possible values.
    encoding: str or None
        Encoding of the data. See `available_encodings()` for possible values.
        If None or empty string use 'PCM_16'.
    gainkey: str or list of str
        Key in the file's metadata that holds some gain information.
        If found, the data will be multiplied with the gain,
        and if available, the corresponding unit is returned.
        See the [audioio.get_gain()](https://bendalab.github.io/audioio/api/audiometadata.html#audioio.audiometadata.get_gain) function for details.
    sep: str
        String that separates section names in `gainkey`.

    Returns
    -------
    filepath: Path
        The actual file name used for writing the data.

    Raises
    ------
    ImportError
        The audioio module is not available.
    ValueError
        `unit` does not match gain in metadata.
    """
    if not data_modules['audioio']:
        raise ImportError
    if amax is None or not np.isfinite(amax):
        amax, u = am.get_gain(metadata, gainkey, sep, 1.0, 'a.u.')
        if not unit:
            unit = u
        elif unit != 'a.u.' and u != 'a.u.' and unit != u:
            raise ValueError(f'unit "{unit}" does not match gain unit "{u}" in metadata')
    if amax != 1.0:
        data = data / amax
        if metadata is None:
            metadata = {}
        if unit == 'a.u.':
            unit = ''
        if not isinstance(gainkey, (list, tuple, np.ndarray)):
            gainkey = [gainkey,]
        gainkey.append('Gain')
        for gk in gainkey:
            m, k = am.find_key(metadata, gk)
            if k in m:
                m[k] = f'{amax:g}{unit}'
                break
        else:
            if 'INFO' in metadata:
                metadata['INFO'][gainkey[0]] = f'{amax:g}{unit}'
            else:
                metadata[gainkey[0]] = f'{amax:g}{unit}'
    aw.write_audio(filepath, data, rate, metadata, locs, labels)
    return Path(filepath)


data_formats_funcs = (
    ('relacs', None, formats_relacs),
    ('fishgrid', None, formats_fishgrid),
    ('pickle', 'pickle', formats_pickle),
    ('numpy', 'numpy', formats_numpy),
    ('matlab', 'scipy', formats_mat),
    ('raw', None, formats_raw),
    ('audio', 'audioio', formats_audioio)
    )
"""List of implemented formats functions.

Each element of the list is a tuple with the format's name, the
module's name in `data_modules` or None, and the formats function.
"""


def available_formats():
    """Data and audio file formats supported by any of the installed modules.

    Returns
    -------
    formats: list of str
        List of supported file formats as strings.
    """
    formats = set()
    for fmt, lib, formats_func in data_formats_funcs:
        if not lib or data_modules[lib]:
            formats |= set(formats_func())
    return sorted(list(formats))


data_encodings_funcs = (
    ('relacs', encodings_relacs),
    ('fishgrid', encodings_fishgrid),
    ('pickle', encodings_pickle),
    ('numpy', encodings_numpy),
    ('matlab', encodings_mat),
    ('raw', encodings_raw),
    ('audio', encodings_audio)
    )
""" List of implemented encodings functions.

Each element of the list is a tuple with the module's name and the encodings function.
"""


def available_encodings(format):
    """Encodings of a data file format.

    Parameters
    ----------
    format: str
        The file format.

    Returns
    -------
    encodings: list of str
        List of supported encodings as strings.
    """
    for module, encodings_func in data_encodings_funcs:
        encs = encodings_func(format)
        if len(encs) > 0:
            return encs
    return []


data_writer_funcs = {
    'relacs': write_relacs,
    'fishgrid': write_fishgrid,
    'pickle': write_pickle,
    'numpy': write_numpy,
    'matlab':  write_mat,
    'raw':  write_raw,
    'audio': write_audioio
    }
"""Dictionary of implemented write functions.

Keys are the format's name and values the corresponding write
function.
"""


def write_data(filepath, data, rate, amax=1.0, unit=None,
               metadata=None, locs=None, labels=None, format=None,
               encoding=None, verbose=0, **kwargs):
    """Write data into a file.

    Parameters
    ----------
    filepath: str or Path
        Full path and name of the file to write.
        File format is determined from extension.
    data: 1-D or 2-D array of floats
        Array with the data (first index time, second index channel).
    rate: float
        Sampling rate of the data in Hertz.
    amax: float
        Maximum possible amplitude of the data in `unit`.
    unit: str
        Unit of the data.
    metadata: nested dict
        Additional metadata.
    locs: None or 1-D or 2-D array of ints
        Marker positions (first column) and spans (optional second column)
        for each marker (rows).
    labels: None or 2-D array of string objects
        Labels (first column) and texts (optional second column)
        for each marker (rows).
    format: str or None
        File format. If None deduce file format from filepath.
        See `available_formats()` for possible values.
    encoding: str or None
        Encoding of the data. See `available_encodings()` for possible values.
        If None or empty string use 'PCM_16'.
    verbose: int
        If >0 show detailed error/warning messages.
    kwargs: dict
        Additional, file format specific keyword arguments.

    Returns
    -------
    filepath: str or None
        On success, the actual file name used for writing the data.

    Raises
    ------
    ValueError
        Unspecified file format.
    IOError
        Requested file format not supported.

    Example
    -------
    ```
    import numpy as np
    from thunderlab.datawriter import write_data
    
    rate = 28000.0
    freq = 800.0
    time = np.arange(0.0, 1.0, 1/rate)     # one second
    data = 2.5*np.sin(2.0*np.p*freq*time)        # 800Hz sine wave
    md = dict(Artist='underscore_')          # metadata
    write_data('audio/file.npz', data, rate, 'mV', md)
    ```
    """
    if not format:
        format = format_from_extension(filepath)
    if not format:
        raise ValueError('unspecified file format')
    for fmt, lib, formats_func in data_formats_funcs:
        if lib and not data_modules[lib]:
            continue
        if format.upper() in formats_func():
            writer_func = data_writer_funcs[fmt]
            filepath = writer_func(filepath, data, rate, amax,
                                   unit, metadata, locs, labels,
                                   format=format, encoding=encoding,
                                   **kwargs)
            if verbose > 0:
                print(f'wrote data to file "{filepath}" using {fmt} format')
                if verbose > 1:
                    print(f'  sampling rate: {rate:g}Hz')
                    print(f'  channels     : {data.shape[1] if len(data.shape) > 1 else 1}')
                    print(f'  frames       : {len(data)}')
                    print(f'  range        : {amax:g}{unit}')
            return filepath
    raise IOError(f'file format "{format.upper()}" not supported.') 


def demo(file_path, channels=2, format=None):
    """Demo of the datawriter functions.

    Parameters
    ----------
    file_path: str
        File path of a data file.
    format: str or None
        File format to be used.
    """
    print('generate data ...')
    rate = 44100.0
    t = np.arange(0.0, 1.0, 1.0/rate)
    data = np.zeros((len(t), channels))
    for c in range(channels):
        data[:,c] = 0.1*(channels-c)*np.sin(2.0*np.pi*(440.0+c*8.0)*t)
        
    print(f"write_data('{file_path}') ...")
    write_data(file_path, data, rate, 1.0, 'mV', format=format, verbose=2)

    print('done.')
    

def main(*cargs):
    """Call demo with command line arguments.

    Parameters
    ----------
    cargs: list of str
        Command line arguments as provided by sys.argv[1:]
    """
    import argparse
    parser = argparse.ArgumentParser(description=
                                     'Checking thunderlab.datawriter module.')
    parser.add_argument('-c', dest='channels', default=2, type=int,
                        help='number of channels to be written')
    parser.add_argument('-f', dest='format', default=None, type=str,
                        help='file format')
    parser.add_argument('file', nargs=1, default='test.npz', type=str,
                        help='name of data file')
    args = parser.parse_args(cargs)
    demo(args.file[0], args.channels, args.format)
    

if __name__ == "__main__":
    main(*sys.argv[1:])

    

