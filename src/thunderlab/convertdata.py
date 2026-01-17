"""Command line script for converting, downsampling, renaming and merging data files.

```sh
convertdata -o test.wav test.raw
```
converts 'test.raw' to 'test.wav'.

The script reads all input files with `dataloader.DataLoader()`, and
writes them along with the metadata to an output file using
`datawriter.write_data()`. Thus, all formats supported by these
functions and the installed python audio modules are supported.

Run
```sh
convertdata -l
```
for a list of supported output file formats and
```sh
convertdata -f wav -l
```
for a list of supported encodings for a given output format.

Running
```sh
convertdata --help
```
prints
```text
usage: convertdata [-h] [--version] [-v] [-l] [-f FORMAT] [-e ENCODING] [-s SCALE] [-u [THRESH]]
                   [-U [THRESH]] [-d FAC] [-c CHANNELS] [-a KEY=VALUE] [-r KEY] [-n NUM] [-o OUTPATH]
                   [-i KWARGS]
                   [files ...]

Convert, downsample, rename, and merge data files.

positional arguments:
  files         one or more input files to be combined into a single output file

options:
  -h, --help    show this help message and exit
  --version     show program's version number and exit
  -v            print debug output
  -l            list supported file formats and encodings
  -f FORMAT     audio format of output file
  -e ENCODING   audio encoding of output file
  -s SCALE      scale the data by factor SCALE
  -u [THRESH]   unwrap clipped data with threshold relative to maximum of input range (default is
                0.5) and divide by two
  -U [THRESH]   unwrap clipped data with threshold relative to maximum of input range (default is
                0.5) and clip
  -d FAC        downsample by integer factor
  -c CHANNELS   comma and dash separated list of channels to be saved (first channel is 0)
  -a KEY=VALUE  add key-value pairs to metadata. Keys can have section names separated by "."
  -r KEY        remove keys from metadata. Keys can have section names separated by "."
  -n NUM        merge NUM input files into one output file
  -o OUTPATH    path or filename of output file. Metadata keys enclosed in curly braces will be
                replaced by their values from the input file
  -i KWARGS     key-word arguments for the data loader function

version 1.6.0 by Benda-Lab (2020-2025)
```

"""

import os
import sys
import argparse
import numpy as np

from pathlib import Path

from audioio import add_metadata, remove_metadata, cleanup_metadata
from audioio import bext_history_str, add_history
from audioio.audioconverter import add_arguments, parse_channels, parse_load_kwargs
from audioio.audioconverter import make_outfile, format_outfile
from audioio.audioconverter import modify_data

from .dataloader import load_data, DataLoader, markers
from .datawriter import available_formats, available_encodings
from .datawriter import format_from_extension, write_data
from .version import __version__, __year__


def check_format(format):
    """
    Check whether requested audio format is valid and supported.

    If the format is not available print an error message on console.

    Parameters
    ----------
    format: string
        Audio format to be checked.

    Returns
    -------
    valid: bool
        True if the requested audio format is valid.
    """
    if not format or format.upper() not in available_formats():
        print(f'! invalid data file format "{format}"!')
        print('run')
        print(f'> {__file__} -l')
        print('for a list of available formats')
        return False
    else:
        return True


def list_formats_encodings(data_format):
    """ List available formats or encodings.

    Parameters
    ----------
    data_format: None or str
        If provided, list encodings for this data format.
    """
    if not data_format:
        print('available file formats:')
        for f in available_formats():
            print(f'  {f}')
    else:
        if not check_format(data_format):
            sys.exit(-1)
        print(f'available encodings for {data_format} file format:')
        for e in available_encodings(data_format):
            print(f'  {e}')


def main(*cargs):
    """Command line script for converting, downsampling, renaming and
    merging data files.

    Parameters
    ----------
    cargs: list of strings
        Command line arguments as returned by sys.argv[1:].

    """
    # command line arguments:
    parser = argparse.ArgumentParser(add_help=True,
        description='Convert, downsample, rename, and merge data files.',
        epilog=f'version {__version__} by Benda-Lab (2020-{__year__})')
    add_arguments(parser)
    if len(cargs) == 0:
        cargs = None
    args = parser.parse_args(cargs)
    
    channels = parse_channels(args.channels)
    
    if args.list_formats:
        if args.data_format is None and len(args.files) > 0:
            args.data_format = args.files[0]
        list_formats_encodings(args.data_format)
        return

    if len(args.files) == 0 or len(args.files[0]) == 0:
        print('! need to specify at least one input file !')
        sys.exit(-1)
        
    # expand wildcard patterns:
    files = []
    if os.name == 'nt':
        for fn in args.files:
            files.extend(glob.glob(fn))
    else:
        files = args.files
        
    nmerge = args.nmerge
    if nmerge == 0:
        nmerge = len(args.files)

    # kwargs for audio loader:
    load_kwargs = parse_load_kwargs(args.load_kwargs)
    
    # read in data:
    try:
        data = DataLoader(files, verbose=args.verbose - 1,
                          **load_kwargs)
    except FileNotFoundError:
        print(f'file "{infile}" not found!')
        sys.exit(-1)
    if len(data.file_paths) < len(files):
        print(f'file "{files[len(data.file_paths)]}" does not continue file "{data.file_paths[-1]}"!')
        sys.exit(-1)
    md = data.metadata()
    add_metadata(md, args.md_list, '.')
    if len(args.remove_keys) > 0:
        remove_metadata(md, args.remove_keys, '.')
        cleanup_metadata(md)
    locs, labels = data.markers()
    pre_history = bext_history_str(data.encoding,
                                   data.rate,
                                   data.channels,
                                   os.fsdecode(data.filepath))
    if args.verbose > 1:
        print(f'loaded data file "{data.filepath}"')
        
    if data.encoding is not None and args.encoding is None:
        args.encoding = data.encoding
    for i0 in range(0, len(args.files), nmerge):
        infile = data.file_paths[i0]
        outfile, data_format = make_outfile(args.outpath, infile,
                                            args.data_format,
                                            nmerge < len(args.files),
                                            format_from_extension)
        if not check_format(data_format):
            sys.exit(-1)
        if infile.resolve() == outfile.resolve():
            print(f'! cannot convert "{infile}" to itself !')
            sys.exit(-1)
            
        if len(data.file_paths) > 1:
            i1 = i0 + nmerge - 1
            if i1 >= len(data.end_indices):
                i1 = len(data.end_indices) - 1
            si = data.start_indices[i0]
            ei = data.end_indices[i1]
        else:
            si = 0
            ei = data.frames
        wdata, wrate = modify_data(data[si:ei], data.rate,
                                   md, channels, args.scale,
                                   args.unwrap_clip, args.unwrap,
                                   data.ampl_max, data.unit,
                                   args.decimate)
        mask = (locs[:, 0] >= si) & (locs[:, 0] < ei)
        wlocs = locs[mask, :]
        if len(wlocs) > 0:
            wlocs[:, 0] -= si
        wlabels = labels[mask, :]
        outfile = format_outfile(outfile, md)
        # history:
        hkey = 'CodingHistory'
        if 'BEXT' in md:
            hkey = 'BEXT.' + hkey
        history = bext_history_str(args.encoding, wrate,
                                   data.shape[1], os.fsdecode(outfile))
        add_history(md, history, hkey, pre_history)
        # write out data:
        try:
            write_data(outfile, wdata, wrate, data.ampl_max, data.unit,
                       md, wlocs, wlabels,
                       format=data_format, encoding=args.encoding)
        except PermissionError:
            print(f'failed to write "{outfile}": permission denied!')
            sys.exit(-1)
        # message:
        if args.verbose > 1:
            print(f'wrote "{outfile}"')
        elif args.verbose:
            print(f'converted data file "{infile}" to "{outfile}"')
    data.close()


if __name__ == '__main__':
    main(*sys.argv[1:])
