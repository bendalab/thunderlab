[![license](https://img.shields.io/pypi/l/thunderlab.svg)](https://github.com/bendalab/thunderlab/blob/master/LICENSE)
[![tests](https://github.com/bendalab/thunderlab/workflows/tests/badge.svg?dummy=42)](https://github.com/bendalab/thunderlab/actions)
[![codecov](https://bendalab.github.io/thunderlab/coverage.svg?dummy=42)](https://bendalab.github.io/thunderlab/cover)
[![PyPI version](https://img.shields.io/pypi/v/thunderlab.svg)](https://pypi.python.org/pypi/thunderlab/)
![downloads](https://img.shields.io/pypi/dm/thunderlab.svg)
[![contributors](https://img.shields.io/github/contributors/bendalab/thunderlab)](https://github.com/bendalab/thunderlab/graphs/contributors)
[![commits](https://img.shields.io/github/commit-activity/m/bendalab/thunderlab)](https://github.com/bendalab/thunderlab/pulse)
<!--
![python](https://img.shields.io/pypi/pyversions/thunderlab.svg)
![issues open](https://img.shields.io/github/issues/bendalab/thunderlab.svg)
![issues closed](https://img.shields.io/github/issues-closed/bendalab/thunderlab.svg)
![pullrequests open](https://img.shields.io/github/issues-pr/bendalab/thunderlab.svg)
![pullrequests closed](https://img.shields.io/github/issues-pr-closed/bendalab/thunderlab.svg)
-->

# ThunderLab

Load and preprocess time series data.

[Documentation](https://bendalab.github.io/thunderlab/) |
[API Reference](https://bendalab.github.io/thunderlab/api/)


## Installation

ThunderLab is available from
[PyPi](https://pypi.org/project/thunderlab/). Simply run:
```
pip install thunderlab
```

If you have problems loading specific audio files with ThunderLab,
then you need to install further packages. Follow the [installation
instructions](https://bendalab.github.io/audioio/installation/) of the
[AudioIO](https://bendalab.github.io/audioio/) package.


## Software

The ThunderLab package provides the following software:

- [`convertdata`](https://bendalab.github.io/thunderlab/api/convertdata/): Convert data from various file formats to audio files.



## Algorithms

Click on the modules for more information.

### Input/output

- [`dataloader`](https://bendalab.github.io/thunderlab/api/dataloader.html): Load time-series data from files.
- [`datawriter`](https://bendalab.github.io/thunderlab/api/datawriter.html): Write time-series data to files.
- [`tabledata`](https://bendalab.github.io/thunderlab/api/tabledata.html): Read and write tables with a rich hierarchical header including units and formats.
- [`configfile`](https://bendalab.github.io/thunderlab/api/configfile.html): Configuration file with help texts for analysis parameter.
- [`consoleinput`](https://bendalab.github.io/thunderlab/api/consoleinput.html): User input from console.

### Basic data analysis

- [`eventdetection`](https://bendalab.github.io/thunderlab/api/eventdetection.html): Detect and hande peaks and troughs as well as threshold crossings in data arrays.
- [`powerspectrum`](https://bendalab.github.io/thunderlab/api/powerspectrum.html): Compute and plot powerspectra and spectrograms for a given minimum frequency resolution.
- [`voronoi`](https://bendalab.github.io/thunderlab/api/voronoi.html): Analyse Voronoi diagrams based on scipy.spatial.
- [`multivariateexplorer`](https://bendalab.github.io/thunderlab/api/multivariateexplorer.html): Simple GUI for viewing and exploring multivariate data.


## Used by

- [thunderfish](https://github.com/bendalab/thunderfish): Algorithms and programs for analysing electric field recordings of weakly electric fish.
- [audian](https://github.com/bendalab/audian) Python-based GUI for
viewing and analyzing recordings of animal vocalizations.