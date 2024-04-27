# Multivariate Explorer

Simple GUI for viewing and exploring multivariate data.


## Command line arguments

```sh
multivariateexplorer --help
```
returns
```plain
usage: multivariateexplorer.py [-h] [--version] [-l] [-d COLUMN] [-c COLUMN] [-m CMAP] [file]

View and explore multivariate data.

positional arguments:
  file        a file containing a table of data (csv file or similar)

options:
  -h, --help  show this help message and exit
  --version   show program's version number and exit
  -l          list all available data columns (features) and exit
  -d COLUMN   data columns (features) to be explored
  -c COLUMN   data column to be used for color code or "row"
  -m CMAP     name of color map to be used

version 1.2.0 by Benda-Lab (2019-2024)

mouse:
left click              select sample
left and drag           rectangular selection of samples and/or zoom
shift + left click/drag add samples to selection
ctrl + left click/drag  remove samples from selection
double left click       run thunderfish on selected EOD waveform

key shortcuts:
c, C                    cycle color map trough data columns
p,P                     toggle between features, PCs, and scaled PCs
<, pageup               decrease number of displayed featured/PCs
>, pagedown             increase number of displayed features/PCs
o, z                    toggle zoom mode on or off
backspace               zoom back
n, N                    decrease, increase number of bins of histograms
H                       toggle between scatter plot and 2D histogram
left, right, up, down   show and move magnified scatter plot
escape                  close magnified scatter plot
ctrl + a                select all
+, -                    increase, decrease pick radius
0                       reset pick radius
l                       list selection on console
w                       toggle maximized waveform plot
h                       toggle help window
```