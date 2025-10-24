"""Simple GUI for viewing and exploring multivariate data.

- `class MultiVariateExplorer`: simple matplotlib-based GUI for viewing and exploring multivariate data.
- `categorize()`: convert categorial string data into integer categories.
- `select_features()`: assemble list of column indices.
- `select_coloring()`: select column from data table for colorizing the data.
- `list_available_features()`: print available features on console.
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.widgets as widgets

from scipy.stats import pearsonr
from sklearn import decomposition
from sklearn import preprocessing

from .tabledata import TableData
from .version import __version__, __year__


class MultivariateExplorer(object):
    """Simple matplotlib-based GUI for viewing and exploring multivariate data.

    Shown are scatter plots of all pairs of variables or PCA axis.
    Points in the scatter plots are colored according to the values of one of the variables.
    Data points can be selected and optionally corresponding waveforms are shown.

    First you initialize the explorer with the data. Then you optionally
    specify how to colorize the data and provide waveform data
    associated with the data. Finally you show the figure:
    ```
    expl = MultivariateExplorer(data)
    expl.set_colors(2)
    expl.set_wave_data(waveforms, 'Time [s]', 'Sine')
    expl.show()
    ```

    The `compute_pca() function computes a principal component analysis (PCA)
    on the input data, and `save_pca()` writes the principal components to a file.

    Customize the appearance and information provided by subclassing
    MultivariateExplorer and reimplementing the functions
    - fix_scatter_plot()
    - fix_waveform_plot()
    - list_selection()
    - analyze_selection()
    See the documentation of these functions for details.
    """

    mouse_actions = [
        ('left click', 'select sample'),
        ('left and drag', 'rectangular selection of samples and/or zoom'),
        ('shift + left click/drag', 'add samples to selection'),
        ('ctrl + left click/drag',  'remove samples from selection')
    ]
    """List of tuples with mouse actions and a description of their action."""
        
    key_actions = [
        ('c, C', 'cycle color map trough data columns'),
        ('p,P', 'toggle between features, PCs, and scaled PCs'),
        ('<, pageup', 'decrease number of displayed featured/PCs'),
        ('>, pagedown', 'increase number of displayed features/PCs'),
        ('o, z',  'toggle zoom mode on or off'),
        ('backspace', 'zoom back'),
        ('n, N', 'decrease, increase number of bins of histograms'),
        ('H', 'toggle between scatter plot and 2D histogram'),
        ('left, right, up, down', 'show and move magnified scatter plot'),
        ('escape', 'close magnified scatter plot'),
        ('ctrl + a', 'select all'),
        ('+, -', 'increase, decrease pick radius'),
        ('0', 'reset pick radius'),
        ('l', 'list selection on console'),
        ('w',  'toggle maximized waveform plot'),
        ('h',  'toggle help window'),
    ]
    """List of tuples with key shortcuts and a description of their action."""
    
    def __init__(self, data, labels=None, title=None):
        """Initialize explorer with scatter-plot data.

        Parameters
        ----------
        data: TableData, 2D array, or list of 1D arrays
            The data to be explored. Each column is a variable.
            For the 2D array the columns are the second dimension,
            for a list of 1D arrays, the list goes over columns,
            i.e. each 1D array is one column.
        labels: list of str
            If data is not a TableData, then this provides labels
            for the data columns.
        title: str
            Title for the window.
        """
        # data. categories and labels:
        self.raw_data = None     # original data table as 2D numpy array (samples x features)
        self.raw_labels = None   # for each feature a label, optional with unit
        self.categories = []     # for each feature None or list of categories
        if isinstance(data, TableData):
            for c, col in enumerate(data):
                if not isinstance(data[col][0], (int, float,
                                                 np.integer, np.floating)):
                    # categorial data:
                    cats, data[:,c] = categorize(data[col])
                    self.categories.append(cats)
                else:
                    self.categories.append(None)
            self.raw_data = data.array()
            if labels is None:
                self.raw_labels = []
                for c in range(data.columns()):
                    if len(data.unit(c)) > 0 and not data.unit(c) in ['-', '1']:
                        self.raw_labels.append(f'{data.label(c)} [{data.unit(c)}]')
                    else:
                        self.raw_labels.append(data.label(c))
            else:
                self.raw_labels = labels
        else:
            if isinstance(data, np.ndarray):
                self.raw_data = data
                self.categories = [None] * data.shape[1]
            else:
                for c, col in enumerate(data):
                    if not isinstance(col[0], (int, float,
                                               np.integer, np.floating)):
                        # categorial data:
                        cats, data[c] = categorize(col)
                        self.categories.append(cats)
                    else:
                        self.categories.append(None)
                self.raw_data = np.asarray(data).T
            self.raw_labels = labels
        # remove columns containing only invalid numbers:
        cols = np.all(~np.isfinite(self.raw_data), 0)
        if np.sum(cols) > 0:
            print('removed columns containing no numbers:',
                  [l for l, c in zip(self.raw_labels, cols) if c])
        self.raw_data = self.raw_data[:, ~cols]
        self.raw_labels = [l for l, c in zip(self.raw_labels, cols) if not c]
        # remove rows containing invalid numbers:
        self.valid_samples = ~np.any(~np.isfinite(self.raw_data), 1)
        self.raw_data = self.raw_data[self.valid_samples, :]
        if np.sum(~self.valid_samples) > 0:
            print(f'removed {np.sum(~self.valid_samples)} rows containing invalid numbers:')
            for k in range(len(self.valid_samples)):
                if not self.valid_samples[k]:
                    print(k)
        self.valid_rows = [k for k in range(len(self.valid_samples))
                           if self.valid_samples[k]]
        # title for the window:
        self.title = title if title is not None else 'MultivariateExplorer'
        # data, pca-data, scaled-pca data (no pca data yet):
        self.all_data = [self.raw_data, None, None]
        self.all_labels = [self.raw_labels, None, None]
        self.all_maxcols = [self.raw_data.shape[1], None, None]
        self.all_titles = ['data', 'PCA', 'scaled PCA'] # added to window title 
        # pca:
        self.pca_tables = [None, None]  # raw and scaled pca coefficients
        self._pca_header(self.raw_data, self.raw_labels)  # prepare header of the pca tables
        # start showing raw data:
        self.show_mode = 0                       # show data, pca or scaled pca
        self.data = self.all_data[self.show_mode]       # the data shown
        self.labels = self.all_labels[self.show_mode]   # the feature labels shown
        self.maxcols = self.all_maxcols[self.show_mode] # maximum number of features currently shown
        if self.maxcols > 6:
            self.maxcols = 6
        # waveform data:
        self.wave_data = []
        self.wave_nested = False
        self.wave_has_xticks = []
        self.wave_xlabels = []
        self.wave_ylabels = []
        self.wave_title = False
        # colors:
        self.color_map = plt.get_cmap('jet')
        self.extra_colors = None         # additional data column to be used for coloring
        self.extra_color_label = None    # label for extra_colors
        self.extra_categories = None     # category name for extra_colors if needed
        self.color_set_index = 0         # -1: rows and extra_colors, 0: data, 1: pca, 2: scaled pca
        self.color_index = 0             # column used for coloring with color_set_index
        self.color_values = None         # data column currently used for coloring as specified by color_set_index and color_index
        self.color_label = None          # label of data currently used for coloring
        self.data_colors = None          # actual colors for color_values
        self.color_vmin = None
        self.color_vmax = None
        self.color_ticks = None
        self.cbax = None                 # axes of color bar
        # figure variables:
        self.plt_params = {}
        for k in ['toolbar', 'keymap.quit', 'keymap.back', 'keymap.forward',
                  'keymap.zoom', 'keymap.pan', 'keymap.xscale', 'keymap.yscale']:
            self.plt_params[k] = plt.rcParams[k]
            if k != 'toolbar':
                plt.rcParams[k] = ''
        self.xborder = 100.0       # pixel for ylabels
        self.yborder = 50.0        # pixel for xlabels
        self.spacing = 10.0        # pixel between plots
        self.mborder = 20.0        # pixel around magnified plot
        self.pick_radius = 4.0
        # histogram plots:
        self.hist_ax = []          # list of histogram axes
        self.hist_indices = []     # feature index of the histogram axes
        self.hist_selector = []    # for each histogram axes a selector
        self.hist_nbins = 30       # number of bins for computing histograms
        # scatter plots:
        self.scatter_ax = []       # list of axes with scatter plots (1D)
        self.scatter_indices = []  # for each axes a tuple of the column and row index
        self.scatter_artists = []  # artists of selected scatter points
        self.scatter_selector = [] # selector for each axes
        self.scatter = True        # scatter (True) or density (False)
        self.mark_data = []        # list of indices of selected data
        self.significance_level = 0.05 # r is bold if p is below
        self.select_zooms = False
        self.zoom_stack = []
        # magnified scatter plot:
        self.magnified_on = False
        self.magnified_backdrop = None
        self.magnified_size = np.array([0.6, 0.6])
        # waveform plots:
        self.wave_ax = []
        # help window:
        self.help_ax = None


    def set_wave_data(self, data, xlabels='', ylabels=[], title=False):
        """Add waveform data to explorer.

        Parameters
        ----------
        data: list of (list of) 2D arrays
            Waveform data associated with each row of the data.
            Elements of the outer list correspond to the rows of the data.
            The inner 2D arrays contain a common x-axes (first column)
            and one or more corresponding y-values (second and optional higher columns).
            Each column for y-values is plotted in its own axes on top of each other,
            from top to bottom.
            The optional inner list of 2D arrays contains several 2D arrays as ascribed above
            each with its own common x-axes.
        xlabel: str or list of str
            The xlabels for the waveform plots. If only a string is given, then
            there will be a common xaxis for all the plots, and only the lowest
            one gets a labeled xaxis. If a list of strings is given, each waveform
            plot gets its own labeled x-axis.
        ylabels: list of str
            The ylabels for each of the waveform plots.
        title: bool or str
            If True or a string, povide space on top of the waveform plots for a title.
            If string, set this as the title for the waveform plots.
        """
        self.wave_data = []
        if data is not None and len(data) > 0:
            self.wave_data = data
            self.wave_has_xticks = []
            self.wave_nested = isinstance(data[0], (list, tuple))
            if self.wave_nested:
                for data in self.wave_data[0]:
                    for k in range(data.shape[1]-2):
                        self.wave_has_xticks.append(False)
                    self.wave_has_xticks.append(True)
            else:
                for k in range(self.wave_data[0].shape[1]-2):
                    self.wave_has_xticks.append(False)
                self.wave_has_xticks.append(True)
            if isinstance(xlabels, (list, tuple)):
                self.wave_xlabels = xlabels
            else:
                self.wave_xlabels = [xlabels]
            self.wave_ylabels = ylabels
            self.wave_title = title
        self.wave_ax = []

        
    def set_colors(self, colors=0, color_label=None, color_map=None):
        """Set data column used to color scatter plots.
        
        Parameters
        ----------
        colors: int or 1D array
           Index to colum in data to be used for coloring scatter plots.
           -2 for coloring row index of data.
           Or data array used to color scalar plots.
        color_label: str
           If colors is an array, this is a label describing the data.
           It is used to label the color bar.
        color_map: str or None
            Name of a matplotlib color map.
            If None 'jet' is used.
        """
        if isinstance(colors, (np.integer, int)):
            if colors < 0:
                self.color_set_index = -1
                self.color_index = 0
            else:
                self.color_set_index = 0
                self.color_index = colors
        else:
            if not isinstance(colors[0], (int, float,
                                          np.integer, np.floating)):
                # categorial data:
                self.extra_categories, self.extra_colors = categorize(colors)
            else:
                self.extra_colors = colors
            self.extra_colors = self.extra_colors[self.valid_samples]
            self.extra_color_label = color_label
            self.color_set_index = -1
            self.color_index = 1
        self.color_map = plt.get_cmap(color_map if color_map else 'jet')

        
    def show(self, ioff=True):
        """Show interactive scatter plots for exploration.
        """
        if ioff:
            plt.ioff()
        else:
            plt.ion()
        plt.rcParams['toolbar'] = 'None'
        plt.rcParams['keymap.quit'] = 'ctrl+w, alt+q, ctrl+q, q'
        plt.rcParams['font.size'] = 12
        self.fig = plt.figure(facecolor='white', figsize=(10, 8))
        self.fig.canvas.manager.set_window_title(self.title + ': ' + self.all_titles[self.show_mode])
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('resize_event', self._on_resize)
        self.fig.canvas.mpl_connect('pick_event', self._on_pick)
        if self.color_map is None:
            self.color_map = plt.get_cmap('jet')
        self._set_color_column()
        self._init_hist_plots()
        self._init_scatter_plots()
        self.wave_ax = []
        if self.wave_data is not None and len(self.wave_data) > 0:
            axx = None
            xi = 0
            for k, has_xticks in enumerate(self.wave_has_xticks):
                ax = self.fig.add_subplot(1, len(self.wave_has_xticks),
                                          1+k, sharex=axx)
                self.wave_ax.append(ax)
                if has_xticks:
                    if xi >= len(self.wave_xlabels):
                        self.wave_xlabels.append('')
                    ax.set_xlabel(self.wave_xlabels[xi])
                    xi += 1
                    axx = None
                else:
                    #ax.xaxis.set_major_formatter(plt.NullFormatter())
                    if axx is None:
                        axx = ax
            for ax, ylabel in zip(self.wave_ax, self.wave_ylabels):
                ax.set_ylabel(ylabel)
            if not isinstance(self.wave_title, bool) and self.wave_title:
                self.wave_ax[0].set_title(self.wave_title)
            self.fix_waveform_plot(self.wave_ax, self.mark_data)
        self._plot_magnified_scatter()
        self._plot_help()
        plt.show()


    def _pca_header(self, data, labels):
        """Set up header for the table of principal components.

        Parameters
        ----------
        data: ndarray of float
            The data (samples x features) without invalid (infinite or
            NaN) numbers.
        labels: list of str
            Labels of the features.
        """
        lbs = []
        for l, d in zip(labels, data):
            if '[' in l:
                lbs.append(l.split('[')[0].strip())
            elif '/' in l:
                lbs.append(l.split('/')[0].strip())
            else:
                lbs.append(l)
        header = TableData(header=lbs)
        header.set_formats('%.3f')
        header.insert(0, ['PC'] + ['-']*header.nsecs, '', '%d')
        header.insert(1, 'variance', '%', '%.3f')
        for k in range(len(self.pca_tables)):
            self.pca_tables[k] = TableData(header)

                
    def compute_pca(self, scale=False, write=False):
        """Compute PCA based on the data.

        Parameters
        ----------
        scale: boolean
            If True standardize data before computing PCA, i.e. remove mean
            of each variabel and divide by its standard deviation.
        write: boolean
            If True write PCA components to standard out.
        """
        # pca:
        pca = decomposition.PCA()
        if scale:
            scaler = preprocessing.StandardScaler()
            scaler.fit(self.raw_data)
            pca.fit(scaler.transform(self.raw_data))
            pca_label = 'sPC'
        else:
            pca.fit(self.raw_data)
            pca_label = 'PC'
        for k in range(len(pca.components_)):
            if np.abs(np.min(pca.components_[k])) > np.max(pca.components_[k]):
                pca.components_[k] *= -1.0
        pca_data = pca.transform(self.raw_data)
        pca_labels = [f'{pca_label}{k+1} ' + (f'({100*v:.1f}%)' if v > 0.01 else (f'{100*v:.2f}%'))
                           for k, v in enumerate(pca.explained_variance_ratio_)]
        if np.min(pca.explained_variance_ratio_) >= 0.01:
            pca_maxcols = pca_data.shape[1]
        else:
            pca_maxcols = np.argmax(pca.explained_variance_ratio_ < 0.01)
        if pca_maxcols < 2:
            pca_maxcols = 2
        if pca_maxcols > 6:
            pca_maxcols = 6
        # table with PCA feature weights:
        pca_table = self.pca_tables[1] if scale else self.pca_tables[0]
        pca_table.clear_data()
        pca_table.set_section(0, pca_label, pca_table.nsecs)
        for k, comp in enumerate(pca.components_):
            pca_table.add(k+1, 0)
            pca_table.add(100.0*pca.explained_variance_ratio_[k])
            pca_table.add(comp)
        if write:
            pca_table.write(table_format='out', unit_style='none')
        # submit data:
        if scale:
            self.all_data[2] = pca_data
            self.all_labels[2] = pca_labels
            self.all_maxcols[2] = pca_maxcols
        else:
            self.all_data[1] = pca_data
            self.all_labels[1] = pca_labels
            self.all_maxcols[1] = pca_maxcols

            
    def save_pca(self, file_name, scale, **kwargs):
        """Write PCA data to file.

        Parameters
        ----------
        file_name: str
            Name of ouput file.
        scale: boolean
            If True write PCA components of standardized PCA.
        kwargs: dict
            Additional parameter for TableData.write()
        """
        if scale:
            pca_file = file_name + '-pcacor'
            pca_table = self.pca_tables[1]
        else:
            pca_file = file_name + '-pcacov'
            pca_table = self.pca_tables[0]
        if 'unit_style' in kwargs:
            del kwargs['unit_style']
        if 'table_format' in kwargs:
            pca_table.write(pca_file, unit_style='none', **kwargs)
        else:
            pca_file += '.dat'
            pca_table.write(pca_file, unit_style='none')

            
    def _set_color_column(self):
        """Initialize variables used for colorization of scatter points."""
        if self.color_set_index == -1:
            if self.color_index == 0:
                self.color_values = np.arange(self.data.shape[0], dtype=float)
                self.color_label = 'sample'
            elif self.color_index == 1:
                self.color_values = self.extra_colors
                self.color_label = self.extra_color_label
        else:
            self.color_values = self.all_data[self.color_set_index][:,self.color_index]
            self.color_label = self.all_labels[self.color_set_index][self.color_index]
        self.color_vmin, self.color_vmax, self.color_ticks = \
          self.fix_scatter_plot(self.cbax, self.color_values,
                                self.color_label, 'c')
        if self.color_ticks is None:
            if self.color_set_index == 0 and \
               self.categories[self.color_index] is not None:
                self.color_ticks = np.arange(len(self.categories[self.color_index]))
            elif self.color_set_index == -1 and \
                 self.color_index == 1 and \
                 self.extra_categories is not None:
                self.color_ticks = np.arange(len(self.extra_categories))
        self.data_colors = self.color_map((self.color_values - self.color_vmin)/(self.color_vmax - self.color_vmin))


    def _add_backdrop(self, ax):
        bbox = ax.get_tightbbox(self.fig.canvas.get_renderer())
        if bbox is not None:
            self.magnified_backdrop = \
                patches.Rectangle((bbox.x0 - self.mborder,
                                   bbox.y0 - self.mborder),
                                  bbox.width + 2*self.mborder,
                                  bbox.height + 2*self.mborder,
                                  transform=None, clip_on=False,
                                  facecolor='#f7f7f7', edgecolor='none',
                                  zorder=-5)
            ax.add_patch(self.magnified_backdrop)

            
    def _create_selector(self, ax):
        try:
            selector = \
                widgets.RectangleSelector(ax, self._on_select,
                                          useblit=True, button=1,
                                          minspanx=0, minspany=0,
                                          spancoords='pixels',
                                          props=dict(facecolor='gray',
                                                     edgecolor='gray',
                                                     alpha=0.2,
                                                     fill=True),
                                          state_modifier_keys=dict(move='',
                                                                   clear='',
                                                                   square='',
                                                                   center='ctrl'))
        except TypeError:
            # old matplotlib:
            selector = widgets.RectangleSelector(ax, self._on_select,
                                                 useblit=True, button=1)
        return selector
    
                            
    def _plot_hist(self, ax, magnifiedax):
        """Plot and label a histogram."""
        try:
            idx = self.hist_ax.index(ax)
            c = self.hist_indices[idx]
            in_hist = True
        except ValueError:
            idx = self.scatter_ax.index(ax)
            c = self.scatter_indices[idx][0]
            in_hist = False
        ax.clear()
        #ax.relim()
        #ax.autoscale(True)
        x = self.data[:,c]
        ax.hist(x, self.hist_nbins)
        #ax.autoscale(False)
        ax.set_xlabel(self.labels[c])
        ax.xaxis.set_major_locator(plt.AutoLocator())
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        if self.show_mode == 0:
            if self.categories[c] is not None:
                ax.set_xticks(np.arange(len(self.categories[c])))
                ax.set_xticklabels(self.categories[c])
            self.fix_scatter_plot(ax, self.data[:,c], self.labels[c], 'x')
        if magnifiedax:
            ax.text(0.05, 0.9, f'n={len(self.data)}',
                    transform=ax.transAxes, zorder=100)
            ax.set_ylabel('count')
            cax = self.hist_ax[self.scatter_indices[-1][0]]
            ax.set_xlim(cax.get_xlim())
        else:
            if c == 0:
                ax.text(0.05, 0.9, f'n={len(self.data)}',
                        transform=ax.transAxes, zorder=100)
                ax.set_ylabel('count')
            else:
                ax.yaxis.set_major_formatter(plt.NullFormatter())
        selector = self._create_selector(ax)
        if in_hist:
            self.hist_selector[idx] = selector
        else:
            self.scatter_selector[idx] = selector
            self.scatter_artists[idx] = None
        ax.relim(True)
        if magnifiedax:
            self._add_backdrop(ax)
            

    def _set_hist_ylim(self):
        ymax = np.max([ax.get_ylim() for ax in self.hist_ax[:self.maxcols]], 0)[1]
        for ax in self.hist_ax:
            ax.set_ylim(0, ymax)

                        
    def _init_hist_plots(self):
        """Initial plots of the histograms."""
        n = self.data.shape[1]
        self.hist_ax = []
        for r in range(n):
            ax = self.fig.add_subplot(n, n, (n-1)*n+r+1)
            self.hist_ax.append(ax)
            self.hist_indices.append(r)
            self.hist_selector.append(None)
            self._plot_hist(ax, False)
        self._set_hist_ylim()

                        
    def _plot_scatter(self, ax, magnifiedax, cax=None):
        """Plot a scatter plot."""
        idx = self.scatter_ax.index(ax)
        c, r = self.scatter_indices[idx]
        if self.scatter: # scatter plot
            ax.clear()
            a = ax.scatter(self.data[:,c], self.data[:,r], s=50,
                           edgecolors='white', linewidths=0.5,
                           picker=self.pick_radius, zorder=10)
            a.set_facecolor(self.data_colors)
            pr, pp = pearsonr(self.data[:,c], self.data[:,r])
            fw = 'bold' if pp < self.significance_level/self.data.shape[1] else 'normal'
            if pr < 0:
                ax.text(0.95, 0.9, f'r={pr:.2f}, p={pp:.3f}', fontweight=fw,
                        transform=ax.transAxes, ha='right', zorder=100)
            else:
                ax.text(0.05, 0.9, f'r={pr:.2f}, p={pp:.3f}', fontweight=fw,
                        transform=ax.transAxes, zorder=100)
            # color bar:
            if cax is not None:
                a = ax.scatter(self.data[:, c], self.data[:, r],
                               c=self.color_values, cmap=self.color_map)
                self.fig.colorbar(a, cax=cax, ticks=self.color_ticks)
                a.remove()
                cax.set_ylabel(self.color_label)
                self.color_vmin, self.color_vmax, self.color_ticks = \
                  self.fix_scatter_plot(self.cbax, self.color_values,
                                        self.color_label, 'c')
                if self.color_ticks is None:
                    if self.color_set_index == 0 and \
                       self.categories[self.color_index] is not None:
                        cax.set_yticklabels(self.categories[self.color_index])
                    elif self.color_set_index == -1 and \
                         self.color_index == 1 and \
                         self.extra_categories is not None:
                        cax.set_yticklabels(self.extra_categories)
        else: # histogram
            if self.show_mode == 0:
                self.fix_scatter_plot(ax, self.data[:,c], self.labels[c], 'x')
                self.fix_scatter_plot(ax, self.data[:,r], self.labels[r], 'y')
            axrange = [ax.get_xlim(), ax.get_ylim()]
            ax.clear()
            ax.hist2d(self.data[:,c], self.data[:,r], self.hist_nbins,
                      range=axrange, cmap=plt.get_cmap('Greys'))
        # selected data:
        a = ax.scatter(self.data[self.mark_data, c],
                       self.data[self.mark_data, r], s=100,
                       edgecolors='black', linewidths=0.5,
                       picker=self.pick_radius, zorder=11)
        a.set_facecolor(self.data_colors[self.mark_data])
        self.scatter_artists[idx] = a
        ax.xaxis.set_major_locator(plt.AutoLocator())
        ax.yaxis.set_major_locator(plt.AutoLocator())
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.yaxis.set_major_formatter(plt.ScalarFormatter())
        if self.show_mode == 0:
            if self.categories[c] is not None:
                ax.set_xticks(np.arange(len(self.categories[c])))
                ax.set_xticklabels(self.categories[c])
            if self.categories[r] is not None:
                ax.set_yticks(np.arange(len(self.categories[r])))
                ax.set_yticklabels(self.categories[r])
        if magnifiedax:
            ax.set_xlabel(self.labels[c])
            ax.set_ylabel(self.labels[r])
            cax = self.scatter_ax[self.scatter_indices[:-1].index(self.scatter_indices[-1])]
            ax.set_xlim(cax.get_xlim())
            ax.set_ylim(cax.get_ylim())
        else:
            if c == 0:
                ax.set_ylabel(self.labels[r])
        if self.show_mode == 0:
            self.fix_scatter_plot(ax, self.data[:, c], self.labels[c], 'x')
            self.fix_scatter_plot(ax, self.data[:, r], self.labels[r], 'y')
        if not magnifiedax:
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            if c > 0:
                ax.yaxis.set_major_formatter(plt.NullFormatter())
        ax.set_xlim(*self.hist_ax[c].get_xlim())
        ax.set_ylim(*self.hist_ax[r].get_xlim())
        if magnifiedax:
            self._add_backdrop(ax)
        selector = self._create_selector(ax)
        self.scatter_selector[idx] = selector
        ax.relim(True)

        
    def _init_scatter_plots(self):
        """Initial plots of scatter plots."""
        self.cbax = self.fig.add_axes([0.5, 0.5, 0.1, 0.5])
        cbax = self.cbax
        n = self.data.shape[1]
        for r in range(1, n):
            for c in range(r):
                ax = self.fig.add_subplot(n, n, (r-1)*n+c+1)
                self.scatter_ax.append(ax)
                self.scatter_indices.append([c, r])
                self.scatter_artists.append(None)
                self.scatter_selector.append(None)
                self._plot_scatter(ax, False, cbax)
                cbax = None

                
    def _plot_magnified_scatter(self):
        """Initial plot of the magnified scatter plot."""
        ax = self.fig.add_axes([0.5, 0.9, 0.05, 0.05])
        ax.set_visible(False)
        self.magnified_on = False
        c = 0
        r = 1
        a = ax.scatter(self.data[:, c], self.data[:, r],
                       s=50, edgecolors='none')
        a.set_facecolor(self.data_colors)
        a = ax.scatter(self.data[self.mark_data, c],
                       self.data[self.mark_data, r], s=80)
        a.set_facecolor(self.data_colors[self.mark_data])
        ax.set_xlabel(self.labels[c])
        ax.set_ylabel(self.labels[r])
        self.fix_scatter_plot(ax, self.data[:, c], self.labels[c], 'x')
        self.fix_scatter_plot(ax, self.data[:, r], self.labels[r], 'y')
        self.scatter_ax.append(ax)
        self.scatter_indices.append([c, r])
        self.scatter_artists.append(a)
        self.scatter_selector.append(None)

        
    def _plot_help(self):
        ax = self.fig.add_subplot(1, 1, 1)
        ax.set_position([0.02, 0.02, 0.96, 0.96])
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        n = len(self.mouse_actions) + len(self.key_actions) + 4
        dy = 1/n
        y = 1 - 1.5*dy
        ax.text(0.05, y, 'Key shortcuts', transform=ax.transAxes,
                fontweight='bold')
        y -= dy
        for a, d in self.key_actions:
            ax.text(0.05, y, a, transform=ax.transAxes)
            ax.text(0.3, y, d, transform=ax.transAxes)
            y -= dy
        y -= dy
        ax.text(0.05, y, 'Mouse actions', transform=ax.transAxes,
                fontweight='bold')
        y -= dy
        for a, d in self.mouse_actions:
            ax.text(0.05, y, a, transform=ax.transAxes)
            ax.text(0.3, y, d, transform=ax.transAxes)
            y -= dy
        ax.set_visible(False)
        self.help_ax = ax
        
        
    def fix_scatter_plot(self, ax, data, label, axis):
        """Customize an axes of a scatter plot.

        This function is called after a scatter plot has been plotted.
        Once for the x axes, once for the y axis and once for the color bar.
        Reimplement this function to set appropriate limits and ticks.

        Return values are only used for the color bar (`axis='c'`).
        Otherwise they are ignored.

        For example, ticks for phase variables can be nicely labeled
        using the unicode character for pi:
        ```
        if 'phase' in label:
            if axis == 'y':
                ax.set_ylim(0.0, 2.0*np.pi)
                ax.set_yticks(np.arange(0.0, 2.5*np.pi, 0.5*np.pi))
                ax.set_yticklabels(['0', u'\u03c0/2', u'\u03c0', u'3\u03c0/2', u'2\u03c0'])
        ```
        
        Parameters
        ----------
        ax: matplotlib axes
            Axes of the scatter plot or color bar to be worked on.
        data: 1D array
            Data array of the axes.
        label: str
            Label coresponding to the data array.
        axis: str
            'x', 'y': set properties of x or y axes of ax.
            'c': set properies of color bar axes (note that ax can be None!)
                 and return vmin, vmax, and ticks.

        Returns
        -------
        min: float
            minimum value of color bar axis
        max: float
            maximum value of color bar axis
        ticks: list of float
            position of ticks for color bar axis
        """
        return np.nanmin(data), np.nanmax(data), None

    
    def fix_waveform_plot(self, axs, indices):
        """Customize waveform plots.

        This function is called once after new data have been plotted
        into the waveform plots.  Reimplement this function to customize
        these plots. In particular to set axis limits and labels, plot
        title, etc.
        You may even open a new figure (with non-blocking `show()`).

        The following member variables might be usefull:
        - `self.wave_data`: the full list of waveform data.
        - `self.wave_nested`: True if the elements of `self.wave_data` are lists of 2D arrays. Otherwise the elements are 2D arrays. The first column of a 2D array contains the x-values, further columns y-values.
        - `self.wave_has_xticks`: List of booleans for each axis. True if the axis has its own xticks.
        - `self.wave_xlabels`: List of xlabels (only for the axis where the corresponding entry in `self.wave_has_xticks` is True).
        - `self.wave_ylabels`: for each axis its ylabel
        
        For example, you can set the linewidth of all plotted waveforms via:
        ```
        for ax in axs:
            for l in ax.lines:
                l.set_linewidth(3.0)
        ```
        or enable markers to be plotted:
        ```
        for ax, yl in zip(axs, self.wave_ylabels):
            if 'Power' in yl:
                for l in ax.lines:
                    l.set_marker('.')
                    l.set_markersize(15.0)
                    l.set_markeredgewidth(0.5)
                    l.set_markeredgecolor('k')
                    l.set_markerfacecolor(l.get_color())
        ```
        Usefull is to reduce the maximum number of y-ticks:
        ```
        axs[0].yaxis.get_major_locator().set_params(nbins=7)
        ```
        or
        ```
        import matplotlib.ticker as ticker
        axs[0].yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ```

        Parameters
        ----------
        axs: list of matplotlib axes
            Axis of the waveform plots to be worked on.
        indices: list of int
            Indices of the waveforms that have been selected and plotted.
        """
        pass

    
    def list_selection(self, indices):
        """List information about the current selection of data points.

        This function is called when 'l' is pressed.  Reimplement this
        function, for example, to print some meaningfull information
        about the current selection of data points on console. You may
        do, however, whatever you want in this function.

        Parameters
        ----------
        indices: list of int
            Indices of the data points that have been selected.
        """
        print('')
        print('selected rows in data table:')
        for i in indices:
            print(self.valid_rows[i])

            
    def analyze_selection(self, index):
        """Provide further information about a single selected data point.

        This function is called when a single data item was double
        clicked.  Reimplement this function to provide some further
        details on this data point.  This can be an additional figure
        window. In this case show it non-blocking:
        `plt.show(block=False)`

        Parameters
        ----------
        index: int
            The index of the selected data point.
        """
        pass

    
    def _set_magnified_pos(self, width, height):
        """Set position of magnified plot."""
        if self.magnified_on:
            xoffs = self.xborder/width
            yoffs = self.yborder/height
            if self.scatter_indices[-1][1] < self.data.shape[1]:
                idx = self.scatter_indices[:-1].index(self.scatter_indices[-1])
                pos = self.scatter_ax[idx].get_position().get_points()
            else:
                pos = self.hist_ax[self.scatter_indices[-1][0]].get_position().get_points()
            pos[0] = np.mean(pos, 0) - 0.5*self.magnified_size
            if pos[0][0] < xoffs: pos[0][0] = xoffs
            if pos[0][1] < yoffs: pos[0][1] = yoffs
            pos[1] = pos[0] + self.magnified_size
            if pos[1][0] > 1.0-self.spacing/width: pos[1][0] = 1.0-self.spacing/width
            if pos[1][1] > 1.0-self.spacing/height: pos[1][1] = 1.0-self.spacing/height
            pos[0] = pos[1] - self.magnified_size
            self.scatter_ax[-1].set_position([pos[0][0], pos[0][1],
                                             self.magnified_size[0], self.magnified_size[1]])
            self.scatter_ax[-1].set_visible(True)
        else:
            self.scatter_ax[-1].set_position([0.5, 0.9, 0.05, 0.05])
            self.scatter_ax[-1].set_visible(False)

            
    def _make_selection(self, ax, key, x0, x1, y0, y1):
        """Select points from a scatter or histogram plot."""
        if not key in ['shift', 'control']:
            self.mark_data = []
        if ax in self.scatter_ax:
            axi = self.scatter_ax.index(ax)
            # from scatter plots:
            c, r = self.scatter_indices[axi]
            if r < self.data.shape[1]:
                # from scatter:
                for ind, (x, y) in enumerate(zip(self.data[:, c], self.data[:, r])):
                    if x >= x0 and x <= x1 and y >= y0 and y <= y1:
                        if ind in self.mark_data:
                            if key == 'control':
                                self.mark_data.remove(ind)
                        elif key != 'control':
                            self.mark_data.append(ind)
            else:
                # from histogram:
                for ind, x in enumerate(self.data[:, c]):
                    if x >= x0 and x <= x1:
                        if ind in self.mark_data:
                            if key == 'control':
                                self.mark_data.remove(ind)
                        elif key != 'control':
                            self.mark_data.append(ind)
        elif ax in self.hist_ax:
            r = self.hist_indices[self.hist_ax.index(ax)]
            # from histogram:
            for ind, x in enumerate(self.data[:, r]):
                if x >= x0 and x <= x1:
                    if ind in self.mark_data:
                        if key == 'control':
                            self.mark_data.remove(ind)
                    elif key != 'control':
                        self.mark_data.append(ind)

                        
    def _update_selection(self):
        """Highlight selected points in the scatter plots and plot corresponding waveforms."""
        # update scatter plots:
        for artist, (c, r) in zip(self.scatter_artists, self.scatter_indices):
            if artist is not None:
                if len(self.mark_data) == 0:
                    artist.set_offsets(np.zeros((0, 2)))
                else:
                    artist.set_offsets(list(zip(self.data[self.mark_data, c],
                                                self.data[self.mark_data, r])))
                    artist.set_facecolors(self.data_colors[self.mark_data])
        # waveform plots:
        if len(self.wave_ax) > 0:
            axdi = 0
            axti = 1
            for xi, ax in enumerate(self.wave_ax):
                ax.clear()
                if len(self.mark_data) > 0:
                    for idx in self.mark_data:
                        if self.wave_nested:
                            data = self.wave_data[idx][axdi]
                        else:
                            data = self.wave_data[idx]
                        if data is not None:
                            ax.plot(data[:, 0], data[:, axti],
                                    c=self.data_colors[idx],
                                    picker=self.pick_radius)
                axti += 1
                if self.wave_has_xticks[xi]:
                    ax.set_xlabel(self.wave_xlabels[axdi])
                    axti = 1
                    axdi += 1
                #else:
                #    ax.xaxis.set_major_formatter(plt.NullFormatter())
            for ax, ylabel in zip(self.wave_ax, self.wave_ylabels):
                ax.set_ylabel(ylabel)
            if not isinstance(self.wave_title, bool) and self.wave_title:
                self.wave_ax[0].set_title(self.wave_title)
            self.fix_waveform_plot(self.wave_ax, self.mark_data)
        self.fig.canvas.draw()

        
    def _set_limits(self, ax, x0, x1, y0, y1):
        if ax in self.hist_ax:
            ax.set_xlim(x0, x1)
            for hax in self.hist_ax:
                hax.set_ylim(y0, y1)
            cc = self.hist_indices[self.hist_ax.index(ax)]
            for sax, (c, r) in zip(self.scatter_ax, self.scatter_indices):
                if c == cc:
                    sax.set_xlim(x0, x1)
                if r == cc:
                    sax.set_ylim(x0, x1)
        if ax in self.scatter_ax:
            idx = self.scatter_ax.index(ax)
            cc, rr = self.scatter_indices[idx]
            self.hist_ax[cc].set_xlim(x0, x1)
            self.hist_ax[rr].set_xlim(y0, y1)
            for sax, (c, r) in zip(self.scatter_ax, self.scatter_indices):
                if c == cc:
                    sax.set_xlim(x0, x1)
                if c == rr:
                    sax.set_xlim(y0, y1)
                if r == cc:
                    sax.set_ylim(x0, x1)
                if r == rr:
                    sax.set_ylim(y0, y1)

                    
    def _on_key(self, event):
        """Handle key events."""
        #print('pressed', event.key)
        if event.key in ['left', 'right', 'up', 'down']:
            if self.magnified_on:
                mc, mr = self.scatter_indices[-1]
                if event.key == 'left':
                    if mc > 0:
                        self.scatter_indices[-1][0] -= 1
                    elif mr > 1:
                        if mr >= self.data.shape[1]:
                            self.scatter_indices[-1][1] = self.maxcols - 1
                        else:
                            self.scatter_indices[-1][1] -= 1
                        self.scatter_indices[-1][0] = self.scatter_indices[-1][1] - 1
                    else:
                        self.scatter_indices[-1][0] = self.data.shape[1] - 1
                        self.scatter_indices[-1][1] = self.data.shape[1]
                elif event.key == 'right':
                    if mc < mr - 1 and mc < self.maxcols - 1:
                        self.scatter_indices[-1][0] += 1
                    elif mr < self.maxcols:
                        self.scatter_indices[-1][0] = 0
                        self.scatter_indices[-1][1] += 1
                        if mr >= self.maxcols:
                            self.scatter_indices[-1][1] = self.data.shape[1]
                    else:
                        self.scatter_indices[-1][0] = 0
                        self.scatter_indices[-1][1] = 1
                elif event.key == 'up':
                    if mr > mc + 1:
                        if mr >= self.data.shape[1]:
                            self.scatter_indices[-1][1] = self.maxcols - 1
                        else:
                            self.scatter_indices[-1][1] -= 1
                    elif mc > 0:
                        self.scatter_indices[-1][0] -= 1
                        self.scatter_indices[-1][1] = self.data.shape[1]
                    else:
                        self.scatter_indices[-1][0] = self.data.shape[1] - 1
                        self.scatter_indices[-1][1] = self.data.shape[1]
                elif event.key == 'down':
                    if mr < self.maxcols:
                        self.scatter_indices[-1][1] += 1
                        if mr >= self.maxcols:
                            self.scatter_indices[-1][1] = self.data.shape[1]
                    elif mc < self.maxcols - 1:
                        self.scatter_indices[-1][0] += 1
                        self.scatter_indices[-1][1] = mc + 2
                        if self.scatter_indices[-1][1] >= self.maxcols:
                            self.scatter_indices[-1][1] = self.data.shape[1]
                    else:
                        self.scatter_indices[-1][0] = 0
                        self.scatter_indices[-1][1] = 1
            for k in reversed(range(len(self.zoom_stack))):
                if self.zoom_stack[k][0] == self.scatter_ax[-1]:
                    del self.zoom_stack[k]
            self.scatter_ax[-1].clear()
            self.scatter_ax[-1].set_visible(True)
            self.magnified_on = True
            self._set_magnified_pos(self.fig.get_window_extent().width,
                                    self.fig.get_window_extent().height)
            if self.scatter_indices[-1][1] < self.data.shape[1]:
                self._plot_scatter(self.scatter_ax[-1], True)
            else:
                self._plot_hist(self.scatter_ax[-1], True)
            self.fig.canvas.draw()
        else:
            if event.key == 'escape':
                if len(self.scatter_ax) >= self.data.shape[1]:
                    self.scatter_ax[-1].set_position([0.5, 0.9, 0.05, 0.05])
                    self.magnified_on = False
                    self.scatter_ax[-1].set_visible(False)
                    self.fig.canvas.draw()
            elif event.key in 'oz':
                self.select_zooms = not self.select_zooms
            elif event.key == 'backspace':
                if len(self.zoom_stack) > 0:
                    self._set_limits(*self.zoom_stack.pop())
                    self.fig.canvas.draw()
            elif event.key in '+=':
                self.pick_radius *= 1.5
            elif event.key in '-':
                if self.pick_radius > 5.0:
                    self.pick_radius /= 1.5
            elif event.key in '0':
                self.pick_radius = 4.0
            elif event.key in ['pageup', 'pagedown', '<', '>']:
                if event.key in ['pageup', '<'] and self.maxcols > 2:
                    self.maxcols -= 1
                elif event.key in ['pagedown', '>'] and self.maxcols < self.raw_data.shape[1]:
                    self.maxcols += 1
                for ax in self.hist_ax:
                    self._plot_hist(ax, False)
                self._update_layout()
            elif event.key == 'w':
                if len(self.wave_data) > 0:
                    if self.maxcols > 0:
                        self.all_maxcols[self.show_mode] = self.maxcols
                        self.maxcols = 0
                    else:
                        self.maxcols = self.all_maxcols[self.show_mode]
                    self._set_layout(self.fig.get_window_extent().width,
                                     self.fig.get_window_extent().height)
                    self.fig.canvas.draw()
            elif event.key == 'ctrl+a':
                self.mark_data = list(range(len(self.data)))
                self._update_selection()
            elif event.key in 'cC':
                if event.key in 'c':
                    self.color_index -= 1
                    if self.color_index < 0:
                        self.color_set_index -= 1
                        if self.color_set_index < -1:
                            self.color_set_index = len(self.all_data)-1
                        if self.color_set_index >= 0:
                            if self.all_data[self.color_set_index] is None:
                                self.compute_pca(self.color_set_index>1, True)
                            self.color_index = self.all_data[self.color_set_index].shape[1]-1
                        else:
                            self.color_index = 0 if self.extra_colors is None else 1
                    self._set_color_column()
                else:
                    self.color_index += 1
                    if (self.color_set_index >= 0 and \
                        self.color_index >= self.all_data[self.color_set_index].shape[1]) or \
                        (self.color_set_index < 0 and \
                         self.color_index >= (1 if self.extra_colors is None else 2)):
                        self.color_index = 0
                        self.color_set_index += 1
                        if self.color_set_index >= len(self.all_data):
                            self.color_set_index = -1
                        elif self.all_data[self.color_set_index] is None:
                            self.compute_pca(self.color_set_index>1, True)
                    self._set_color_column()
                for ax in self.scatter_ax:
                    ax.collections[0].set_facecolors(self.data_colors)
                for a in self.scatter_artists:
                    if a is not None:
                        a.set_facecolors(self.data_colors[self.mark_data])
                for ax in self.wave_ax:
                    for l, c in zip(ax.lines, self.data_colors[self.mark_data]):
                        l.set_color(c)
                        l.set_markerfacecolor(c)
                self._plot_scatter(self.scatter_ax[0], False, self.cbax)
                self.fix_scatter_plot(self.cbax, self.color_values,
                                      self.color_label, 'c')
                self.fig.canvas.draw()
            elif event.key in 'nN':
                if event.key in 'N':
                    self.hist_nbins = (self.hist_nbins*3)//2
                elif self.hist_nbins >= 15:
                    self.hist_nbins = (self.hist_nbins*2)//3
                else:
                    self.hist_nbins = 10
                for ax in self.hist_ax:
                    self._plot_hist(ax, False)
                self._set_hist_ylim()
                if self.scatter_indices[-1][1] >= self.data.shape[1]:
                    self._plot_hist(self.scatter_ax[-1], True, True)
                elif not self.scatter:
                    self._plot_scatter(self.scatter_ax[-1], True)
                if not self.scatter:
                    for ax in self.scatter_ax[:-1]:
                        self._plot_scatter(ax, False)
                self.fig.canvas.draw()
            elif event.key in 'H':
                self.scatter = not self.scatter
                for ax in self.scatter_ax[:-1]:
                    self._plot_scatter(ax, False)
                if self.scatter_indices[-1][1] < self.data.shape[1]:
                    self._plot_scatter(self.scatter_ax[-1], True)
                self.fig.canvas.draw()
            elif event.key in 'pP':
                if len(self.scatter_ax) >= self.data.shape[1]:
                    self.scatter_ax[-1].set_position([0.5, 0.9, 0.05, 0.05])
                    self.scatter_indices[-1] = [0, 1]
                    self.magnified_on = False
                    self.scatter_ax[-1].set_visible(False)
                self.all_maxcols[self.show_mode] = self.maxcols
                if event.key == 'P':
                    self.show_mode += 1
                    if self.show_mode >= len(self.all_data):
                        self.show_mode = 0
                else:
                    self.show_mode -= 1
                    if self.show_mode < 0:
                        self.show_mode = len(self.all_data)-1
                if self.show_mode == 1:
                    print('principal components')
                elif self.show_mode == 2:
                    print('scaled principal components')
                else:
                    print('data')
                if self.all_data[self.show_mode] is None:
                    self.compute_pca(self.show_mode>1, True)
                self.data = self.all_data[self.show_mode]
                self.labels = self.all_labels[self.show_mode]
                self.maxcols = self.all_maxcols[self.show_mode]
                self.zoom_stack = []
                self.fig.canvas.manager.set_window_title(self.title + ': ' + self.all_titles[self.show_mode])
                for ax in self.hist_ax:
                    self._plot_hist(ax, False)
                self._set_hist_ylim()
                for ax in self.scatter_ax:
                    self._plot_scatter(ax, False)
                self._update_layout()
            elif event.key in 'l':
                if len(self.mark_data) > 0:
                    self.list_selection(self.mark_data)
            elif event.key in 'h':
                self.help_ax.set_visible(not self.help_ax.get_visible())
                self.fig.canvas.draw()

            
    def _on_select(self, eclick, erelease):
        """Handle selection events."""
        if eclick.dblclick:
            if len(self.mark_data) > 0:
                self.analyze_selection(self.mark_data[-1])
            return
        x0 = min(eclick.xdata, erelease.xdata)
        x1 = max(eclick.xdata, erelease.xdata)
        y0 = min(eclick.ydata, erelease.ydata)
        y1 = max(eclick.ydata, erelease.ydata)
        ax = erelease.inaxes
        if ax is None:
            ax = eclick.inaxes
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        dx = 0.02*(xmax-xmin)
        dy = 0.02*(ymax-ymin)
        if x1 - x0 < dx and y1 - y0 < dy:
            bbox = ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            width, height = bbox.width, bbox.height
            width *= self.fig.dpi
            height *= self.fig.dpi
            dx = self.pick_radius*(xmax-xmin)/width
            dy = self.pick_radius*(ymax-ymin)/height
            x0 = erelease.xdata - dx
            x1 = erelease.xdata + dx
            y0 = erelease.ydata - dy
            y1 = erelease.ydata + dy
        elif self.select_zooms:
            self.zoom_stack.append((ax, xmin, xmax, ymin, ymax))
            self._set_limits(ax, x0, x1, y0, y1)
        self._make_selection(ax, erelease.key, x0, x1, y0, y1)
        self._update_selection()

        
    def _on_pick(self, event):
        """Handle pick events."""
        for ax in self.wave_ax:
            for k, l in enumerate(ax.lines):
                if l is event.artist:
                    self.mark_data = [self.mark_data[k]]
        for ax in self.scatter_ax:
            if ax.collections[0] is event.artist:
                self.mark_data = event.ind
        self._update_selection()
        if event.mouseevent.dblclick:
            if len(self.mark_data) > 0:
                self.analyze_selection(self.mark_data[-1])

                    
    def _set_layout(self, width, height):
        """Update positions and visibility of all plots."""
        xoffs = self.xborder/width
        yoffs = self.yborder/height
        xs = self.spacing/width
        ys = self.spacing/height
        if self.maxcols > 0:
            dx = (1.0-xoffs)/self.maxcols
            dy = (1.0-yoffs)/self.maxcols
            xw = dx - xs
            yw = dy - ys
        # histograms:
        for c, ax in enumerate(self.hist_ax):
            if c < self.maxcols:
                ax.set_position([xoffs+c*dx, yoffs, xw, yw])
                ax.set_visible(True)
            else:
                ax.set_visible(False)
                ax.set_position([0.99, 0.01, 0.01, 0.01])
        # scatter plots:
        for ax, (c, r) in zip(self.scatter_ax[:-1], self.scatter_indices[:-1]):
            if r < self.maxcols:
                ax.set_position([xoffs+c*dx, yoffs+(self.maxcols-r)*dy, xw, yw])
                ax.set_visible(True)
            else:
                ax.set_visible(False)
                ax.set_position([0.99, 0.01, 0.01, 0.01])
        # color bar:
        if self.maxcols > 0:
            self.cbax.set_position([xoffs+dx, yoffs+(self.maxcols-1)*dy, 0.3*xoffs, yw])
            self.cbax.set_visible(True)
        else:
            self.cbax.set_visible(False)
            self.cbax.set_position([0.99, 0.01, 0.01, 0.01])
        # magnified plot:
        if self.maxcols > 0:
            self._set_magnified_pos(width, height)
            if self.magnified_backdrop is not None:
                bbox = self.scatter_ax[-1].get_tightbbox(self.fig.canvas.get_renderer())
                if bbox is not None:
                    self.magnified_backdrop.set_bounds(bbox.x0 - self.mborder,
                                                       bbox.y0 - self.mborder,
                                                       bbox.width + 2*self.mborder,
                                                       bbox.height + 2*self.mborder)
        else:
            self.scatter_ax[-1].set_position([0.5, 0.9, 0.05, 0.05])
            self.scatter_ax[-1].set_visible(False)
        # waveform plots:
        if len(self.wave_ax) > 0:
            if self.maxcols > 0:
                x0 = xoffs+((self.maxcols+1)//2)*dx
                y0 = ((self.maxcols+1)//2)*dy
                if self.maxcols%2 == 0:
                    x0 += xoffs
                    y0 += yoffs - ys
                else:
                    y0 += ys
            else:
                x0 = xoffs
                y0 = 0.0
            yp = 1.0
            dy = 1.0-y0
            dy -= np.sum(self.wave_has_xticks)*yoffs
            yp -= ys
            dy -= ys
            if self.wave_title:
                yp -= 2*ys
                dy -= 2*ys
            dy /= len(self.wave_ax)
            for ax, has_xticks in zip(self.wave_ax, self.wave_has_xticks):
                yp -= dy
                ax.set_position([x0, yp, 1.0-x0-xs, dy])
                if has_xticks:
                    yp -= yoffs
                else:
                    yp -= ys

            
    def _update_layout(self):
        """Update content and position of magnified plot."""
        if self.scatter_indices[-1][1] < self.data.shape[1]:
            if self.scatter_indices[-1][1] >= self.maxcols:
                self.scatter_indices[-1][1] = self.maxcols-1
            if self.scatter_indices[-1][0] >= self.scatter_indices[-1][1]:
                self.scatter_indices[-1][0] = self.scatter_indices[-1][1]-1
            self._plot_scatter(self.scatter_ax[-1], True)
        else:
            if self.scatter_indices[-1][0] >= self.maxcols:
                self.scatter_indices[-1][0] = self.maxcols-1
                self._plot_hist(self.scatter_ax[-1], True)
        self._set_hist_ylim()
        self._set_layout(self.fig.get_window_extent().width,
                         self.fig.get_window_extent().height)
        self.fig.canvas.draw()

        
    def _on_resize(self, event):
        """Adapt layout of plots to new figure size."""
        self._set_layout(event.width, event.height)


def categorize(data):
    """Convert categorial string data into integer categories.

    Parameters
    ----------
    data: list or ndarray of str
        Data with textual categories.

    Returns
    -------
    categories: list of str
        A sorted unique list of the strings in `data`,
        that is the names of the categories.
    cdata: ndarray of int
        A copy of the input `data` where each string value is replaced
        by an integer number that is an index into the returned `categories`.
    """
    cats = sorted(set(data))
    cdata = np.array([cats.index(x) for x in data], dtype=int)
    return cats, cdata


def select_features(data, columns):
    """Assemble list of column indices.

    Parameters
    ----------
    data: TableData
        The table from which to select features.
    columns: list of str
        Feature names (column headers) to be selected from the data.
        If a column is listed twice (even times) it is not added.

    Returns
    -------
    data_cols: list of int
        List of indices into data columns for selecting features.
    """
    if len(columns) == 0:
        data_cols = list(np.arange(len(data)))
    else:
        data_cols = []
        for c in columns:
            idx = data.index(c)
            if idx is None:
                print(f'"{c}" is not a valid data column')
            elif idx in data_cols:
                data_cols.remove(idx)
            else:
                data_cols.append(idx)
    return data_cols


def select_coloring(data, data_cols, color_col):
    """Select column from data table for colorizing the data.

    Pass the output of this function on to MultivariateExplorer.set_colors().

    Parameters
    ----------
    data: TableData
        Table with all EOD properties from which columns are selected.
    data_cols: list of int
        List of columns selected to be explored.
    color_col: str or int
        Column to be selected for coloring the data.
        If 'row' then use the row index of the data in the table for coloring.

    Returns
    -------
    colors: int or list of values or None
        Either index of `data_cols` or additional data from the data table
        to be used for coloring.
    color_label: str or None
        Label for labeling the color bar.
    color_idx: int or None
        Index of color column in `data`.
    error: None or str
        In case an invalid column is selected, an error string.
    """
    color_idx = data.index(color_col)
    colors = None
    color_label = None
    if color_idx is None and color_col != 'row':
        return None, None, None, f'"{color_col}" is not a valid column for color code'
    if color_idx is None:
        colors = -2
    elif color_idx in data_cols:
        colors = data_cols.index(color_idx)
    else:
        if len(data.unit(color_idx)) > 0 and not data.unit(color_idx) in ['-', '1']:
            color_label = f'{data.label(color_idx)} [{data.unit(color_idx)}]'
        else:
            color_label = data.label(color_idx)
        colors = data[:, color_idx]
    return colors, color_label, color_idx, None


def list_available_features(data, data_cols=[], color_col=None):
    """Print available features on console.

    Parameters
    ----------
    data: TableData
        The full data table.
    data_cols: list of int
        List of indices of columns (features) in the table
        that are passed on to the MultivariateExplorer.
    color_col: int or None
        Index of columns (feature) in the table
        that is initially used for color coding the data.
    """
    print('available features:')
    for k, c in enumerate(data.keys()):
        s = [' '] * 3
        if k in data_cols:
            s[1] = '*'
        if color_col is not None and k == color_col:
            s[0] = 'C'
        print(''.join(s) + c)
    if len(data_cols) > 0:
        print('*: feature selected for exploration')
    if color_col is not None:
        print('C: feature selected for color coding the data')


class PrintHelp(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):
        parser.print_help()
        print('')
        print('mouse:')
        for ma in MultivariateExplorer.mouse_actions:
            print('%-23s %s' % ma)
        print('%-23s %s' % ('double left click', 'run thunderfish on selected EOD waveform'))
        print('')
        print('key shortcuts:')
        for ka in MultivariateExplorer.key_actions:
            print('%-23s %s' % ka)
        parser.exit()      


def demo():
    """Run the multivariate explorer with a random test data set.
    """
    # generate data:
    n = 100
    data = []
    data.append(np.random.randn(n) + 2.0)
    data.append(1.0+0.1*data[0] + 1.5*np.random.randn(n))
    data.append(10*(-3.0*data[0] + 2.0*data[1] + 1.8*np.random.randn(n)))
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
        #waveforms.append([np.column_stack((time, x)), np.column_stack((time, y))])
    # initialize explorer:
    expl = MultivariateExplorer(data,
                                list(map(chr, np.arange(len(data))+ord('A'))),
                                'Explorer')
    expl.set_wave_data(waveforms, 'Time', ['Sine', 'Gauss'])
    # explore data:
    expl.set_colors()
    expl.show()
        

def main(*cargs):
    # parse command line:
    parser = argparse.ArgumentParser(add_help=False,
        description='View and explore multivariate data.',
        epilog = f'version {__version__} by Benda-Lab (2019-{__year__})')
    parser.add_argument('-h', '--help', nargs=0, action=PrintHelp,
                        help='show this help message and exit')
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('-l', dest='list_features', action='store_true',
                        help='list all available data columns (features) and exit')
    parser.add_argument('-d', dest='data_cols', action='append',
                        default=[], metavar='COLUMN',
                        help='data columns (features) to be explored')
    parser.add_argument('-c', dest='color_col', default=None,
                        type=str, metavar='COLUMN',
                        help='data column to be used for color code or "row"')
    parser.add_argument('-m', dest='color_map', default='jet',
                        type=str, metavar='CMAP',
                        help='name of color map to be used')
    parser.add_argument('file', nargs='?', default='', type=str,
                        help='a file containing a table of data (csv file or similar)')
    if len(cargs) == 0:
        cargs = None
    args = parser.parse_args(cargs)
    if args.file:
        # load data:
        data = TableData(args.file)
        data_cols = select_features(data, args.data_cols)
        # select column used for coloring the data:
        colors, color_label, color_col, error = \
          select_coloring(data, data_cols, args.color_col)
        if error:
            parser.error(error)
        # list features:
        if args.list_features:
            list_available_features(data, data_cols, color_col)
            parser.exit()
        # explore data:
        expl = MultivariateExplorer(data[:, data_cols])
        expl.set_colors(colors, color_label, args.color_map)
        expl.show()
    else:
        demo()


if __name__ == '__main__':
    main(*sys.argv[1:])
