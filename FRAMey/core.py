"""
Set of routines to work like FRAMe R-code by Jared Ragland
"""

from __future__ import print_function, division, absolute_import
from builtins import *

from collections import OrderedDict
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from .cached_decorators import cached, cached_clear



__author__ = 'William Krekelberg'
__email__ = 'wpk@nist.gov'

def read_input(path, header=[0, 1], index_col=0, col_to_numeric=True, **kwargs):
    """
    create DataFrame from input file

    Parameters
    ----------
    path : str
        file path
    header : default=[0, 1]
        header lines.  See pd.read_csv
    index_col : default=0
        index column, See pd.read_csv
    col_to_numerc : bool, default=True
        If True, convert column names to numeric values if appropriate.
        See pd.to_numeric

    kwargs : extra arguments to pd.read_csv

    Returns
    -------
    df : pd.DataFrame
    """

    df = pd.read_csv(path, header=header, index_col=index_col, **kwargs)

    if col_to_numeric:
        df.rename(columns=lambda x: pd.to_numeric(x, errors='ignore'), inplace=True)
    return df


def load_test_data(out_file=None, read_kws=None, write_kws=None):
    """
    Load example dataset

    Parameters
    ----------
    out_file : str, optional
        if supplied, save example to file out_file

    read_kws : extra arguments to read_input

    write_kws : extra arguments to df.to_csv

    Returns
    -------
    df : pd.DataFrame
    """
    url = 'https://raw.githubusercontent.com/wpk-nist-gov/FRAMey/develop/example_data/test_input.csv'

    read_kws = read_kws or {}
    df = read_input(url, **read_kws)

    if out_file is not None:
        ext = os.path.splitext(out_file)[1]
        assert ext == '.csv'
        write_kws = write_kws or {}
        df.to_csv(out_file, **write_kws)

    return df


def _drop_leading_QC(df, leading_samples=3):
    samples = (
        df.query('Class=="QC"')['Sample']
        .drop_duplicates()
        .sort_values()
        .values[:leading_samples]
    )
    return df.query('Class != "QC" or Sample not in @samples')

def flatten_columns(df, sep='_', copy=False):
    """flatten heirarchical columns"""
    if copy:
        df = df.copy()
    df.columns = df.columns.map(sep.join)
    return df


_attr_map = {
    '<' : '__lt__',
    '<=' : '__le__',
    '>': '__gt__',
    '>=': '__ge__',
    '==': '__eq__'}

class _Threshold(object):
    def __init__(self, base_series, expr='<', val=1.0,
                 plotter='kde',
                 dist_kws = None,
                 dist2_kws = None,
                 vline_kws = None,
                 xlabel=None,
                 colors = ['tab:blue','tab:orange']):
        """ perform operations of form df[var_name] .expr. val"""
        self._cache = {}
        self.s = base_series
        self.expr = expr
        self.bound = val

        # plotter params
        assert plotter in ['kde','hist']
        self._plotter = plotter
        assert len(colors) >= 2
        self._colors = colors
        self._dist_kws = dist_kws or {}
        self._dist2_kws = dist2_kws or {}
        self._vline_kws = vline_kws or {}

        self._xlabel = xlabel or self.s.name

    @property
    def s(self):
        return self._s

    @s.setter
    @cached_clear()
    def s(self, val):
        if val is not None:
            assert isinstance(val, pd.Series)
        self._s = val

    @property
    def expr(self):
        return self._expr

    @expr.setter
    @cached_clear()
    def expr(self, val):
        assert val in _attr_map.keys()
        self._expr = val
        self._attr = _attr_map[self._expr]

    @property
    def bound(self):
        return self._bound

    @bound.setter
    @cached_clear()
    def bound(self, val):
        self._bound = val


    def apply_mask(self, val):
        """ apply val .expr. self.bound"""
        expr = getattr(val, self._attr)
        return expr(self.bound)

    @property
    @cached()
    def mask(self):
        """mask where self.s .expr. self.bound"""
        return self.apply_mask(self.s)

    @property
    @cached()
    def remove(self):
        """self.s[mask]"""
        return self.s[self.mask]


    def _make_kde_base(self, data, ax, **kwargs):
        # turn on default stuff
        kwargs = dict(dict(shade=True, bw='silverman', color=self._colors[0], alpha=1.0), **kwargs)
        sns.kdeplot(data, ax=ax, **kwargs)
        x, y = ax.lines[0].get_data()
        return x, y

    def _plot_fill_between(self, x, y, ax, **kwargs):
        kwargs = dict(dict(color=self._colors[1], alpha=1.0), **kwargs)
        ax.plot(x, y, **kwargs)
        ax.fill_between(x, 0, y, **kwargs)

    def _make_kde(self, data, f_filter, ax=None, bound=None,
                  dist_kws=None, dist2_kws=None, vline_kws=None):
        if ax is None:
            ax = plt.gca()

        dist_kws = dist_kws or {}
        x, y = self._make_kde_base(data, ax=ax, **dist_kws)

        if f_filter is not None:
            msk = f_filter(x)
            dist2_kws = dist2_kws or {}
            self._plot_fill_between(x[msk], y[msk], ax=ax, **dist2_kws)
        else:
            msk = None

        if bound is not None:
            vline_kws = vline_kws or {}
            vline_kws.setdefault('color','k')
            ax.axvline(x=bound, **vline_kws)

        return ax, x, y, msk

    def _make_hist_base(self, data, ax, **kwargs):
        # turn on default stuff
        kwargs = dict(dict(bins=50, color=self._colors[0], normed=True), **kwargs)
        y, x, p = ax.hist(data,**kwargs)
        x = 0.5 * (x[1:] + x[:-1])
        return x, y

    def _plot_bar(self, x, y, ax, **kwargs):
        kwargs = dict(dict(color=self._colors[1], **kwargs))
        ax.bar(x, y, width = x[1] - x[0], **kwargs)

    def _make_hist(self, data, f_filter, ax=None, bound=None,
                   dist_kws=None, dist2_kws=None, vline_kws=None):
        if ax is None:
            ax = plt.gca()
        dist_kws = dist_kws or {}
        x, y = self._make_hist_base(data=data, ax=ax, **dist_kws)

        if f_filter is not None:
            msk = f_filter(x)
            dist2_kws = dist2_kws or {}
            self._plot_bar(x[msk], y[msk], ax=ax, **dist2_kws)
        else:
            msk = None

        if bound is not None:
            vline_kws = vline_kws or {}
            vline_kws.setdefault('color','k')
            ax.axvline(x=bound, **vline_kws)

        return ax, x, y, msk


    def plot(self, ax=None, dist_kws=None, dist2_kws=None, vline_kws=None,
             xlabel=None):

        data = self.s.dropna().values
        f_filter = self.apply_mask
        bound = self.bound

        dist_kws = dist_kws or {}
        dist2_kws = dist2_kws or {}
        vline_kws = vline_kws or {}

        dist_kws = dict(self._dist_kws, **dist_kws)
        dist2_kws = dict(self._dist2_kws, **dist2_kws)
        vline_kws = dict(self._vline_kws, **vline_kws)

        if self._plotter == 'kde':
            _plot = self._make_kde
        else:
            _plot = self._make_hist

        ax, x, y, msk = _plot(data=data, f_filter=f_filter, ax=ax, bound=bound,
                              dist_kws=dist_kws, dist2_kws=dist2_kws, vline_kws=vline_kws)

        if xlabel is None:
            xlabel = self._xlabel
        ax.set_xlabel(xlabel)


    def to_str(self, left, expr=None, val_post=None, right=None):

        expr = expr or self.expr
        val_post = val_post or ''
        right = right or ''

        s='{removed} features removed due to {left} **{expr}{val}{val_post}** {right}'.format(
            removed=len(self.remove),
            left=left,
            expr=expr,
            val=self.bound,
            val_post=val_post,
            right=right)
        return s




class _FRAMe(object):
    """
    Base class for frame
    """
    def __init__(self, table,
                 info_columns=None,
                 leading_samples=3,
                 combine_classes_on_text='Class',
                 colors=None, cmap=None):

        self._cache = {}

        self.table = table

        self._info_columns = info_columns or []
        self._leading_samples = leading_samples

        self._combine_classes_on_text = combine_classes_on_text
        self._set_colors(colors, cmap)


    @property
    def table(self):
        """place holder for input data"""
        return self._table

    @table.setter
    @cached_clear()
    def table(self, val):
        """set input data"""
        # ensure column 'Sample' has numeric values
        self._table = val.rename(columns=lambda x: pd.to_numeric(x, errors='ignore'))
        #self._table = val


    @property
    @cached()
    def flat(self):
        """Flat data less leading QC samples"""

        # function to add group column
        group_func = lambda x: (x['Class']
                                .mask(x['Class'].str.contains(self._combine_classes_on_text), 'Sample'))
        return (
            self.table
            # drop info columns
            .drop(self._info_columns, axis=1, level=0)
            # add index name
            .rename_axis('Feature', axis=0)
            # reshape
            .reset_index()
            .pipe(pd.melt, id_vars='Feature', value_name='Area')
            # ensure sample is numeric
            .assign(Sample = lambda x: pd.to_numeric(x['Sample'], errors='ignore'))
            # drop qc samples
            .pipe(_drop_leading_QC, self._leading_samples)
            # add group column
            .assign(Group = group_func )
        )

    @property
    @cached()
    def Samples(self):
        """Samples without leading QC"""
        return self.flat['Sample'].unique()


    @property
    @cached()
    def stats_class(self):
        """frame with stats for Class,Feature grouped stats"""
        return (
            self.flat
            .dropna(subset=['Area'])
            .groupby(['Class','Feature'])['Area']
            .agg(['mean','median','std'])
            .eval('rsd = std / mean * 100', inplace=False)
        )

    @property
    @cached()
    def stats_group(self):
        """frame with stats for Group,Feature grouped stats""" 
        return (
            self.flat
            .dropna(subset=['Area'])
            .groupby(['Group','Feature'])['Area']
            .agg(['mean','median', 'std'])
            .eval('rsd = std / mean * 100', inplace=False)
        )


    @classmethod
    def from_input_Rfile(cls, filename,
                         info_columns=None, leading_samples=3,
                         read_kws=None,
                         **kwargs):

        read_kws = read_kws or {}
        df = read_input(filename, **read_kws)

        # read_kws = dict(dict(header=[0, 1], index_col=0))
        # df = pd.read_csv(filename, **read_kws)



        return cls(df,
                   info_columns=info_columns,
                   leading_samples=leading_samples,
                   **kwargs)


    def _set_colors(self, colors=None, cmap=None):
        # cmap
        if cmap is None and colors is None:
            colors = ['tab:blue','tab:orange']

        if colors is not None:
            cmap = LinearSegmentedColormap.from_list('mymap', colors, N=len(colors))

        if cmap is None:
            raise ValueError('must specify colors or cmap')

        cmap = cmap._resample(2)
        self._cmap = cmap


class FRAMe(_FRAMe):

    _default_filters = OrderedDict([
        ('blank_contribution', 5.0),
        ('median_loq_low', 10.0),
        ('median_loq_high', 50.0),
        ('qc_count', 60.0), # renamed from qc_count_freq
        ('qc_rsd', 20.0), 
        ('sample_count', 80.0), # renamed from sample_count_freq
        ('low_variability',120.0 )
    ])


    def __init__(self, table,
                 info_columns=None,
                 leading_samples=3,
                 combine_classes_on_text='Class',
                 colors=None, cmap=None,
                 blank_contribution=5.0,
                 median_loq_low=10.0,
                 median_loq_high=50.0,
                 qc_count=60.0,
                 qc_rsd=20.0,
                 sample_count=80.0,
                 low_variability=120.0):
        """
        Parameters
        ----------
        data : pandas.DataFrame
            input data
        info_columns : list, optional
            default = ['Info]
        leading_samples : int
            number of leading QC samples
        combine_classes_on_text : str, default='Class'
            string to convert to Groups
        colors : list of colors, default=['tab:blue','tab:orange']
            colors for kept and removed samples
        cmap : colormap, optional
            if passed and colors=None, then use this map for colors

        blank_contribution, median_loq_low, median_loq_high,
        qc_count, qc_rsd, sample_count, low_variability : floats
            Thresholds

        """

        super(FRAMe, self).__init__(table=table,
                                    info_columns=info_columns,
                                    leading_samples=leading_samples,
                                    combine_classes_on_text=combine_classes_on_text,
                                    colors=colors, cmap=cmap)


        # setup thresholds
        self._thresholds = self._default_filters.copy()
        self.update_thresholds(
            blank_contribution=blank_contribution,
            median_loq_low=median_loq_low,
            median_loq_high=median_loq_high,
            qc_count=qc_count,
            qc_rsd=qc_rsd,
            sample_count=sample_count,
            low_variability=low_variability)

    @cached_clear()
    def update_thresholds(self,**kwargs):
        for k in kwargs:
            if k not in self._default_filters:
                raise ValueError('{} not a valid filter name'.format(k))
            self._thresholds[k] = kwargs[k]

    @property
    def filter_names(self):
        return self._default_filters.keys()

    @property
    def thresholds(self):
        return self._thresholds

    # Blank Contributions
    @property
    @cached()
    def _blank_contribution(self):
        return (
            self.stats_class
            .query('Class in ["Blank", "QC"]')['mean']
            .unstack('Class')
            .dropna(subset=['Blank'])
            .eval('percent = Blank / QC * 100', inplace=False)
        )

    @property
    @cached()
    def blank_contribution(self):
        s = self._blank_contribution['percent'].rename('Blank Contribution')
        b = self._thresholds['blank_contribution']
        return _Threshold(s, '>', b,
                          colors = [self._cmap(0), self._cmap(1)],
                          dist_kws = dict(clip=(0, 200)))

    # Low variability in the data
    @property
    @cached()
    def _low_variability(self):
        return (
            self.stats_group['rsd']
            .unstack('Group')
            .dropna(subset=['Sample'])
            .drop('Blank',axis=1)
            .eval('percent = Sample / QC * 100', inplace=False)
        )

    @property
    @cached()
    def low_variability(self):
        s = self._low_variability['percent'].rename('Sample Variability')
        b = self._thresholds['low_variability']
        return _Threshold(s, '<', b,
                          colors = [self._cmap(0), self._cmap(1)],
                          dist_kws = dict(clip=(0, 400)))


    @staticmethod
    def _get_count_freq(df):
        """
        get count and count_freq for flat frame
        """
        return (
            df
            .assign(count_mean = lambda x: np.where(np.isnan(x['Area']), 0.0, 1.0))
            .groupby('Feature')[['count_mean']]
            .mean()
            .eval('count_freq = count_mean  * 100', inplace=False)
        )

    @property
    @cached()
    def _sample_count(self):
        return self._get_count_freq(self.flat.query('Group=="Sample"'))

    @property
    @cached()
    def sample_count(self):
        s = self._sample_count['count_freq'].rename('Sample Count')
        b = self._thresholds['sample_count']
        e = '<='
        return _Threshold(s, e, b,
                          colors = [self._cmap(0), self._cmap(1)],
                          xlabel='Sample Detection %',
                          dist_kws = dict(clip=(0, 100)))


    @property
    @cached()
    def _qc_count(self):
        return self._get_count_freq(self.flat.query('Group=="QC"'))

    @property
    @cached()
    def qc_count(self):
        s = self._qc_count['count_freq'].rename('QC Count')
        b = self._thresholds['qc_count']
        e = '<='
        return _Threshold(s, e, b,
                          plotter='hist',
                          xlabel='QC Detection %',
                          colors = [self._cmap(0), self._cmap(1)],
                          dist_kws = dict(bins=50))



    @property
    @cached()
    def _qc_rsd(self):
        return (
            self.stats_class.dropna()
            .query('Class=="QC"')
            .reset_index('Class')[['rsd']]
        )

    @property
    @cached()
    def qc_rsd(self):
        s = self._qc_rsd['rsd'].rename('QC RSD')
        b = self._thresholds['qc_rsd']
        e = '>='
        return _Threshold(s, e, b,
                          xlabel='QC Consistency',
                          colors = [self._cmap(0), self._cmap(1)],
                          dist_kws = dict(clip=(0,100)))




    @property
    @cached()
    def _median_check(self):

        LOQ_low = self._thresholds['median_loq_low']
        LOQ_high = self._thresholds['median_loq_high']

        return (
            self.stats_group
            # munge
            .unstack('Group')
            # reorder
            .reorder_levels([1,0], axis=1)
            .pipe(flatten_columns)
            [['Blank_mean','Blank_std','Sample_median']]
            .dropna(subset=['Blank_std'])
            .assign(y=lambda x:  x['Blank_mean'] + x['Blank_std']*3)
            .assign(
                loq_low=lambda x: x['y'] * LOQ_low,
                loq_high=lambda x: x['y']* LOQ_high,
                median_ratio=lambda x: x['Sample_median'] / x['y']
            )
            # Kill eval, as @LOQ... bug in py3
            # .eval('''
            # y = Blank_mean + Blank_std*3
            # loq_low = y * @LOQ_low
            # loq_high = y * @LOQ_high
            # median_ratio = Sample_median / y''', inplace=False)
        )


    @property
    @cached()
    def median_loq_low(self):
        s = self._median_check['median_ratio'].rename("LOQ Ratio Low")
        b = self._thresholds['median_loq_low']
        e = '<='
        return _Threshold(s, e, b,
                          xlabel='LOQ Ratio',
                          colors=[self._cmap(0), self._cmap(1)],
                          dist_kws=dict(clip=(0,100)))

    @property
    @cached()
    def median_loq_high(self):
        s = self._median_check['median_ratio'].rename("LOQ Ratio High")
        b = self._thresholds['median_loq_high']
        e = '<='
        return _Threshold(s, e, b,
                          xlabel='LOQ Ratio',
                          colors=[self._cmap(0), self._cmap(1)],
                          dist_kws=dict(clip=(0,100)))


    @property
    @cached()
    def Features(self):
        """All features index"""
        return pd.Index(self.flat['Feature'].sort_values().unique(), name='Feature')

    @property
    def nFeatures(self):
        """total number of features"""
        return len(self.Features)


    @property
    @cached()
    def _wide_remove_num(self):
        idx = self.Features
        out = pd.DataFrame(None, idx)

        for filt in self.filter_names:

            s = getattr(self, filt).remove
            idx = s.index
            name = s.name
            out.loc[idx, name] = 1

        # name columns
        out.columns = out.columns.rename('Filter')
        return (
            out.fillna(0)
            .assign(all_filters = lambda x: np.sum(x.values, axis=1))
            .rename(columns = dict(all_filters='all filters'))
        )

    @property
    @cached()
    def _remove_num(self):
        return (
            self._wide_remove_num.stack('Filter').rename('Count')
        )

    @property
    @cached()
    def _wide_remove(self):
        return self._wide_remove_num > 0

    @property
    @cached()
    def _remove(self):
        return self._wide_remove.stack('Filter')


    @property
    @cached()
    def Features_remove(self):
        """index of features to be removed"""
        msk = self._wide_remove['all filters']
        return self.Features[msk]

    @property
    @cached()
    def Features_keep(self):
        """index of features to be kept"""
        msk = ~self._wide_remove['all filters']
        return self.Features[msk]


    @property
    def nFeatures_remove(self):
        return len(self.Features_remove)

    @property
    def nFeatures_keep(self):
        return len(self.Features_keep)


    def map_remove(self, ax=None, yticklabels=1000, cbar_kws=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12,12))
        if cbar_kws is None:
            cbar_kws = {}

        cbar_default = dict(
            ticks=[0.25, 0.75],
            format = plt.FuncFormatter(lambda val, loc: 'removed' if val >=0.5 else 'kept')
        )
        cbar_kws = dict(cbar_default, **cbar_kws)

        sns.heatmap(self._wide_remove, yticklabels=yticklabels, cmap=self._cmap, ax=ax,
                    cbar_kws=cbar_kws)


    def plot_dist(self, axes=None, width_per=4.0, height=4.0, despine=True, tight=True):
        if axes is None:
            fig, axes = plt.subplots(2, 3, figsize=(3 * width_per, 2* height))
        else:
            axes = np.asarray(axes).flatten()


        axes = np.asarray(axes).flatten()
        assert len(axes) >= 6
        if despine:
            for a in axes:
                sns.despine(ax=a)

        self.blank_contribution.plot(ax=axes[0])

        self.median_loq_low.plot(ax=axes[1])
        self.median_loq_high.plot(ax=axes[1])

        self.qc_count.plot(ax=axes[2])
        self.qc_rsd.plot(ax=axes[3])
        self.sample_count.plot(ax=axes[4])
        self.low_variability.plot(ax=axes[5])

        axes[0].set_ylabel('metric kernel density')

        if tight:
            axes[0].get_figure().tight_layout()

        return axes



    @property
    @cached()
    def summary_remove(self):
        """
        summary of features removed due to the various filters
        """

        L = []
        # % based filters
        for k in ['blank_contribution', 'qc_count', 'qc_rsd','sample_count','low_variability']:
            x = getattr(self, k)
            line = (len(x.remove),
                    x.s.name,
                    x.expr,
                    '{}%'.format(x.bound)
                   )
            L.append(line)

        # LOQ based filters
        for k in ['median_loq_low','median_loq_high']:
            x = getattr(self, k)
            line = (
                len(x.remove),
                x.s.name,
                x.expr,
                '{}*LOQ'.format(x.bound)
            )
            L.append(line)

        return pd.DataFrame(L, columns=['removed','filter',' ','   '])


    @property
    @cached()
    def summary_remove_markdown(self):
        """
        create summary table in markdown form
        """

        L = [self.blank_contribution.to_str(
                left='blank contribution',
                val_post='%',
                right='of mean QC feature area'),
            self.qc_count.to_str(
                left='presence in',
                val_post='%',
                right='of QC samples'),
            self.qc_rsd.to_str(
                left='QC RSD',
                val_post='%'),
            self.sample_count.to_str(
                left='presence in',
                val_post='%',
                right='of biological samples'),
            self.low_variability.to_str(
                left=' RSD ratios between biological and QC samples',
                val_post='%'),
            self.median_loq_low.to_str(
                left='median area in biological samples',
                val_post='*LOQ'),
            self.median_loq_high.to_str(
                left='median area in biological samples',
                val_post='*LOQ')
        ]

        return '\n'.join(['* '+x for x in L])


    @property
    @cached()
    def table_remove(self):
        """
        DataFrame containing data and filters to remove
        """
        rows = self.Features_remove

        # info
        a = self.table.loc[rows, self._info_columns]

        # filters
        renamer = pd.Series({True:'Removed', False: 'Kept'})
        b = (
            self._wide_remove.loc[rows]
            .drop('all filters', axis=1)
            .applymap(lambda x: renamer[x])
        )
        b.columns = pd.MultiIndex.from_product([['Filter'], b.columns])

        # data
        #
        c = self.table.loc[rows,self.Samples.tolist()]
        return pd.concat((a, b, c), join='outer', axis=1).rename_axis(None, axis=0)



    @property
    @cached()
    def table_keep(self):
        """
        DataFrame containing data to keep
        """
        return (
            self.table[self._info_columns + self.Samples.tolist()]
            .loc[self.Features_keep]
            .rename_axis(None,0)
        )



    def summary(self, input_file=None, ofile_remove=None, ofile_keep=None):
        from IPython.display import display, Markdown
        import datetime, os, pwd


        if input_file is not None:
            input_file = os.path.abspath(input_file)

        header = _header_template.format(
            date=str(datetime.datetime.now()),
            user=pwd.getpwuid(os.getuid())[0],
            input_file=input_file or ''
        )

        if ofile_keep is not None:
            self.table_keep.to_csv(ofile_keep)
            ofile_keep = os.path.abspath(ofile_keep)

        if ofile_remove is not None:
            self.table_remove.to_csv(ofile_remove)
            ofile_remove = os.path.abspath(ofile_remove)


        summary = _summary_template.format(
            nFeatures=self.nFeatures,
            nFeatures_remove=self.nFeatures_remove,
            nFeatures_keep=self.nFeatures_keep,
            summary_table=self.summary_remove_markdown,
            ofile_remove=ofile_remove,
            ofile_keep=ofile_keep
        )


        disp = lambda x: display(Markdown(x))

        disp(header)
        disp(summary)

        self.map_remove()
        plt.show()
        disp(_map_legend)

        self.plot_dist()
        plt.show()
        disp(_dist_legend)






def full_analysis(path, data=None,
                  info_columns=['Info'],
                  leading_samples=3,
                  combine_classes_on_text='Class',
                  colors=None,
                  blank_contribution=5.0,
                  median_loq_low=10.0,
                  median_loq_high=50.0,
                  qc_count=60.0,
                  qc_rsd=20.0,
                  sample_count=80.0,
                  low_variability=120.0,
                  remove_ext='_features_excluded.csv',
                  keep_ext='_features_remaining.csv'):
    """
    perform full analysis on input file

    Parameters
    ----------
    path : str
        path to input file (base for output files)
    data : pd.DataFrame, optional
        if present use this frame for analysis
    info_columns : list, default=['Info]
        the top level of the info columns
    combine_classes_on_text : str
        if Class contains this, then Group=Sample
    colors : list, default=['tab:blue','tab:orange']
        colors for keep and remove

    blank_contribution, median_loq_low, median_loq_high,
    qc_count, qc_rsd, sample_count, low_variability : floats
            Thresholds

    remove_ext, keep_ext : str
        if None, then skip saving

    Returns
    -------
    out : FRAMe object
    """

    if data is None:
        data = read_input(path)

    f = FRAMe(data,
              info_columns=info_columns,
              combine_classes_on_text=combine_classes_on_text,
              colors=colors,
              blank_contribution=blank_contribution,
              median_loq_low=median_loq_low,
              median_loq_high=median_loq_high,
              qc_count=qc_count,
              qc_rsd=qc_rsd,
              sample_count=sample_count,
              low_variability=low_variability)


    # f = FRAMe.from_input_Rfile(path,
    #                            info_columns=info_columns,
    #                            combine_classes_on_text=combine_classes_on_text,
    #                            colors=colors,
    #                            blank_contribution=blank_contribution,
    #                            median_loq_low=median_loq_low,
    #                            median_loq_high=median_loq_high,
    #                            qc_count=qc_count,
    #                            qc_rsd=qc_rsd,
    #                            sample_count=sample_count,
    #                            low_variability=low_variability)


    basename = os.path.splitext(path)[0]
    if remove_ext is not None:
        ofile_remove = basename + remove_ext
    else:
        ofile_remove = None

    if keep_ext is not None:
        ofile_keep = basename + keep_ext
    else:
        ofile_keep = None

    f.summary(input_file=path, ofile_remove=ofile_remove, ofile_keep=ofile_keep)

    return f



def check_test_FRAMe(obj):
    """
    checks if output from test.csv is good

    If no error, then good
    """
    url_remove = 'https://raw.githubusercontent.com/wpk-nist-gov/FRAMey/develop/example_data/test_excluded.csv'
    url_keep = 'https://raw.githubusercontent.com/wpk-nist-gov/FRAMey/develop/example_data/test_remaining.csv'
    df_remove= read_input(url_remove)
    df_keep = read_input(url_keep)


    pd.testing.assert_frame_equal(df_remove, obj.table_remove[df_remove.columns])
    pd.testing.assert_frame_equal(df_keep, obj.table_keep[df_keep.columns])

    




_header_template="""# Feature reduction assistant for metabalomics
### NIST Marine ESB Data Tool Development
### FRAMey v0.1: last update November 2017

---

* Timestamp: {date}
* User: {user}
* Input data: {input_file}

---
"""

_summary_template = """
* **Total Features:** {nFeatures}
* **Removed Features:** {nFeatures_remove}
* **Kept Features:** {nFeatures_keep}


{summary_table}

* The reduced dataset containing features passing all filters has been saved to: {ofile_remove}
* A dataset containing features removed by filters has been saved to: {ofile_keep}
---
"""


_map_legend = """
**Figure 1:** Visual expression of feature filtration by category.

---
"""

_dist_legend = """
**Figure 2:** Details of filter application effect. Density of occurrence for each filter metric. Red lines indicate the chosen quality thresholds. Only a reasonable range of the density curves are shown.

---
"""
