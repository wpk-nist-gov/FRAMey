"""
Set of routines to work like FRAMe R-code by Jared Ragland
"""

from __future__ import print_function, division, absolute_import
from builtins import *

from collections import OrderedDict

import pandas as pd
import numpy as np
import statsmodels.nonparametric.api as smnp
import matplotlib.pyplot as plt
import seaborn.apionly as sns
from matplotlib.colors import LinearSegmentedColormap

from .cached_decorators import cached, cached_clear



__author__ = 'William Krekelberg'
__email__ = 'wpk@nist.gov'

def _parse_input_flat(df):
    # NOTE: this version faster than below
    _t = df.T

    # setup columns
    columns = _t.iloc[0].values
    _t.columns = columns
    _t.index.name = 'Sample'


    _t = pd.melt(_t.iloc[1:].reset_index(), id_vars=['Sample','Class'], var_name='Feature', value_name='Area')
    # types
    for x in ['Sample','Feature']:
        _t[x]= _t[x].astype(np.int)
    for x in ['Area']:
        _t[x] = _t[x].astype(np.float)
    return _t#.set_index(['Sample','Class'])

def _parse_input_flat2(df):
    _t = (
        df
        .transpose()
        # renmae index
        .rename_axis('Sample',axis=0)
        # rename columns
        .pipe(lambda x: x.rename(columns = x.iloc[0]))
        # drop first row
        .drop('Sample', axis=0)
        .set_index('Class',append=True)
        # rename columns
        .rename_axis('Feature',axis=1)
        # flatten
        .stack('Feature',dropna=False)
        # to dataframe
        .reset_index(name='Area')
        # types
        # .assign(
        #     Sample = lambda x: x['Sample'].astype(np.int),
        #     Feature = lambda x: x['Feature'].astype(np.int),
        #     Area = lambda x: x['Area'].astype(np.float)
        # )
    )
    for x in ['Sample','Feature']:
        _t[x]= _t[x].astype(np.int)
    for x in ['Area']:
        _t[x] = _t[x].astype(np.float)
    return _t#.set_index(['Sample','Class'])




def _drop_leading_QC(df, leading_samples=3):
    samples = (
        df.query('Class=="QC"')['Sample']
        .drop_duplicates()
        .sort_values()
        .values[:leading_samples]
    )
    return df.query('Class != "QC" or Sample not in @samples')


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
            xlabel = self.s.name
        ax.set_xlabel(xlabel)





class _FRAMe(object):
    """
    Base class for frame
    """
    def __init__(self, data, info_data=None,
                 combine_classes_on_text='Class',
                 colors=None, cmap=None):

        self._cache = {}

        self.table = data
        self._info_data = info_data

        self._combine_classes_on_text = combine_classes_on_text

        self._set_colors(colors, cmap)


    @property
    def table(self):
        return self._data

    @table.setter
    @cached_clear()
    def table(self, val):
        self._data = val


    @property
    @cached()
    def table_group(self):
        """
        add group column to table
        """

        rename = lambda x: (
            np.where(x['Class'].str.contains(self._combine_classes_on_text),
                     'Sample',
                     x['Class']))
        return (
            self.table
            .assign(Group = rename)
        )


    @property
    @cached()
    def wide_table_group(self):

        return (
            self.table_group
            .set_index(['Sample','Group', 'Class','Feature'])['Area']
            .unstack('Feature')
        )

    @property
    @cached()
    def summary_class_feature(self):
        return (
            self.table
            .dropna(subset=['Area'])
            .groupby(['Class','Feature'])['Area']
            .agg(['mean','median','std'])
            .eval('rsd = std / mean * 100', inplace=False)
        )

    @property
    @cached()
    def summary_group_feature(self):
        # func to replace Class names with Sample
        return (
            self.table_group
            .dropna(subset=['Area'])
            .groupby(['Group','Feature'])['Area']
            .agg(['mean','median', 'std'])
            .eval('rsd = std / mean * 100', inplace=False)
        )


    @classmethod
    def from_input_RFrame(cls, df, info_columns=None, leading_samples=3, **kwargs):
        if info_columns is None:
            info_data = None
            info_columns = []
        else:
            info_data = df.set_index('Sample')[info_columns]

        df = (
            df
            .drop(info_columns,axis=1)
            .pipe(_parse_input_flat)
            .pipe(_drop_leading_QC, leading_samples)
        )
        return cls(df, info_data=info_data, **kwargs)

    @classmethod
    def from_input_Rfile(cls, filename, info_columns=None, leading_samples=3,
                         read_kws=None,
                         **kwargs):

        read_kws = read_kws or {}
        df = pd.read_csv(filename, **read_kws)
        return cls.from_input_RFrame(df,
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


    def __init__(self, data, info_data=None,
                 combine_classes_on_text='Class',
                 colors=None, cmap=None,
                 **thresholds):
        """
        Parameters
        ----------
        data : pandas.DataFrame
            input data
        info_data : pandas.DataFrame
        """

        super(FRAMe, self).__init__(data=data, info_data=info_data,
                           combine_classes_on_text=combine_classes_on_text,
                           colors=colors, cmap=cmap)


        # setup thresholds
        self._thresholds = self._default_filters.copy()
        self.update_thresholds(**thresholds)

    @cached_clear()
    def update_thresholds(self,**kwargs):
        for k in kwargs:
            if k not in self._default_filters:
                raise ValueError('{} not a valid filter name'.format(k))
            self._thresholds[k] = kwargs[k]

    @property
    def filter_names(self):
        return self._default_filters.keys()

    # Blank Contributions
    @property
    @cached()
    def _blank_contribution(self):
        return (
            self.summary_class_feature
            .query('Class in ["Blank", "QC"]')['mean']
            .unstack('Class')
            .dropna(subset=['Blank'])
            .eval('percent = Blank / QC * 100', inplace=False)
        )

    @property
    @cached()
    def blank_contribution(self):
        s = self._blank_contribution['percent'].rename('blank contribution')
        b = self._thresholds['blank_contribution']
        return _Threshold(s, '>', b,
                          colors = [self._cmap(0), self._cmap(1)],
                          dist_kws = dict(clip=(0, 200)))

    # Low variability in the data
    @property
    @cached()
    def _low_variability(self):
        return (
            self.summary_group_feature['rsd']
            .unstack('Group')
            .dropna(subset=['Sample'])
            .drop('Blank',axis=1)
            .eval('percent = Sample / QC * 100', inplace=False)
        )

    @property
    @cached()
    def low_variability(self):
        s = self._low_variability['percent'].rename('low variability')
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
        return self._get_count_freq(self.table_group.query('Group=="Sample"'))

    @property
    @cached()
    def sample_count(self):
        s = self._sample_count['count_freq'].rename('Sample Detection %')
        b = self._thresholds['sample_count']
        e = '<='
        return _Threshold(s, e, b,
                          colors = [self._cmap(0), self._cmap(1)],
                          dist_kws = dict(clip=(0, 100)))


    @property
    @cached()
    def _qc_count(self):
        return self._get_count_freq(self.table_group.query('Group=="QC"'))

    @property
    @cached()
    def qc_count(self):
        s = self._qc_count['count_freq'].rename('QC Detection %')
        b = self._thresholds['qc_count']
        e = '<='
        return _Threshold(s, e, b,
                          plotter='hist',
                          colors = [self._cmap(0), self._cmap(1)],
                          dist_kws = dict(bins=50))



    @property
    @cached()
    def _qc_rsd(self):
        return (
            self.summary_class_feature.dropna()
            .query('Class=="QC"')
            .reset_index('Class')[['rsd']]
        )

    @property
    @cached()
    def qc_rsd(self):
        s = self._qc_rsd['rsd'].rename('QC consistency')
        b = self._thresholds['qc_rsd']
        e = '>='
        return _Threshold(s, e, b,
                          colors = [self._cmap(0), self._cmap(1)],
                          dist_kws = dict(clip=(0,100)))




    @property
    @cached()
    def _median_check(self):
        check_blanks = (
            self.summary_group_feature
            .query('Group=="Blank"')
            .reset_index('Group')
            [['mean','std']]
            .rename(columns = lambda x: 'Blank_' + x)
        )

        check_samples = (
            self.summary_group_feature
            .query('Group=="Sample"')
            .reset_index('Group')
            [['median']]
            .rename(columns = lambda x: 'Sample_' + x)
        )

        LOQ_low = self._thresholds['median_loq_low']
        LOQ_high = self._thresholds['median_loq_high']

        return (
            pd.concat((check_blanks, check_samples),axis=1)
            .dropna(subset=['Blank_std'])
            .eval('''
            y = Blank_mean + Blank_std*3
            loq_low = y * @LOQ_low
            loq_high = y * @LOQ_high
            median_ratio = Sample_median / y''', inplace=False)
        )


    @property
    @cached()
    def median_loq_low(self):
        s = self._median_check['median_ratio'].rename("LOQ ratio low")
        b = self._thresholds['median_loq_low']
        e = '<='
        return _Threshold(s, e, b,
                          colors=[self._cmap(0), self._cmap(1)],
                          dist_kws=dict(clip=(0,100)))

    @property
    @cached()
    def median_loq_high(self):
        s = self._median_check['median_ratio'].rename("LOQ ratio high")
        b = self._thresholds['median_loq_high']
        e = '<='
        return _Threshold(s, e, b,
                          colors=[self._cmap(0), self._cmap(1)],
                          dist_kws=dict(clip=(0,100)))


    @property
    @cached()
    def Features_index(self):
        return pd.Index(self.table['Feature'].sort_values().unique(), name='Feature')



    @property
    @cached()
    def _wide_remove_num(self):
        idx = self.Features_index
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
    def nFeatures(self):
        return len(self.Features_index)

    @property
    @cached()
    def nFeatures_remove(self):
        return self._wide_remove['all filters'].sum()

    @property
    @cached()
    def nFeatures_keep(self):
        return self.nFeatures - self.nFeatures_remove


    def map_remove(self, ax=None, yticklabels=1000, cbar_kws=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,10))
        if cbar_kws is None:
            cbar_kws = {}

        cbar_default = dict(
            ticks=[0.25, 0.75],
            format = plt.FuncFormatter(lambda val, loc: 'removed' if val >=0.5 else 'kept')
        )
        cbar_kws = dict(cbar_default, **cbar_kws)

        sns.heatmap(self._wide_remove, yticklabels=yticklabels, cmap=self._cmap, ax=ax,
                    cbar_kws=cbar_kws)


    def plot_dist(self, axes=None, despine=True):
        if axes is None:
            fig, axes = plt.subplots(1, 6, figsize=(6*2, 3))
        else:
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




    @property
    @cached()
    def summary_remove(self):
        L = []
        for k in ['blank_contribution', 'qc_count', 'qc_rsd','sample_count','low_variability']:
            x = getattr(self, k)
            line = (len(x.remove),
                    x.s.name,
                    x.expr,
                    '{}%'.format(x.bound)
                   )
            L.append(line)


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

