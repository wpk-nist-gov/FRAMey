"""
Set of routines to work like FRAMe R-code by Jared Ragland
"""

from __future__ import print_function, division, absolute_import
from builtins import *

import pandas as pd
import numpy as np
import statsmodels.nonparametric.api as smnp
import matplotlib.pyplot as plt
import seaborn.apionly as sns

from .cached_decorators import cached, cached_clear


__author__ = 'William Krekelberg'
__email__ = 'wpk@nist.gov'


def _parse_input_flat(df):
    _t = df.T

    # setup columns
    columns = _t.iloc[0].values
    _t.columns = columns
    _t.index.name = 'Sample'

    # types
    #_t = _t.iloc[1:].set_index('Class',append=True)

    _t = pd.melt(_t.iloc[1:].reset_index(), id_vars=['Sample','Class'], var_name='Feature', value_name='Area')
    # types
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
    def __init__(self, var_name=None, expr='<', val=1.0):
        """ perform operations of form df[var_name] .expr. val"""
        self.var_name = var_name
        self.expr = expr
        self._attr = _attr_map[self.expr]
        self.val = val

    def _parse_var_name(self, var_name=None):
        var_name = var_name or self.var_name
        if var_name is None:
            raise ValueError('must specify var_name')
        return var_name

    def query_frame(self, df, var_name=None):
        var_name = self._parse_var_name(var_name)
        query = '{} {} {}'.format(var_name, self._expr, self.val)
        #return query
        return df.query(query)

    def mask_frame(self, df, var_name=None):
        var_name = self._parse_var_name(var_name)
        expr = getattr(df[var_name], self._attr)
        return expr(self.val)

    def mask(self, x):
        expr = getattr(x, self._attr)
        return expr(self.val)

class FRAMe(object):

    def __init__(self, data,
                 thresholds=None,
                 params=None,
                 colors=None,
                 cmap=None):

        self._cache = {}
        self._data = data

        # thresholds
        thresholds = thresholds or {}
        default_thresholds = dict(
            blank_contribution = 5,
            qc_rsd = 20,
            qc_count_freq = 60,
            sample_count_freq = 80,
            rsd_ratios = 120,
        )
        thresholds = dict(default_thresholds, **thresholds)
        self._thresholds = thresholds

        # parameters
        if params is None:
            params = {}
        default_params = dict(
            LOQ_low = 10,
            LOQ_high = 50,
            leading_qc_samples = 3,
            combine_classes_on_text = 'Class',
        )
        params = dict(default_params, **params)
        self._params = params

        if colors is not None:
            colors = ['tab:blue','tab:orange']

        # cmap
        if cmap is None:
            cmap = plt.cm.gray_r
        cmap = cmap._resample(2)
        self._cmap = cmap

    @property
    def thresholds(self):
        return self._thresholds

    @thresholds.setter



    @property
    def table(self):
        return self._data

    @property
    @cached()
    def table_group(self):
        """
        add group column to table
        """

        rename = lambda x: (
            np.where(x['Class'].str.contains(self._params['combine_classes_on_text']),
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


    @property
    @cached()
    def _blank_contritribution(self):
        return (
            self.summary_class_feature
            .query('Class in ["Blank", "QC"]')['mean']
            .unstack('Class')
            .dropna(subset=['Blank'])
            .eval('percent = Blank / QC * 100', inplace=False)
        )

    @property
    @cached()
    def _remove_blank_contribution(self):
        val = self._thresholds['blank_contribution']
        return (
            self._blank_contritribution
            .query('percent > @val')
        )

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
    def _remove_low_variability(self):
        val = self._thresholds['rsd_ratios']
        return (
            self._low_variability
            .query('percent < @val')
        )

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
    def _remove_sample_count(self):
        val = self._thresholds['sample_count_freq']
        return (
            self._sample_count
            .query('count_freq <= @val')
        )

    @property
    @cached()
    def _qc_count(self):
        return self._get_count_freq(self.table_group.query('Group=="QC"'))

    @property
    @cached()
    def _remove_qc_count(self):
        val = self._thresholds['qc_count_freq']
        return (
            self._qc_count
            .query('count_freq <= @val')
        )

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
    def _remove_qc_rsd(self):
        val = self._thresholds['qc_rsd']
        return(
            self._qc_rsd
            .query('rsd >= @val')
        )



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

        LOQ_low = self._params['LOQ_low']
        LOQ_high = self._params['LOQ_high']
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
    def _remove_median_LOQ_low(self):
        val = self._params['LOQ_low']
        return (
            self._median_check
            .query('median_ratio <= @val')
        )

    @property
    @cached()
    def _remove_median_LOQ_high(self):
        val = self._params['LOQ_high']
        return (
            self._median_check
            .query('median_ratio <= @val')
        )


    @property
    @cached()
    def _wide_remove_num(self):
        idx = pd.Index(self.table['Feature'].sort_values().unique(), name='Feature')
        out = pd.DataFrame(None, idx)

        for sub in [
                '_remove_blank_contribution',
                '_remove_median_LOQ_low',
                '_remove_median_LOQ_high',
                '_remove_qc_count',
                '_remove_qc_rsd',
                '_remove_sample_count',
                '_remove_low_variability'
        ]:

            idx = getattr(self, sub).index

            name = sub.replace('_remove_','')
            out.loc[idx, name] = 1

        # name columns
        out.columns = out.columns.rename('Filter')
        return (
            out.fillna(0)
            .assign(all_filters = lambda x: np.sum(x.values, axis=1))
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


    def map_removed(self, ax=None, yticklabels=1000, cbar_kws=None):


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

    @staticmethod
    def _make_kde_base(data, **kwargs):
        sns.kdeplot(data, **kwargs)
        ax = plt.gca()
        x, y = ax.lines[0].get_data()
        return x, y

    def _kde_blank_contribution(self, **kwargs):
        pass
        #x, y = 




    @classmethod
    def from_input_csv(cls, filename, info_columns=None, leading_samples=3, thresholds=None, params=None, **kwargs):
        if info_columns == None:
            info_columns = []

        df = (
            pd.read_csv(filename, **kwargs)
            .drop(info_columns,axis=1)
            .pipe(_parse_input_flat)
            .pipe(_drop_leading_QC, leading_samples)
        )
        return cls(df, thresholds=thresholds, params=params)

        






