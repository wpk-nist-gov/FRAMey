"""
Code to summarize data
"""

from __future__ import print_function, division, absolute_import

import datetime, os, pwd
from IPython.display import display, Markdown

import matplotlib.pyplot as plt

from .core import (
    read_input, FRAMe
)



def summarize_FRAMe(f, input_file=None, remove_file=None, keep_file=None):

    if input_file is not None:
        input_file = os.path.abspath(input_file)

    header = _header_template.format(
        date=str(datetime.datetime.now()),
        user=pwd.getpwuid(os.getuid())[0],
        input_file=input_file or ''
    )

    if keep_file is not None:
        f.table_keep.to_csv(keep_file)
        keep_file = os.path.abspath(keep_file)

    if remove_file is not None:
        f.table_remove.to_csv(remove_file)
        remove_file = os.path.abspath(remove_file)


    summary = _summary_template.format(
        nFeatures=f.nFeatures,
        nFeatures_remove=f.nFeatures_remove,
        nFeatures_keep=f.nFeatures_keep,
        summary_table=f.summary_remove_markdown,
        remove_file=remove_file,
        keep_file=keep_file
    )


    disp = lambda x: display(Markdown(x))

    disp(header)
    disp(summary)

    f.map_remove()
    plt.show()
    disp(_map_legend)

    f.plot_dist()
    plt.show()
    disp(_dist_legend)

# Templates for output

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

* The reduced dataset containing features passing all filters has been saved to: {remove_file}
* A dataset containing features removed by filters has been saved to: {keep_file}
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
                  remove_file=None,
                  keep_file=None,
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

    remove_file, keep_file : str, optional
        if supplied, save removed and keep in these files

    remove_ext, keep_ext : str, optionoal.
        if None, then skip saving.  If supplied,
        remove (and keep) in form basename(path) + remove_ext

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
    if remove_file is None:
        if remove_ext is not None:
            remove_file = basename + remove_ext

    if keep_file is None:
        if keep_ext is not None:
            keep_file = basename + keep_ext

    summarize_FRAMe(f, input_file=path, remove_file=remove_file, keep_file=keep_file)
    # f.summary(input_file=path, remove_file=remove_file, keep_file=keep_file)

    return f
