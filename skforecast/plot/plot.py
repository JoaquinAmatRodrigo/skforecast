################################################################################
#                                   Plot                                       #
#                                                                              #
# This work by Joaquin Amat Rodrigo and Javier Escobar Ortiz is licensed       #
# under a Creative Commons Attribution 4.0 International License.              #
################################################################################
# coding=utf-8
 
from typing import Union
from ..utils import check_optional_dependency

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from statsmodels.graphics.tsaplots import plot_acf
except Exception as e:
    package_name = str(e).split(" ")[-1].replace("'", "")
    check_optional_dependency(package_name=package_name)


def plot_residuals(
    residuals: Union[np.ndarray, pd.Series]=None,
    y_true: Union[np.ndarray, pd.Series]=None,
    y_pred: Union[np.ndarray, pd.Series]=None,
    fig: matplotlib.figure.Figure=None,
    **kwargs
) -> None:
    """
    Parameters
    ----------
    residuals: pandas series, numpy ndarray, default `None`.
        Values of residuals. If `None`, residuals are calculated internally using
        `y_true` and `y_true`.

    y_true: pandas series, numpy ndarray, default `None`.
        Ground truth (correct) values. Ignored if residuals is not `None`.

    y_pred: pandas series, numpy ndarray, default `None`. 
        Values of predictions. Ignored if residuals is not `None`.
        
    fig: matplotlib.figure.Figure, default `None`. 
        Pre-existing fig for the plot. Otherwise, call matplotlib.pyplot.figure()
        internally.
        
    kwargs
        Other keyword arguments are passed to matplotlib.pyplot.figure()
        
    Returns
    -------
        None
    
    """
    
    if residuals is None and (y_true is None or y_pred is None):
        raise ValueError(
            "If `residuals` argument is None then, `y_true` and `y_pred` must be provided."
        )
        
    if residuals is None:
        residuals = y_pred - y_true
            
    if fig is None:
        fig = plt.figure(constrained_layout=True, **kwargs)
        
    gs  = matplotlib.gridspec.GridSpec(2, 2, figure=fig)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[1, 1])
    
    ax1.plot(residuals)
    sns.histplot(residuals, kde=True, bins=30, ax=ax2)
    plot_acf(residuals, ax=ax3, lags=60)
    
    ax1.set_title("Residuals")
    ax2.set_title("Distribution")
    ax3.set_title("Autocorrelation")