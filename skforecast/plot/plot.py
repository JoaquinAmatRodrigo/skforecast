################################################################################
#                                   Plot                                       #
#                                                                              #
# This work by Joaquin Amat Rodrigo and Javier Escobar Ortiz is licensed       #
# under a Creative Commons Attribution 4.0 International License.              #
################################################################################
# coding=utf-8
 
from typing import Union
import numpy as np
import pandas as pd
from ..utils import check_optional_dependency

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
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
    residuals: pandas Series, numpy ndarray, default `None`.
        Values of residuals. If `None`, residuals are calculated internally using
        `y_true` and `y_true`.

    y_true: pandas Series, numpy ndarray, default `None`.
        Ground truth (correct) values. Ignored if residuals is not `None`.

    y_pred: pandas Series, numpy ndarray, default `None`. 
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


def plot_multivariate_time_series_corr(
    corr: pd.DataFrame,
    ax: matplotlib.axes.Axes=None,
    **fig_kw
) -> None:
    """
    Heatmap plot of a correlation matrix.

    Parameters
    ----------
    corr : pandas DataFrame
        correlation matrix

    ax : matplotlib.axes.Axes, default `None`. 
        Pre-existing ax for the plot. Otherwise, call matplotlib.pyplot.subplots()
        internally.

    fig_kw : dict
        Other keyword arguments are passed to matplotlib.pyplot.subplots()
    
    Returns
    -------
        None

    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, **fig_kw)
    
    sns.heatmap(
        corr,
        annot=True,
        linewidths=.5,
        ax=ax,
        cmap=sns.color_palette("viridis", as_cmap=True)
    )

    ax.set_xlabel('Time series')
    fig.show()