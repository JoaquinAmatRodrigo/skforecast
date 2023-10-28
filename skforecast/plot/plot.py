################################################################################
#                                   Plot                                       #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8
 
from typing import Union, Any, Optional
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
    **fig_kw
) -> matplotlib.figure.Figure:
    """
    Parameters
    ----------
    residuals : pandas Series, numpy ndarray, default `None`.
        Values of residuals. If `None`, residuals are calculated internally using
        `y_true` and `y_true`.
    y_true : pandas Series, numpy ndarray, default `None`.
        Ground truth (correct) values. Ignored if residuals is not `None`.
    y_pred : pandas Series, numpy ndarray, default `None`. 
        Values of predictions. Ignored if residuals is not `None`.
    fig : matplotlib.figure.Figure, default `None`. 
        Pre-existing fig for the plot. Otherwise, call matplotlib.pyplot.figure()
        internally.
    fig_kw : dict
        Other keyword arguments are passed to matplotlib.pyplot.figure()

    Returns
    -------
    fig: matplotlib.figure.Figure
        Matplotlib Figure.
    
    """
    
    if residuals is None and (y_true is None or y_pred is None):
        raise ValueError(
            "If `residuals` argument is None then, `y_true` and `y_pred` must be provided."
        )
        
    if residuals is None:
        residuals = y_pred - y_true
            
    if fig is None:
        fig = plt.figure(constrained_layout=True, **fig_kw)
        
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

    return fig


def plot_multivariate_time_series_corr(
    corr: pd.DataFrame,
    ax: matplotlib.axes.Axes=None,
    **fig_kw
) -> matplotlib.figure.Figure:
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
    fig: matplotlib.figure.Figure
        Matplotlib Figure.

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
    
    return fig
    

def plot_prediction_distribution(
    bootstrapping_predictions: pd.DataFrame,
    bw_method: Optional[Any]=None,
    **fig_kw
) -> matplotlib.figure.Figure:
    """
    Ridge plot of bootstrapping predictions. This plot is very useful to understand 
    the uncertainty of forecasting predictions.

    Parameters
    ----------
    bootstrapping_predictions : pandas DataFrame
        Bootstrapping predictions created with `Forecaster.predict_bootstrapping`.
    bw_method : str, scalar, Callable, default `None`
        The method used to calculate the estimator bandwidth. This can be 'scott', 
        'silverman', a scalar constant or a Callable. If None (default), 'scott' 
        is used. See scipy.stats.gaussian_kde for more information.
    fig_kw : dict
        All additional keyword arguments are passed to the `pyplot.figure` call.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib Figure.
    
    """

    index = bootstrapping_predictions.index.astype(str).to_list()[::-1]
    palette = sns.cubehelix_palette(len(index), rot=-.25, light=.7, reverse=False)
    fig, axs = plt.subplots(len(index), 1, sharex=True, **fig_kw)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    for i, step in enumerate(index):
        plot = (
            bootstrapping_predictions.loc[step, :]
            .plot.kde(ax=axs[i], bw_method=bw_method, lw=0.5)
        )

        # Fill density area
        x = plot.get_children()[0]._x
        y = plot.get_children()[0]._y
        axs[i].fill_between(x, y, color=palette[i])
        prediction_mean = bootstrapping_predictions.loc[step, :].mean()
        
        # Closest point on x to the prediction mean
        idx = np.abs(x - prediction_mean).argmin()
        axs[i].vlines(x[idx], ymin=0, ymax=y[idx], linestyle="dashed", color='w')

        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
        axs[i].set_yticklabels([])
        axs[i].set_yticks([])
        axs[i].set_ylabel(step, rotation='horizontal')
        axs[i].set_xlabel('prediction')

    fig.subplots_adjust(hspace=-0)
    fig.suptitle('Forecasting distribution per step')

    return fig
