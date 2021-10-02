import typing
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf

def plot_residuals(residuals: np.array=None, y_true: np.array=None,
                   y_pred: np.array=None, ax: matplotlib.axes.Axes=None,
                   **kwargs):
    '''
    
    Parameters
    ----------
    residuals: np.array, default `None`.
        
    y_true: np.array, default `None`.
                   
    y_pred: np.array, default `None`. 
    
    fig: matplotlib.figure.Figure, default `None`. 
        Pre-existing fig for the plot. Otherwise, call matplotlib.pyplot.gca() internally.
        
    kwargs
        Other keyword arguments are passed to matplotlib.pyplot.figure()
    '''
    
    if residuals is None and (y_true is None or y_pred is None):
        raise Exception(
            "If `residuals` aregument is None then, y_true and y_pred must be provided."
        )
        
    if residuals is None:
        residuals = y_pred - y_true
        
    

    #TODO: icnluir índice temporal
    
    fig = plt.figure(constrained_layout=True, **kwargs)
    gs = matplotlib.gridspec.GridSpec(2, 2, figure=fig)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[1, 1])
    
    ax1.plot(residuals)
    sns.histplot(residuals, kde=True, bins=30, ax=ax2)
    plot_acf(residuals, ax=ax3, lags=60)
    
    ax1.set_title("Residuals")
    ax2.set_title("Distribution")
    ax3.set_title("Autocorrelation")
