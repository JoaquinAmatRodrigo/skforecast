################################################################################
#                                metrics                                       #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from typing import Union, Callable
import numpy as np
import pandas as pd
import inspect
from functools import wraps
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_log_error,
)

def _get_metric(metric: str) -> Callable:
    """
    Get the corresponding scikit-learn function to calculate the metric.

    Parameters
    ----------
    metric : str
        Metric used to quantify the goodness of fit of the model.

    Returns
    -------
    metric : Callable
        scikit-learn function to calculate the desired metric.

    """
    allowed_metrics = [
        "mean_squared_error",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_log_error",
        "mean_absolute_scaled_error",
        "root_mean_squared_scaled_error",
    ]

    if metric not in allowed_metrics:
        raise ValueError((f"Allowed metrics are: {allowed_metrics}. Got {metric}."))

    metrics = {
        "mean_squared_error": mean_squared_error,
        "mean_absolute_error": mean_absolute_error,
        "mean_absolute_percentage_error": mean_absolute_percentage_error,
        "mean_squared_log_error": mean_squared_log_error,
        "mean_absolute_scaled_error": mean_absolute_scaled_error,
        "root_mean_squared_scaled_error": root_mean_squared_scaled_error,
    }

    metric = add_y_train_argument(metrics[metric])

    return metric


def add_y_train_argument(func):
    """
    Add `y_train` argument to a function if it is not already present.

    Parameters
    ----------
    func : callable
        Function to which the argument is added.

    Returns
    -------
    wrapper : callable
        Function with `y_train` argument added.
    """
    sig = inspect.signature(func)
    
    if "y_train" in sig.parameters:
        return func

    new_params = list(sig.parameters.values()) + [
        inspect.Parameter("y_train", inspect.Parameter.KEYWORD_ONLY, default=None)
    ]
    new_sig = sig.replace(parameters=new_params)

    @wraps(func)
    def wrapper(*args, y_train=None, **kwargs):
        return func(*args, **kwargs)
    
    wrapper.__signature__ = new_sig
    
    return wrapper


def mean_absolute_scaled_error(
    y_true: Union[pd.Series, np.array],
    y_pred: Union[pd.Series, np.array],
    y_train: Union[pd.Series, np.array],
) -> float:
    """
    Mean Absolute Scaled Error (MASE)
    MASE is a scale-independent error metric that measures the accuracy of
    a forecast. It is the mean absolute error of the forecast divided by the
    mean absolute error of a naive forecast in the training set. The naive
    forecast is the one obtained by shifting the time series by one period.

    Parameters
    ----------
    y_true : Union[pd.Series, np.array]
        True values of the target variable.
    y_pred : Union[pd.Series, np.array]
        Predicted values of the target variable.
    y_train : Union[pd.Series, np.array]
        True values of the target variable in the training set.

    Returns
    -------
    float
        MASE value.
    """

    return np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(np.diff(y_train)))


def root_mean_squared_scaled_error(
    y_true: Union[pd.Series, np.array],
    y_pred: Union[pd.Series, np.array],
    y_train: Union[pd.Series, np.array],
) -> float:
    """
    Root Mean Squared Scaled Error (RMSSE)
    RMSSE is a scale-independent error metric that measures the accuracy of
    a forecast. It is the root mean squared error of the forecast divided by
    the root mean squared error of a naive forecast in the training set. The
    naive forecast is the one obtained by shifting the time series by one period.

    Parameters
    ----------
    y_true : Union[pd.Series, np.array]
        True values of the target variable.
    y_pred : Union[pd.Series, np.array]
        Predicted values of the target variable.
    y_train : Union[pd.Series, np.array]
        True values of the target variable in the training set.

    Returns
    -------
    float
        RMSSE value.
    """

    return np.sqrt(np.mean((y_true - y_pred) ** 2)) / np.sqrt(
        np.mean(np.diff(y_train) ** 2)
    )
