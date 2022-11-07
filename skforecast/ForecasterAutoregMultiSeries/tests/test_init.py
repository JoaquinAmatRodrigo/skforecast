# Unit test __init__ ForecasterAutoregMultiSeries
# ==============================================================================
import re
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.linear_model import LinearRegression


def test_init_exception_when_series_weights_is_not_a_dict():
    """
    Test exception is raised when series_weights is not a dict.
    """
    series_weights = 'not_callable_or_dict'
    err_msg = re.escape(f"Argument `series_weights` must be a dict of floats. Got {type(series_weights)}.")
    with pytest.raises(TypeError, match = err_msg):
        ForecasterAutoregMultiSeries(LinearRegression(), lags=3, series_weights=series_weights)


def test_init_exception_when_weight_func_is_not_a_callable_or_dict():
    """
    Test exception is raised when weight_func is not a callable or a dict.
    """
    weight_func = 'not_callable_or_dict'
    err_msg = re.escape(f"Argument `weight_func` must be a callable or a dict of callables. Got {type(weight_func)}.")
    with pytest.raises(TypeError, match = err_msg):
        ForecasterAutoregMultiSeries(LinearRegression(), lags=3, weight_func=weight_func)