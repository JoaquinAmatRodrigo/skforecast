# Unit test __init__ ForecasterAutoregMultiSeries
# ==============================================================================
import re
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor



def test_init_exception_when_series_weights_is_not_a_dict():
    """
    Test exception is raised when series_weights is not a dict.
    """
    series_weights = 'not_callable_or_dict'
    err_msg = re.escape(
        f"Argument `series_weights` must be a dict of floats or ints."
        f"Got {type(series_weights)}."
    )
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


def test_init_when_weight_func_is_provided_and_regressor_has_not_sample_weights():
    """
    Test warning is created when weight_func is provided but the regressor has no argument
    sample_weights in his fit method.
    """

    def weight_func():
        pass

    warn_msg = re.escape(
                    f"""
                    Argument `weight_func` is ignored since regressor KNeighborsRegressor()
                    does not accept `sample_weight` in its `fit` method.
                    """
                )
    with pytest.warns(UserWarning, match = warn_msg):
        forecaster = ForecasterAutoregMultiSeries(
                        regressor      = KNeighborsRegressor(),
                        lags           = 3,
                        weight_func    = weight_func
                     )
    
    assert forecaster.weight_func is None
    assert forecaster.source_code_weight_func is None


