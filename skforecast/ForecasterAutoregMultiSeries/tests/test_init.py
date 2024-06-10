# Unit test __init__ ForecasterAutoregMultiSeries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.linear_model import LinearRegression


@pytest.mark.parametrize("dif", 
                         [0, 0.5, 1.5, 'not_int'], 
                         ids = lambda dif : f'differentiation: {dif}')
def test_init_ValueError_when_differentiation_argument_is_not_int_or_greater_than_0(dif):
    """
    Test ValueError is raised when differentiation is not an int or greater than 0.
    """
    err_msg = re.escape(
        (f"Argument `differentiation` must be an integer equal to or "
         f"greater than 1. Got {dif}.")
    )
    with pytest.raises(ValueError, match = err_msg):
         ForecasterAutoregMultiSeries(
             regressor       = LinearRegression(),
             lags            = 5,
             differentiation = dif
         )


@pytest.mark.parametrize("dif", 
                         [1, 2], 
                         ids = lambda dif : f'differentiation: {dif}')
def test_init_window_size_is_increased_when_differentiation(dif):
    """
    Test window_size is increased when including differentiation.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor       = LinearRegression(),
                     lags            = 5,
                     differentiation = dif
                 )
    
    assert forecaster.window_size == len(forecaster.lags)
    assert forecaster.window_size_diff == len(forecaster.lags) + dif


def test_ForecasterAutoregMultiSeries_init_invalid_encoding():
    """
    Test ValueError is raised when encoding is not valid.
    """

    err_msg = re.escape(
        ("Argument `encoding` must be one of the following values: 'ordinal', "
         "'ordinal_category', 'onehot'. Got 'invalid_encoding'.")
    )
    with pytest.raises(ValueError, match = err_msg):
        ForecasterAutoregMultiSeries(
            regressor = LinearRegression(),
            lags      = [1, 2, 3],
            encoding  = 'invalid_encoding',
        )


def test_ForecasterAutoregMultiSeries_init_not_scaling_with_linear_model():
    """
    Test Warning is raised when Forecaster has no transformer_series and it
    is using a linear model.
    """

    warn_msg = re.escape(
        ("When using a linear model, it is recommended to use a transformer_series "
         "to ensure all series are in the same scale. You can use, for example, a "
         "`StandardScaler` from sklearn.preprocessing.") 
    )
    with pytest.warns(UserWarning, match = warn_msg):
        ForecasterAutoregMultiSeries(
            regressor = LinearRegression(),
            lags      = [1, 2, 3]
        )