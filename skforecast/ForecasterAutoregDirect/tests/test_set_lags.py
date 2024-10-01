# Unit test set_lags ForecasterAutoregDirect
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.preprocessing import RollingFeatures
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect


@pytest.mark.parametrize("lags", 
                         [3, [1, 2, 3], np.array([1, 2, 3]), range(1, 4)],
                         ids = lambda lags: f'lags: {lags}')
def test_set_lags_with_different_inputs(lags):
    """
    Test how lags and max_lag attributes change with lags argument of different types.
    """
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=5, steps=2)
    forecaster.set_lags(lags=lags)

    np.testing.assert_array_almost_equal(forecaster.lags, np.array([1, 2, 3]))
    assert forecaster.max_lag == 3
    assert forecaster.window_size == 3


def test_set_lags_when_differentiation_is_not_None():
    """
    Test how `window_size` is also updated when the forecaster includes 
    differentiation.
    """
    forecaster = ForecasterAutoregDirect(
                     regressor       = LinearRegression(),
                     lags            = 3,
                     steps           = 2,
                     differentiation = 1
                 )
    
    forecaster.set_lags(lags=5)

    np.testing.assert_array_almost_equal(forecaster.lags, np.array([1, 2, 3, 4, 5]))
    assert forecaster.max_lag == 5
    assert forecaster.window_size == 5 + 1


def test_set_lags_when_window_features():
    """
    Test how `window_size` is also updated when the forecaster includes
    window_features.
    """
    rolling = RollingFeatures(stats='mean', window_sizes=6)
    forecaster = ForecasterAutoregDirect(
                     regressor       = LinearRegression(),
                     lags            = 9,
                     steps           = 2,
                     window_features = rolling
                 )
    
    forecaster.set_lags(lags=5)

    np.testing.assert_array_almost_equal(forecaster.lags, np.array([1, 2, 3, 4, 5]))
    assert forecaster.max_lag == 5
    assert forecaster.max_size_window_features == 6
    assert forecaster.window_size == 6


def test_set_lags_to_None():
    """
    Test how lags and max_lag attributes change when lags is set to None.
    """
    rolling = RollingFeatures(stats='mean', window_sizes=3)
    forecaster = ForecasterAutoregDirect(
                     regressor       = LinearRegression(),
                     lags            = 5,
                     steps           = 3,
                     window_features = rolling
                 )
    
    forecaster.set_lags(lags=None)

    assert forecaster.lags is None
    assert forecaster.max_lag is None
    assert forecaster.max_size_window_features == 3
    assert forecaster.window_size == 3
