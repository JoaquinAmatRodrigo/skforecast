# Unit test set_lags ForecasterAutoreg
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression 


def test_set_lags_when_lags_argument_is_int():
    """
    Test how lags and max_lag attributes change when lags argument is integer
    positive (5).
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.set_lags(lags=5)

    assert (forecaster.lags == np.array([1, 2, 3, 4, 5])).all()
    assert forecaster.max_lag == 5


def test_set_lags_when_lags_argument_is_list():
    """
    Test how lags and max_lag attributes change when lags argument is a list
    of positive integers.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.set_lags(lags=[1,2,3])

    assert (forecaster.lags == np.array([1, 2, 3])).all()
    assert forecaster.max_lag == 3


def test_set_lags_when_lags_argument_is_1d_numpy_array():
    """
    Test how lags and max_lag attributes change when lags argument is 1d numpy
    array of positive integers.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.set_lags(lags=np.array([1,2,3]))

    assert (forecaster.lags == np.array([1, 2, 3])).all()
    assert forecaster.max_lag == 3


def test_set_lags_when_differentiation_is_not_None():
    """
    Test how `window_size` is also updated when the forecaster includes differentiation.
    """
    lags = 5
    differentiation = 1

    forecaster = ForecasterAutoreg(
                     regressor       = LinearRegression(),
                     lags            = lags,
                     differentiation = differentiation
                 )
    
    forecaster.set_lags(lags=lags)

    assert (forecaster.lags == np.array([1, 2, 3, 4, 5])).all()
    assert forecaster.max_lag == 5
    assert forecaster.window_size == lags + differentiation