# Unit test set_lags ForecasterAutoregMultiSeries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.linear_model import LinearRegression


def test_set_lags_exception_when_lags_argument_is_int_lower_than_1():
    """
    Test exception is raised when lags argument is lower than 1.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    err_msg = re.escape('Minimum value of lags allowed is 1.')
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_lags(lags=-10)


def test_set_lags_exception_when_lags_argument_has_any_value_lower_than_1():
    """
    Test exception is raised when lags argument has at least one value
    lower than 1.
    """
    
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    err_msg = re.escape('Minimum value of lags allowed is 1.')
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_lags(lags=range(0, 4))  


def test_set_lags_exception_when_lags_argument_is_not_valid_type():
    """
    Test exception is raised when lags argument is not a valid type.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    lags = 'not_valid_type'

    err_msg = re.escape(
                f"`lags` argument must be `int`, `1D np.ndarray`, `range` or `list`. "
                f"Got {type(lags)}"
            )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_lags(lags=lags)         


def test_set_lags_when_lags_argument_is_int():
    """
    Test how lags and max_lag attributes change when lags argument is integer
    positive (5).
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.set_lags(lags=5)

    assert (forecaster.lags == np.array([1, 2, 3, 4, 5])).all()
    assert forecaster.max_lag == 5


def test_set_lags_when_lags_argument_is_list():
    """
    Test how lags and max_lag attributes change when lags argument is a list
    of positive integers.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.set_lags(lags=[1,2,3])

    assert (forecaster.lags == np.array([1, 2, 3])).all()
    assert forecaster.max_lag == 3


def test_set_lags_when_lags_argument_is_1d_numpy_array():
    """
    Test how lags and max_lag attributes change when lags argument is 1d numpy
    array of positive integers.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.set_lags(lags=np.array([1,2,3]))
    
    assert (forecaster.lags == np.array([1, 2, 3])).all()
    assert forecaster.max_lag == 3