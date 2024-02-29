# Unit test set_lags ForecasterAutoregMultiVariate
# ==============================================================================
import pytest
import numpy as np
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from sklearn.linear_model import LinearRegression      


def test_set_lags_when_lags_argument_is_int():
    """
    Test how lags and max_lag attributes change when lags argument is integer
    positive (5).
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1', 
                                               lags=3, steps=2)
    forecaster.set_lags(lags=5)

    assert (forecaster.lags == np.array([1, 2, 3, 4, 5])).all()
    assert forecaster.max_lag == 5
    assert forecaster.window_size == 5


def test_set_lags_when_lags_argument_is_list():
    """
    Test how lags and max_lag attributes change when lags argument is a list
    of positive integers.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1', 
                                               lags=3, steps=2)
    forecaster.set_lags(lags=[1, 2, 3])

    assert (forecaster.lags == np.array([1, 2, 3])).all()
    assert forecaster.max_lag == 3
    assert forecaster.window_size == 3


def test_set_lags_when_lags_argument_is_1d_numpy_array():
    """
    Test how lags and max_lag attributes change when lags argument is 1d numpy
    array of positive integers.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1', 
                                               lags=3, steps=2)
    forecaster.set_lags(lags=np.array([1, 2, 3]))
    
    assert (forecaster.lags == np.array([1, 2, 3])).all()
    assert forecaster.max_lag == 3
    assert forecaster.window_size == 3


def test_set_lags_when_lags_argument_is_a_dict():
    """
    Test how lags and max_lag attributes change when lags argument is a dict.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1', 
                                               lags=3, steps=2)
    forecaster.set_lags(lags={'l1': 3, 'l2': [1, 5]})
    
    assert (forecaster.lags['l1'] == np.array([1, 2, 3])).all()
    assert (forecaster.lags['l2'] == np.array([1, 5])).all()
    assert forecaster.max_lag == 5
    assert forecaster.window_size == 5


def test_set_lags_when_lags_argument_is_a_dict_with_None():
    """
    Test how lags and max_lag attributes change when lags argument is a dict 
    containing None.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1', 
                                               lags=3, steps=2)
    forecaster.set_lags(lags={'l1': 3, 'l2': None})
    
    assert (forecaster.lags['l1'] == np.array([1, 2, 3])).all()
    assert isinstance(forecaster.lags['l2'], type(None))
    assert forecaster.max_lag == 3
    assert forecaster.window_size == 3


@pytest.mark.parametrize("lags, max_lag",
                         [(3, 3), 
                          ({'l1': 3, 'l2': 4}, 4), 
                          ({'l1': None, 'l2': 5}, 5)],
                         ids = lambda value : f'lags: {value}')
def test_set_lags_max_lag_stored(lags, max_lag):
    """
    Test max_lag is equal to the maximum lag.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1', 
                                               lags=1, steps=2)
    forecaster.set_lags(lags=lags)
    
    assert forecaster.max_lag == max_lag
    assert forecaster.window_size == max_lag