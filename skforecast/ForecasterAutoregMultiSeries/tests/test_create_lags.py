# Unit test _create_lags ForecasterAutoregMultiSeries
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.linear_model import LinearRegression


def test_create_lags_output():
    '''
    Test matrix of lags is created properly when langs=3 and y=np.arange(10).
    '''
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    results = forecaster._create_lags(y=np.arange(10))
    expected = (np.array([[2., 1., 0.],
                          [3., 2., 1.],
                          [4., 3., 2.],
                          [5., 4., 3.],
                          [6., 5., 4.],
                          [7., 6., 5.],
                          [8., 7., 6.]]),
               np.array([3., 4., 5., 6., 7., 8., 9.]))

    assert (results[0] == expected[0]).all()
    assert (results[1] == expected[1]).all()
    
    
def test_create_lags_exception_when_len_of_y_is_lower_than_maximum_lag():
    '''
    Test exception is raised when length of y is lower than maximum lag included
    in the forecaster.
    '''
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=10)
    with pytest.raises(Exception):
        forecaster._create_lags(y=np.arange(5))