
import sys
sys.path.insert(1, '/home/ximo/Documents/GitHub/skforecast')

import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression

# Test method _create_lags()
#-------------------------------------------------------------------------------
def test_create_lags_when_lags_is_3_and_y_is_numpy_arange_10():
    '''
    Check matrix of lags is created properly
    '''
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
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
    


def test_create_lags_exception_when_len_of_y_is_less_than_maximum_lag():
   
    forecaster = ForecasterAutoreg(LinearRegression(), lags=10)
    with pytest.raises(Exception):
        forecaster._create_lags(y=np.arange(5))