# Unit test __init__ ForecasterAutoregMultiSeries
# ==============================================================================
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.linear_model import LinearRegression


@pytest.mark.parametrize("lags            , expected", 
                         [(10              , np.arange(10) + 1), 
                          ([1, 2, 3]       , np.array([1, 2, 3])), 
                          (range(1, 4)     , np.array(range(1, 4))), 
                          (np.arange(1, 10), np.arange(1, 10))], 
                         ids = lambda values : f'values: {values}'
                        )
def test_init_input_lags_parameter(lags, expected):
    """
    Test creation of attribute lags with different arguments.
    This test could replace the next 4 tests.
    """
    # Example how to skip one part of the test
    if isinstance(lags, int): 
        pytest.skip('Not of interest.')
    
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=lags)

    assert (forecaster.lags == expected).all()


def test_init_lags_when_integer():
    '''
    Test creation of attribute lags when integer is passed.
    '''
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=10)

    assert (forecaster.lags == np.arange(10) + 1).all()
    

def test_init_lags_when_list():
    '''
    Test creation of attribute lags when list is passed.
    '''
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=[1, 2, 3])

    assert (forecaster.lags == np.array([1, 2, 3])).all()


def test_init_lags_when_range():
    '''
    Test creation of attribute lags when range is passed.
    '''
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=range(1, 4))

    assert (forecaster.lags == np.array(range(1, 4))).all()


def test_init_lags_when_numpy_arange():
    '''
    Test creation of attribute lags when numpy arange is passed.
    '''
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=np.arange(1, 10))
    
    assert (forecaster.lags == np.arange(1, 10)).all()


def test_init_exception_when_lags_is_int_lower_than_1():
    '''
    Test exception is raised when lags is initialized with int lower than 1.
    '''
    with pytest.raises(Exception):
        ForecasterAutoregMultiSeries(LinearRegression(), lags=-10)


def test_init_exception_when_lags_has_values_lower_than_1():
    '''
    Test exception is raised when lags is initialized with any value lower than 1.
    '''
    for lags in [[0, 1], range(0, 2), np.arange(0, 2)]:
        with pytest.raises(Exception):
            ForecasterAutoregMultiSeries(LinearRegression(), lags=lags)


def test_init_exception_when_lags_list_or_numpy_array_with_values_not_int():
    '''
    Test exception is raised when lags is list or numpy array and element(s) are not int.
    '''
    lags_list = [1, 1.5, [1, 2], range(5)]
    lags_np_array = np.array([1.2, 1.5])
    
    for lags in [lags_list, lags_np_array]:
        for lag in lags:
            if not isinstance(lag, (int, np.int64, np.int32)):
                with pytest.raises(Exception):
                    ForecasterAutoregMultiSeries(LinearRegression(), lags=lags)


def test_init_exception_when_lags_is_not_valid_type():
    '''
    Test exception is raised when lags is not a valid type.
    '''
    with pytest.raises(Exception):
        ForecasterAutoregMultiSeries(LinearRegression(), lags='not_valid_type')