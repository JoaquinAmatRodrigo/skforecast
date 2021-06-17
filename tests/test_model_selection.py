import pytest
from pytest import approx
import numpy as np
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import time_series_spliter
from skforecast.model_selection import cv_forecaster

def test_time_series_spliter():
    '''
    check results of time_series_spliter()
    '''
    results = time_series_spliter(np.arange(5), initial_train_size=3, steps=1)
    
    assert(list(results) == [(range(0, 3), range(3, 4)), (range(0, 4), range(4, 5))])
    
    
def test_time_series_spliter_exceptions():
    '''
    Check exceptions for time_series_spliter().
    '''
    
    with pytest.raises(Exception):
        list(time_series_spliter([1, 2, 3], initial_train_size=3, steps=1))

    with pytest.raises(Exception):
        list(time_series_spliter(y=100, initial_train_size=3, steps=1))
        
    with pytest.raises(Exception):
        list(time_series_spliter(
                y=np.arange(10).reshape(5,2),
                initial_train_size=3,
                steps=1)
        )
