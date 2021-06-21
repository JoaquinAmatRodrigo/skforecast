import pytest
from pytest import approx
import numpy as np
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import time_series_spliter
from skforecast.model_selection import cv_forecaster

# Test test_time_series_spliter
#-------------------------------------------------------------------------------
def test_time_series_spliter_when_y_is_numpy_array_with_more_than_1_dimesion():

    results = time_series_spliter(np.arange(10).reshape(-1, 2), initial_train_size=3, steps=1)
    with pytest.raises(Exception):
       list(results)
       
def test_time_series_spliter_when_y_is_list():

    results = time_series_spliter([0,1,2,3,4], initial_train_size=3, steps=1)
    with pytest.raises(Exception):
       list(results)
    
def test_time_series_spliter_when_y_is_numpy_arange_5_initial_train_size_3_steps_1_allow_incomplete_fold_True():

    results  = time_series_spliter(np.arange(5), initial_train_size=3, steps=1, verbose=False)
    results  = list(results)
    expected = [(range(0, 3), range(3, 4)), (range(0, 4), range(4, 5))]
    assert results == expected
    
def test_time_series_spliter_when_y_is_numpy_arange_5_initial_train_size_3_steps_2_allow_incomplete_fold_True():

    results  = time_series_spliter(np.arange(5), initial_train_size=3, steps=2)
    results  = list(results)
    expected = [(range(0, 3), range(3, 5))]
    assert results == expected
    
def test_time_series_spliter_when_y_is_numpy_arange_5_initial_train_size_3_steps_3_allow_incomplete_fold_True():

    results  = time_series_spliter(np.arange(5), initial_train_size=3, steps=3)
    results  = list(results)
    expected = [(range(0, 3), range(3, 5))]
    assert results == expected
