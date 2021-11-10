import sys
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterBase import ForecasterBase
from sklearn.linear_model import LinearRegression



def test_y_pd_series_exception_when_y_is_int():
    
    @ForecasterBase._y_pd_series
    def y_int(y: int):
        return

    with pytest.raises(Exception):
        y_int(10) 
    
        
def test_y_pd_series_exception_when_y_is_list():
    
    @ForecasterBase._y_pd_series
    def y_list(y: list):
        return

    with pytest.raises(Exception):
        y_list([1, 2, 3, 4]) 
        
        
def test_y_pd_series_exception_when_y_is_numpy_array_with_more_than_one_dimension():
    
    @ForecasterBase._y_pd_series
    def y_ndarray(y: np.ndarray):
        return

    with pytest.raises(Exception):
        y_ndarray(np.arange(10).reshape(-1, 1)) 