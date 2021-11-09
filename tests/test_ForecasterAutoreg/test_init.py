import sys
sys.path.insert(1, '/home/ximo/Documents/GitHub/skforecast')

import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression

def test_init_lags_when_integer_is_passed():
   
    forecaster = ForecasterAutoreg(LinearRegression(), lags=10)
    assert (forecaster.lags == np.arange(10) + 1).all()
    
def test_init_lags_when_list_is_passed():
   
    forecaster = ForecasterAutoreg(LinearRegression(), lags=[1, 2, 3])
    assert (forecaster.lags == np.array([1, 2, 3])).all()
    
def test_init_lags_when_range_is_passed():
   
    forecaster = ForecasterAutoreg(LinearRegression(), lags=range(1, 4))
    assert (forecaster.lags == np.array(range(1, 4))).all()
    
def test_init_lags_when_numpy_arange_is_passed():
   
    forecaster = ForecasterAutoreg(LinearRegression(), lags=np.arange(1, 10))
    assert (forecaster.lags == np.arange(1, 10)).all()

def test_init_exception_when_lags_is_int_less_than_1():
    
    with pytest.raises(Exception):
        ForecasterAutoreg(LinearRegression(), lags=-10)
        
def test_init_exception_when_lags_is_range_starting_at_zero():
    
    with pytest.raises(Exception):
        ForecasterAutoreg(LinearRegression(), lags=range(0, 4))
            
def test_init_exception_when_lags_is_numpy_arange_starting_at_zero():
    
    with pytest.raises(Exception):
        ForecasterAutoreg(LinearRegression(), lags=np.arange(0, 4))
        
def test_init_exception_when_lags_is_list_starting_at_zero():
    
    with pytest.raises(Exception):
        ForecasterAutoreg(LinearRegression(), lags=[0, 1, 2])