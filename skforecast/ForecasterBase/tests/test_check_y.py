import sys
sys.path.insert(1, '/home/ximo/Documents/GitHub/skforecast')

import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression



def test_check_y_exception_when_y_is_int():
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    with pytest.raises(Exception):
        forecaster._check_y(y=10)
        
def test_check_y_exception_when_y_is_list():
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    with pytest.raises(Exception):
        forecaster._check_y(y=[1, 2, 3])
        
        
def test_check_y_exception_when_y_is_numpy_array_with_more_than_one_dimension():
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    with pytest.raises(Exception):
        forecaster._check_y(y=np.arange(10).reshape(-1, 1))