import sys
sys.path.insert(1, '/home/ximo/Documents/GitHub/skforecast')

import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression


def test_set_out_sample_residuals_exception_when_residuals_argument_is_list():
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    with pytest.raises(Exception):
        forecaster.set_out_sample_residuals(residuals=[1,2,3])
        
        
def test_set_out_sample_residuals_when_residuals_lenght_is_less_or_equal_to_1000():
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.set_out_sample_residuals(residuals=np.arange(10))
    assert (forecaster.out_sample_residuals == np.arange(10)).all()
    

def test_set_out_sample_residuals_when_residuals_lenght_is_greater_than_1000():
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.set_out_sample_residuals(residuals=np.arange(2000))
    assert len(forecaster.out_sample_residuals) == 1000