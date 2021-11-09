import sys
sys.path.insert(1, '/home/ximo/Documents/GitHub/skforecast')

import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression


def test_estimate_boot_interval_exception_when_steps_argument_is_less_than_1():
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(10))
    with pytest.raises(Exception):
        forecaster._estimate_boot_interval(steps=0)
        
        
def test_estimate_boot_interval_exception_when_in_sample_residuals_argument_is_False_and_out_sample_residuals_attribute_is_empty():
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(10))
    with pytest.raises(Exception):
        forecaster._estimate_boot_interval(steps=5, in_sample_residuals=False)
        
        
def test_estimate_boot_interval_exception_when_forecaster_fitted_with_exog_but_exog_argument_is_None():
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(10), exog=np.arange(10))
    with pytest.raises(Exception):
        forecaster._estimate_boot_interval(steps=5)
        
        
def test_estimate_boot_interval_exception_when_forecaster_fitted_without_exog_but_exog_argument_is_not_None():
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(10))
    with pytest.raises(Exception):
        forecaster._estimate_boot_interval(steps=5, exog=np.arange(10))
        
        
def test_estimate_boot_interval_exception_when_lenght_exog_argument_is_less_than_steps():
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(10), exog=np.arange(10))
    with pytest.raises(Exception):
        forecaster._estimate_boot_interval(steps=5, exog=np.arange(3))
        

def test_estimate_boot_interval_exception_when_lenght_last_window_argument_is_less_than_max_lag_attribute():
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(10))
    with pytest.raises(Exception):
        forecaster._estimate_boot_interval(steps=5, last_window=np.array([1,2]))
        
        
def test_estimate_boot_interval_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_True():
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(10))
    forecaster.in_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=10)
    expected = np.array([[20., 20.]])
    results = forecaster._estimate_boot_interval(steps=1, in_sample_residuals=True)  
    assert results == approx(expected)
    
    
def test_estimate_boot_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_True():
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(10))
    forecaster.in_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=10)
    expected = np.array([[20.        , 20.        ],
                        [24.33333333, 24.33333333]])
    results = forecaster._estimate_boot_interval(steps=2, in_sample_residuals=True)  
    assert results == approx(expected)
    
    
def test_estimate_boot_interval_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_False():
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(10))
    forecaster.out_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=10)
    expected = np.array([[20., 20.]])
    results = forecaster._estimate_boot_interval(steps=1, in_sample_residuals=False)  
    assert results == approx(expected)
    
    
def test_estimate_boot_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_False():
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(10))
    forecaster.out_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=10)
    expected = np.array([[20.        , 20.        ],
                        [24.33333333, 24.33333333]])
    results = forecaster._estimate_boot_interval(steps=2, in_sample_residuals=False)  
    assert results == approx(expected)