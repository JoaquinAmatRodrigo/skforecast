import sys
sys.path.insert(1, '/home/ximo/Documents/GitHub/skforecast')

import pytest
from pytest import approx
import numpy as np
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_True():
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(10))
    forecaster.in_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=10)
    expected = np.array([[10., 20., 20.]])
    results = forecaster.predict_interval(steps=1, in_sample_residuals=True)  
    assert results == approx(expected)
    
    
def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_True():
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(10))
    forecaster.in_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=10)
    expected = np.array([[10.        , 20.        , 20.        ],
                         [11.        , 24.33333333, 24.33333333]])
    results = forecaster.predict_interval(steps=2, in_sample_residuals=True)  
    assert results == approx(expected)
    
    
def testpredict_interval_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_False():
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(10))
    forecaster.out_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=10)
    expected = np.array([[10., 20., 20.]])
    results = forecaster.predict_interval(steps=1, in_sample_residuals=False)  
    assert results == approx(expected)
    
    
def testpredict_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_False():
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(10))
    forecaster.out_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=10)
    expected = np.array([[10.        , 20.        , 20.        ],
                         [11.        , 24.33333333, 24.33333333]])
    results = forecaster.predict_interval(steps=2, in_sample_residuals=False)  
    assert results == approx(expected)