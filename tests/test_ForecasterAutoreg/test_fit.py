import sys
sys.path.insert(1, '/home/ximo/Documents/GitHub/skforecast')

import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression

        
def test_fit_exception_when_y_and_exog_have_different_lenght():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=5)
    with pytest.raises(Exception):
        forecaster.fit(y=np.arange(50), exog=np.arange(10))
    with pytest.raises(Exception):
        forecaster.fit(y=np.arange(50), exog=pd.Series(np.arange(10)))


def test_last_window_stored_when_fit_forecaster():
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(50))
    assert (forecaster.last_window == np.array([47, 48, 49])).all()
    
    
def test_in_sample_residuals_stored_when_fit_forecaster():
    
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(5))
    expected = np.array([0, 0])
    results = forecaster.in_sample_residuals  
    assert results == approx(expected)