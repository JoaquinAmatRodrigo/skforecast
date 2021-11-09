import sys
sys.path.insert(1, '/home/ximo/Documents/GitHub/skforecast')

import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor



def test_get_coef_when_regressor_is_LinearRegression():
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(5))
    expected = np.array([0.33333333, 0.33333333, 0.33333333])
    assert forecaster.get_coef() == approx(expected)
    
def test_get_coef_when_regressor_is_Ridge():
    forecaster = ForecasterAutoreg(Ridge(), lags=3)
    forecaster.fit(y=np.arange(5))
    expected = np.array([0.2, 0.2, 0.2])
    assert forecaster.get_coef() == approx(expected)
    
def test_get_coef_when_regressor_is_Lasso():
    forecaster = ForecasterAutoreg(Lasso(), lags=3)
    forecaster.fit(y=np.arange(50))
    expected = np.array([9.94565217e-01, 6.16219995e-17, 0.00000000e+00])
    assert forecaster.get_coef() == approx(expected)


def test_get_coef_when_regressor_is_RandomForest():
    forecaster = ForecasterAutoreg(RandomForestRegressor(n_estimators=1, max_depth=2), lags=3)
    forecaster.fit(y=np.arange(5))
    expected = None
    assert forecaster.get_coef() is None