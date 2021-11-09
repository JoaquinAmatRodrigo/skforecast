import sys
sys.path.insert(1, '/home/ximo/Documents/GitHub/skforecast')

import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

def test_get_feature_importances_when_regressor_is_RandomForest():
    forecaster = ForecasterAutoreg(RandomForestRegressor(n_estimators=1, max_depth=2, random_state=123), lags=3)
    forecaster.fit(y=np.arange(10))
    expected = np.array([0.94766355, 0., 0.05233645])
    assert forecaster.get_feature_importances() == approx(expected)
    
def test_get_feature_importances_when_regressor_is_GradientBoostingRegressor():
    forecaster = ForecasterAutoreg(GradientBoostingRegressor(n_estimators=1, max_depth=2, random_state=123), lags=3)
    forecaster.fit(y=np.arange(10))
    expected = np.array([0.1509434 , 0.05660377, 0.79245283])
    assert forecaster.get_feature_importances() == approx(expected)
    
def test_get_feature_importances_when_regressor_is_linear_model():
    forecaster = ForecasterAutoreg(Lasso(), lags=3)
    forecaster.fit(y=np.arange(50))
    expected = None
    assert forecaster.get_feature_importances() is None
    