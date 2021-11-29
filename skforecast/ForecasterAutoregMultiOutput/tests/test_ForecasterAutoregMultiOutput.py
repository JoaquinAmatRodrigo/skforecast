import sys
sys.path.insert(1, '/home/ximo/Documents/GitHub/skforecast')

import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast import __version__
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor




        
# Test method fit()
#-------------------------------------------------------------------------------
def test_fit_exception_when_y_and_exog_have_different_lenght():

    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=5, steps=2)
    with pytest.raises(Exception):
        forecaster.fit(y=np.arange(50), exog=np.arange(10))
    with pytest.raises(Exception):
        forecaster.fit(y=np.arange(50), exog=pd.Series(np.arange(10)))


def test_last_window_stored_when_fit_forecaster():
    
    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=3, steps=2)
    forecaster.fit(y=np.arange(50))
    assert (forecaster.last_window == np.array([47, 48, 49])).all()
    
    
# Test method predict()
#-------------------------------------------------------------------------------
def test_predict_exception_when_forecaster_fited_without_exog_and_exog_passed_when_predict():

    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=3, steps=2)
    forecaster.fit(y=np.arange(50))
    with pytest.raises(Exception):
        forecaster.predict(exog=np.arange(10))


def test_predict_exception_when_forecaster_fited_with_exog_but_not_exog_passed_when_predict():

    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=3, steps=2)
    forecaster.fit(y=np.arange(50), exog=np.arange(50))
    with pytest.raises(Exception):
        forecaster.predict()
        
        
def test_predict_exception_when_exog_lenght_is_less_than_steps():

    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=3, steps=10)
    forecaster.fit(y=np.arange(50), exog=np.arange(50))
    with pytest.raises(Exception):
        forecaster.predict(exog=np.arange(5))
        
        
def test_predict_exception_when_last_window_argument_is_not_numpy_array_or_pandas_series():

    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=3, steps=2)
    forecaster.fit(y=np.arange(50))
    with pytest.raises(Exception):
        forecaster.predict(last_window=[1,2,3])


def test_predict_exception_when_last_window_lenght_is_less_than_maximum_lag():

    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=3, steps=2)
    forecaster.fit(y=np.arange(50))
    with pytest.raises(Exception):
        forecaster.predict(last_window=pd.Series([1, 2]))
        

def test_predict_output_when_regresor_is_LinearRegression_lags_is_3_steps_5_ytrain_is_numpy_arange_50():

    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=3, steps=5)
    forecaster.fit(y=np.arange(50))
    predictions = forecaster.predict()
    expected = np.array([50., 51., 52., 53., 54.])
    assert predictions == approx(expected)


def test_predict_output_when_regresor_is_LinearRegression_lags_is_5_steps_2_ytrain_is_numpy_arange_50():

    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=5, steps=2)
    forecaster.fit(y=np.arange(50))
    predictions = forecaster.predict()
    expected = np.array([50., 51.])
    assert predictions == approx(expected)


def test_predict_exception_when_exog_passed_in_predict_has_different_columns_than_exog_used_to_fit_nparray():

    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=5, steps=2)
    forecaster.fit(y=np.arange(10), exog=np.arange(30).reshape(-1, 3))
    with pytest.raises(Exception):
        forecaster.predict(steps=10, exog=np.arange(30).reshape(-1, 2))

def test_predict_exception_when_exog_passed_in_predict_has_different_columns_than_exog_used_to_fit_pdDataDrame():

    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=5, steps=2)
    forecaster.fit(y=np.arange(10), exog=pd.DataFrame(np.arange(30).reshape(-1, 3)))
    with pytest.raises(Exception):
        forecaster.predict(steps=10, exog=pd.DataFrame(np.arange(30).reshape(-1, 2)))
    




        

    
    
# Test method test_set_paramns()
#-------------------------------------------------------------------------------
def test_set_paramns():
    
    forecaster = ForecasterAutoregMultiOutput(LinearRegression(fit_intercept=True), lags=3, steps=2)
    new_paramns = {'fit_intercept': False}
    forecaster.set_params(**new_paramns)
    expected = {'copy_X': True,
                 'fit_intercept': False,
                 'n_jobs': None,
                 'normalize': False,
                 'positive': False
                }
    assert forecaster.regressor.get_params() == expected


# Test method test_set_lags()
#-------------------------------------------------------------------------------
def test_set_lags_excepion_when_lags_argument_is_int_less_than_1():
    
    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=3, steps=2)
    with pytest.raises(Exception):
        ForecasterAutoreg(LinearRegression(), lags=-10)

def test_set_lags_excepion_when_lags_argument_has_any_value_less_than_1():
    
    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=3, steps=2)
    with pytest.raises(Exception):
        ForecasterAutoreg(LinearRegression(), lags=range(0, 4))
        
        
def test_set_lags_when_lags_argument_is_int():
    
    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=3, steps=2)
    forecaster.set_lags(lags=5)
    assert (forecaster.lags == np.array([1, 2, 3, 4, 5])).all()
    assert forecaster.max_lag == 5
    
def test_set_lags_when_lags_argument_is_list():
    
    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=3, steps=2)
    forecaster.set_lags(lags=[1,2,3])
    assert (forecaster.lags == np.array([1, 2, 3])).all()
    assert forecaster.max_lag == 3
    
def test_set_lags_when_lags_argument_is_1d_numpy_array():
    
    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=3, steps=2)
    forecaster.set_lags(lags=np.array([1,2,3]))
    assert (forecaster.lags == np.array([1, 2, 3])).all()
    assert forecaster.max_lag == 3
        
    
# Test method get_coef()
#-------------------------------------------------------------------------------
def test_get_coef_when_regressor_is_LinearRegression_steps_1_lags_3():
    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=3, steps=1)
    forecaster.fit(y=np.arange(5))
    expected = np.array([0.33333333, 0.33333333, 0.33333333])
    assert forecaster.get_coef(step=1) == approx(expected)

def test_get_coef_when_regressor_is_LinearRegression_steps_2_lags_3():
    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=3, steps=2)
    forecaster.fit(y=np.arange(10))
    expected = np.array([0.33333333, 0.33333333, 0.33333333])
    assert forecaster.get_coef(step=1) == approx(expected)
    assert forecaster.get_coef(step=2) == approx(expected)
    
def test_get_coef_when_regressor_is_RandomForest():
    forecaster = ForecasterAutoregMultiOutput(RandomForestRegressor(n_estimators=1, max_depth=2), lags=3, steps=2)
    forecaster.fit(y=np.arange(5))
    expected = None
    assert forecaster.get_coef(step=1) is None