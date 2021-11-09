import sys
sys.path.insert(1, '/home/ximo/Documents/GitHub/skforecast')

import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression


def test_predict_exception_when_steps_lower_than_1():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(50))
    with pytest.raises(Exception):
        forecaster.predict(steps=0)

def test_predict_exception_when_forecaster_fited_without_exog_and_exog_passed_when_predict():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(50))
    with pytest.raises(Exception):
        forecaster.predict(steps=10, exog=np.arange(10))


def test_predict_exception_when_forecaster_fited_with_exog_but_not_exog_passed_when_predict():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(50), exog=np.arange(50))
    with pytest.raises(Exception):
        forecaster.predict(steps=10)


def test_predict_exception_when_exog_passed_in_predict_has_different_columns_than_exog_used_to_fit_nparray():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(10), exog=np.arange(30).reshape(-1, 3))
    with pytest.raises(Exception):
        forecaster.predict(steps=10, exog=np.arange(30).reshape(-1, 2))

def test_predict_exception_when_exog_passed_in_predict_has_different_columns_than_exog_used_to_fit_pdDataDrame():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(10), exog=pd.DataFrame(np.arange(30).reshape(-1, 3)))
    with pytest.raises(Exception):
        forecaster.predict(steps=10, exog=pd.DataFrame(np.arange(30).reshape(-1, 2)))
        
        
def test_predict_exception_when_exog_lenght_is_less_than_steps():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(50), exog=np.arange(50))
    with pytest.raises(Exception):
        forecaster.predict(steps=10, exog=np.arange(5))
        
        
def test_predict_exception_when_last_window_argument_is_not_numpy_array_or_pandas_series():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(50))
    with pytest.raises(Exception):
        forecaster.predict(steps=10, last_window=[1,2,3])


def test_predict_exception_when_last_window_lenght_is_less_than_maximum_lag():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(50))
    with pytest.raises(Exception):
        forecaster.predict(steps=10, last_window=pd.Series([1, 2]))
        

def test_predict_output_when_regresor_is_LinearRegression_lags_is_3_ytrain_is_numpy_arange_50_and_steps_is_5():

    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=np.arange(50))
    predictions = forecaster.predict(steps=5)
    expected = np.array([50., 51., 52., 53., 54.])
    assert predictions == approx(expected)