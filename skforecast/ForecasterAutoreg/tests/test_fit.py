# Unit test fit ForecasterAutoreg
# ==============================================================================
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


def test_forecaster_index_freq_stored():
    '''
    Test serie_with_DatetimeIndex.index.freqstr is stored in forecaster.index_freq.
    '''
    serie_with_DatetimeIndex = pd.Series(
                data  = [1, 2, 3, 4, 5],
                index = pd.date_range(start='2022-01-01', periods=5)
                )
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=serie_with_DatetimeIndex)
    expected = serie_with_DatetimeIndex.index.freqstr
    results = forecaster.index_freq
    assert results == expected
    
    
def test_fit_in_sample_residuals_stored():
    '''
    Test that values of in_sample_residuals are stored after fitting.
    '''
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(5)))
    expected = np.array([0, 0])
    results = forecaster.in_sample_residuals
    assert results.values == approx(expected)


def test_fit_in_sample_residuals_stored_XGBRegressor():
    '''
    Test that values of in_sample_residuals are stored after fitting with XGBRegressor.
    '''
    forecaster = ForecasterAutoreg(XGBRegressor(random_state=123), lags=3)
    forecaster.fit(y=pd.Series(np.arange(5)))
    expected = np.array([-0.0008831, 0.00088406])
    results = forecaster.in_sample_residuals.to_numpy()
    assert np.isclose(expected, results).all()


def test_fit_same_residuals_when_residuals_greater_than_1000():
    '''
    Test fit return same residuals when residuals len is greater than 1000.
    Testing with two different forecaster.
    '''
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(1200)))
    results_1 = forecaster.in_sample_residuals
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(1200)))
    results_2 = forecaster.in_sample_residuals
    assert (results_1 == results_2).all()


def test_fit_last_window_stored():
    '''
    Test that values of last window are stored after fitting.
    '''
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(50)))
    expected = pd.Series(np.array([47, 48, 49]), index=[47, 48, 49])
    assert (forecaster.last_window == expected).all()