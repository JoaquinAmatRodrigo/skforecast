# Unit test fit ForecasterAutoreg
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Fixtures
from .fixtures_ForecasterAutoreg import y

def custom_weights(index): # pragma: no cover
    """
    Return 0 if index is one of '2022-01-05', '2022-01-06', 1 otherwise.
    """
    weights = np.where(
                (index >= 20) & (index <= 40),
                0,
                1
              )
    
    return weights


def test_forecaster_DatetimeIndex_index_freq_stored():
    """
    Test serie_with_DatetimeIndex.index.freqstr is stored in forecaster.index_freq.
    """
    serie_with_DatetimeIndex = pd.Series(
        data  = [1, 2, 3, 4, 5],
        index = pd.date_range(start='2022-01-01', periods=5)
    )
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=serie_with_DatetimeIndex)
    expected = serie_with_DatetimeIndex.index.freqstr
    results = forecaster.index_freq

    assert results == expected


def test_forecaster_index_step_stored():
    """
    Test serie without DatetimeIndex, step is stored in forecaster.index_freq.
    """
    y = pd.Series(data=np.arange(10))
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=y)
    expected = y.index.step
    results = forecaster.index_freq

    assert results == expected
    
    
def test_fit_in_sample_residuals_stored():
    """
    Test that values of in_sample_residuals are stored after fitting.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(5)))
    results = forecaster.in_sample_residuals
    expected = np.array([0., 0.])

    assert isinstance(results, np.ndarray)
    np.testing.assert_array_equal(results, expected)


def test_fit_in_sample_residuals_stored_XGBRegressor():
    """
    Test that values of in_sample_residuals are stored after fitting with XGBRegressor.
    """
    forecaster = ForecasterAutoreg(XGBRegressor(random_state=123), lags=3)
    forecaster.fit(y=pd.Series(np.arange(5)))
    results = forecaster.in_sample_residuals
    expected = np.array([-0.0008831, 0.00088406])

    assert isinstance(results, np.ndarray)
    assert all(np.isclose(results, expected))


def test_fit_same_residuals_when_residuals_greater_than_1000():
    """
    Test fit return same residuals when residuals len is greater than 1000.
    Testing with two different forecaster.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(1200)))
    results_1 = forecaster.in_sample_residuals
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(1200)))
    results_2 = forecaster.in_sample_residuals
    
    assert isinstance(results_1, np.ndarray)
    assert isinstance(results_2, np.ndarray)
    np.testing.assert_array_equal(results_1, results_2)


def test_fit_in_sample_residuals_not_stored():
    """
    Test that values of in_sample_residuals are not stored after fitting
    when `store_in_sample_residuals=False`.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(5)), store_in_sample_residuals=False)
    results = forecaster.in_sample_residuals

    assert results is None


def test_fit_last_window_stored():
    """
    Test that values of last window are stored after fitting.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(50)))
    expected = pd.Series(np.array([47, 48, 49]), index=[47, 48, 49])

    pd.testing.assert_series_equal(forecaster.last_window, expected)


def test_fit_model_coef_when_using_weight_func():
    """
    Check the value of the regressor coefs when using a `weight_func`.
    """
    forecaster = ForecasterAutoreg(
                     regressor   = LinearRegression(),
                     lags        = 5,
                     weight_func = custom_weights
                 )
    forecaster.fit(y=y)
    results = forecaster.regressor.coef_
    expected = np.array([0.01211677, -0.20981367,  0.04214442, -0.0369663, -0.18796105])

    np.testing.assert_almost_equal(results, expected)


def test_fit_model_coef_when_not_using_weight_func():
    """
    Check the value of the regressor coefs when not using a `weight_func`.
    """
    forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=5)
    forecaster.fit(y=y)
    results = forecaster.regressor.coef_
    expected = np.array([0.16773502, -0.09712939,  0.10046413, -0.09971515, -0.15849756])

    np.testing.assert_almost_equal(results, expected)