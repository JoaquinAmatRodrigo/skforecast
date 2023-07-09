# Unit test fit ForecasterAutoregDirect
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


def test_forecaster_DatetimeIndex_index_freq_stored():
    """
    Test serie_with_DatetimeIndex.index.freqstr is stored in forecaster.index_freq.
    """
    serie_with_DatetimeIndex = pd.Series(
        data  = np.arange(10),
        index = pd.date_range(start='2022-01-01', periods=10)
    )
    forecaster = ForecasterAutoregDirect(XGBRegressor(random_state=123), lags=3, steps=2)
    forecaster.fit(y=serie_with_DatetimeIndex)
    expected = serie_with_DatetimeIndex.index.freqstr
    results = forecaster.index_freq

    assert results == expected


def test_forecaster_index_step_stored():
    """
    Test serie without DatetimeIndex, step is stored in forecaster.index_freq.
    """
    y = pd.Series(data=np.arange(10))
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=2)
    forecaster.fit(y=y)
    expected = y.index.step
    results = forecaster.index_freq

    assert results == expected
    

@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'], 
                         ids=lambda n_jobs: f'n_jobs: {n_jobs}')
def test_fit_in_sample_residuals_stored(n_jobs):
    """
    Test that values of in_sample_residuals are stored after fitting.
    """
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=2, n_jobs=n_jobs)
    forecaster.fit(y=pd.Series(np.arange(5)))
    expected = {1: np.array([0.]),
                2: np.array([0.])}
    results = forecaster.in_sample_residuals

    assert isinstance(results, dict)
    assert all(isinstance(x, np.ndarray) for x in results.values())
    assert results.keys() == expected.keys()
    assert all(all(np.isclose(results[k], expected[k])) for k in expected.keys())


@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'], 
                         ids=lambda n_jobs: f'n_jobs: {n_jobs}')
def test_fit_in_sample_residuals_stored_XGBRegressor(n_jobs):
    """
    Test that values of in_sample_residuals are stored after fitting with XGBRegressor.
    """
    forecaster = ForecasterAutoregDirect(XGBRegressor(random_state=123), lags=3, steps=2, n_jobs=n_jobs)
    forecaster.fit(y=pd.Series(np.arange(5)))
    expected = {1: np.array([7.15255737e-07]),
                2: np.array([7.15255737e-07])}
    results = forecaster.in_sample_residuals

    assert isinstance(results, dict)
    assert all(isinstance(x, np.ndarray) for x in results.values())
    assert results.keys() == expected.keys()
    assert all(all(np.isclose(results[k], expected[k])) for k in expected.keys())


@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'], 
                         ids=lambda n_jobs: f'n_jobs: {n_jobs}')
def test_fit_same_residuals_when_residuals_greater_than_1000(n_jobs):
    """
    Test fit return same residuals when residuals len is greater than 1000.
    Testing with two different forecaster.
    """
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=2, n_jobs=n_jobs)
    forecaster.fit(y=pd.Series(np.arange(1200)))
    results_1 = forecaster.in_sample_residuals

    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=2, n_jobs=n_jobs)
    forecaster.fit(y=pd.Series(np.arange(1200)))
    results_2 = forecaster.in_sample_residuals

    assert isinstance(results_1, dict)
    assert all(isinstance(x, np.ndarray) for x in results_1.values())
    assert isinstance(results_2, dict)
    assert all(isinstance(x, np.ndarray) for x in results_2.values())
    assert results_1.keys() == results_2.keys()
    assert all(len(results_1[k] == 1000) for k in results_1.keys())
    assert all(len(results_2[k] == 1000) for k in results_2.keys())
    assert all(all(results_1[k] == results_2[k]) for k in results_2.keys())


@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'], 
                         ids=lambda n_jobs: f'n_jobs: {n_jobs}')
def test_fit_in_sample_residuals_not_stored(n_jobs):
    """
    Test that values of in_sample_residuals are not stored after fitting
    when `store_in_sample_residuals=False`.
    """
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=2, n_jobs=n_jobs)
    forecaster.fit(y=pd.Series(np.arange(5)), store_in_sample_residuals=False)
    expected = {1: None, 2: None}
    results = forecaster.in_sample_residuals

    assert isinstance(results, dict)
    assert results.keys() == expected.keys()
    assert all(results[k] == expected[k] for k in expected.keys())


def test_fit_last_window_stored():
    """
    Test that values of last window are stored after fitting.
    """    
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=2)
    forecaster.fit(y=pd.Series(np.arange(50)))
    expected = pd.Series(np.array([47, 48, 49]), index=[47, 48, 49])
    results = forecaster.last_window

    pd.testing.assert_series_equal(expected, results)