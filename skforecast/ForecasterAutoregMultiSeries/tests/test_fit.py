# Unit test fit ForecasterAutoregMultiSeries
# ==============================================================================
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


def test_forecaster_index_freq_stored():
    '''
    Test series.index.freqstr is stored in forecaster.index_freq.
    '''
    series = pd.DataFrame({'1': pd.Series(np.arange(5)), 
                           '2': pd.Series(np.arange(5))
                          })

    series.index = pd.date_range(start='2022-01-01', periods=5, freq='1D')

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series)
    expected = series.index.freqstr
    results = forecaster.index_freq

    assert results == expected


def test_fit_in_sample_residuals_stored():
    '''
    Test that values of in_sample_residuals are stored after fitting.
    '''
    series = pd.DataFrame({'1': pd.Series(np.arange(5)), 
                           '2': pd.Series(np.arange(5))
                           })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series)
    expected = {'1': np.array([0, 0]),
                '2': np.array([0, 0]),
               }
    results = forecaster.in_sample_residuals

    assert results.keys() == expected.keys()
    assert list(results.values())[0] == approx(list(expected.values())[0])
    assert list(results.values())[1] == approx(list(expected.values())[1])


def test_fit_in_sample_residuals_stored_XGBRegressor():
    '''
    Test that values of in_sample_residuals are stored after fitting with XGBRegressor.
    '''
    series = pd.DataFrame({'1': pd.Series(np.arange(5)), 
                           '2': pd.Series(np.arange(5))
                           })

    forecaster = ForecasterAutoregMultiSeries(XGBRegressor(random_state=123), lags=3)
    forecaster.fit(series=series)
    expected = {'1': np.array([-0.00049472,  0.00049543]),
                '2': np.array([-0.00049472,  0.00049543]),
               }
    results = forecaster.in_sample_residuals

    assert results.keys() == expected.keys()
    assert np.isclose(list(results.values())[0], list(expected.values())[0]).all()
    assert np.isclose(list(results.values())[1], list(expected.values())[1]).all()


def test_fit_same_residuals_when_residuals_greater_than_1000():
    '''
    Test fit return same residuals when residuals len is greater than 1000.
    Testing with two different forecaster.
    '''
    series = pd.DataFrame({'1': pd.Series(np.arange(520)), 
                           '2': pd.Series(np.arange(520))
                           })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series)
    results_1 = forecaster.in_sample_residuals
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series)
    results_2 = forecaster.in_sample_residuals

    assert results_1.keys() == results_2.keys()
    assert list(results_1.values())[0] == approx(list(results_2.values())[0])
    assert list(results_1.values())[1] == approx(list(results_2.values())[1])


def test_fit_last_window_stored():
    '''
    Test that values of last window are stored after fitting.
    '''
    series = pd.DataFrame({'1': pd.Series(np.arange(5)), 
                           '2': pd.Series(np.arange(5))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series)
    expected = pd.DataFrame({'1': pd.Series(np.array([2, 3, 4])), 
                             '2': pd.Series(np.array([2, 3, 4]))
                            })
    expected.index = pd.RangeIndex(start=2, stop=5, step=1)

    pd.testing.assert_frame_equal(forecaster.last_window, expected)