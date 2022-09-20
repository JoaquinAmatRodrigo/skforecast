# Unit test fit ForecasterAutoregMultiSeries
# ==============================================================================
import re
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


def test_fit_exception_when_levels_of_transformer_series_not_equal_to_series_levels():
    """
    Test exception is raised when `transformer_series` is a dict and its keys are
    not the same as forecaster.series_levels.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)), 
                           '2': pd.Series(np.arange(5))
                           })

    dict_transformers = {'1': StandardScaler(), 
                         '3': StandardScaler()
                        }
    forecaster = ForecasterAutoregMultiSeries(regressor          = LinearRegression(), 
                                              lags               = 3,
                                              transformer_series = dict_transformers)
    forecaster.series_levels = ['1', '2']
    
    err_msg = re.escape(
                (f'When `transformer_series` parameter is a `dict`, its keys '
                 f'must be the same as `series_levels` : {forecaster.series_levels}')
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.fit(series=series)


def test_forecaster_index_freq_stored():
    """
    Test series.index.freqstr is stored in forecaster.index_freq.
    """
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
    """
    Test that values of in_sample_residuals are stored after fitting
    when `store_in_sample_residuals=True`.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)), 
                           '2': pd.Series(np.arange(5))
                           })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=True)
    expected = {'1': np.array([0, 0]),
                '2': np.array([0, 0]),
               }
    results = forecaster.in_sample_residuals

    assert results.keys() == expected.keys()
    assert list(results.values())[0] == approx(list(expected.values())[0])
    assert list(results.values())[1] == approx(list(expected.values())[1])


def test_fit_in_sample_residuals_stored_XGBRegressor():
    """
    Test that values of in_sample_residuals are stored after fitting with XGBRegressor.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)), 
                           '2': pd.Series(np.arange(5))
                           })

    forecaster = ForecasterAutoregMultiSeries(XGBRegressor(random_state=123), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=True)
    expected = {'1': np.array([-0.00049472,  0.00049543]),
                '2': np.array([-0.00049472,  0.00049543]),
               }
    results = forecaster.in_sample_residuals

    assert results.keys() == expected.keys()
    assert np.isclose(list(results.values())[0], list(expected.values())[0]).all()
    assert np.isclose(list(results.values())[1], list(expected.values())[1]).all()


def test_fit_same_residuals_when_residuals_greater_than_1000():
    """
    Test fit return same residuals when residuals len is greater than 1000.
    Testing with two different forecaster. Residuals shouldn't be more than 
    1000 values.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(1010)), 
                           '2': pd.Series(np.arange(1010))
                           })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=True)
    results_1 = forecaster.in_sample_residuals
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=True)
    results_2 = forecaster.in_sample_residuals

    assert len(results_1['1']) == 1000
    assert len(results_2['2']) == 1000
    assert results_1.keys() == results_2.keys()
    assert list(results_1.values())[0] == approx(list(results_2.values())[0])
    assert list(results_1.values())[1] == approx(list(results_2.values())[1])


def test_fit_in_sample_residuals_not_stored():
    """
    Test that values of in_sample_residuals are not stored after fitting
    when `store_in_sample_residuals=False`.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)), 
                           '2': pd.Series(np.arange(5))
                           })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=False)
    expected = {'1': np.array([None]),
                '2': np.array([None])
               }
    results = forecaster.in_sample_residuals

    assert results.keys() == expected.keys()
    assert list(results.values())[0] == list(expected.values())[0]
    assert list(results.values())[1] == list(expected.values())[1]


def test_fit_last_window_stored():
    """
    Test that values of last window are stored after fitting.
    """
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