# Unit test fit ForecasterAutoregMultiVariate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def test_fit_correct_dict_create_transformer_series():
    """
    Test fit method creates correctly all the auxiliary dicts transformer_series_.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(10)), 
                           'l3': pd.Series(np.arange(10))})

    transformer_series = {'l1': StandardScaler(), 'l3': StandardScaler(), 'l4': StandardScaler()}

    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = LinearRegression(), 
                     level              = 'l1',
                     lags               = 3,
                     steps              = 2,
                     transformer_series = transformer_series
                 )
    
    forecaster.fit(series=series, store_in_sample_residuals=False)
    expected_transformer_series_ = {'l1': forecaster.transformer_series_['l1'], 'l2': None, 'l3': forecaster.transformer_series_['l3']}

    assert forecaster.transformer_series_ == expected_transformer_series_

    forecaster.fit(series=series[['l1', 'l2']], store_in_sample_residuals=False)
    expected_transformer_series_ = {'l1': forecaster.transformer_series_['l1'], 'l2': None}

    assert forecaster.transformer_series_ == expected_transformer_series_


def test_forecaster_DatetimeIndex_index_freq_stored():
    """
    Test series.index.freqstr is stored in forecaster.index_freq.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(10))})

    series.index = pd.date_range(start='2022-01-01', periods=10, freq='1D')

    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), 
                                               level='l1', lags=3, steps=1)
    forecaster.fit(series=series)
    expected = series.index.freqstr
    results = forecaster.index_freq

    assert results == expected


def test_forecaster_index_step_stored():
    """
    Test serie without DatetimeIndex, step is stored in forecaster.index_freq.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(10))})
    
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), 
                                               level='l1', lags=3, steps=2)
    forecaster.fit(series=series)
    expected = series.index.step
    results = forecaster.index_freq

    assert results == expected
    

@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'], 
                         ids=lambda n_jobs: f'n_jobs: {n_jobs}')
def test_fit_in_sample_residuals_stored(n_jobs):
    """
    Test that values of in_sample_residuals are stored after fitting.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(10))})
    
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1', 
                                               lags=3, steps=2, n_jobs=n_jobs)
    forecaster.fit(series=series)
    expected = {1: np.array([0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 
                             0.0000000e+00, 0.0000000e+00, 8.8817842e-16]),
                2: np.array([0., 0., 0., 0., 0., 0.])}
    results = forecaster.in_sample_residuals

    assert isinstance(results, dict)
    assert np.all(isinstance(x, np.ndarray) for x in results.values())
    assert results.keys() == expected.keys()
    assert np.all(np.all(np.isclose(results[k], expected[k])) for k in expected.keys())


@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'], 
                         ids=lambda n_jobs: f'n_jobs: {n_jobs}')
def test_fit_same_residuals_when_residuals_greater_than_1000(n_jobs):
    """
    Test fit return same residuals when residuals len is greater than 1000.
    Testing with two different forecaster.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(1200)), 
                           'l2': pd.Series(np.arange(1200))})
    
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1', 
                                               lags=3, steps=2, n_jobs=n_jobs)
    forecaster.fit(series=series)
    results_1 = forecaster.in_sample_residuals

    forecaster = ForecasterAutoregMultiVariate(LinearRegression(),  
                                               level='l1', lags=3, steps=2)
    forecaster.fit(series=series)
    results_2 = forecaster.in_sample_residuals

    assert isinstance(results_1, dict)
    assert np.all(isinstance(x, np.ndarray) for x in results_1.values())
    assert isinstance(results_2, dict)
    assert np.all(isinstance(x, np.ndarray) for x in results_2.values())
    assert results_1.keys() == results_2.keys()
    assert np.all(len(results_1[k] == 1000) for k in results_1.keys())
    assert np.all(len(results_2[k] == 1000) for k in results_2.keys())
    assert np.all(np.all(results_1[k] == results_2[k]) for k in results_2.keys())


@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'], 
                         ids=lambda n_jobs: f'n_jobs: {n_jobs}')
def test_fit_in_sample_residuals_not_stored(n_jobs):
    """
    Test that values of in_sample_residuals are not stored after fitting
    when `store_in_sample_residuals=False`.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(5)), 
                           'l2': pd.Series(np.arange(5))})

    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1', 
                                               lags=3, steps=2, n_jobs=n_jobs)
    forecaster.fit(series=series, store_in_sample_residuals=False)
    expected = {1: None, 2: None}
    results = forecaster.in_sample_residuals

    assert isinstance(results, dict)
    assert results.keys() == expected.keys()
    assert np.all(results[k] == expected[k] for k in expected.keys())


@pytest.mark.parametrize("store_last_window", 
                         [True, ['l1', 'l2'], False], 
                         ids=lambda lw: f'store_last_window: {lw}')
def test_fit_last_window_stored(store_last_window):
    """
    Test that values of last window are stored after fitting.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(50, 60))})

    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), 
                                               level='l1', lags=3, steps=2)
    forecaster.fit(series=series, store_last_window=store_last_window)

    expected = pd.DataFrame({
        'l1': pd.Series(np.array([7, 8, 9])), 
        'l2': pd.Series(np.array([57, 58, 59]))
    })
    expected.index = pd.RangeIndex(start=7, stop=10, step=1)

    if store_last_window:
        pd.testing.assert_frame_equal(forecaster.last_window, expected)
        assert forecaster.series_col_names == ['l1', 'l2']
        assert forecaster.series_X_train == ['l1', 'l2']
    else:
        assert forecaster.last_window == None


def test_fit_last_window_stored_when_different_lags():
    """
    Test that values of last window are stored after fitting when different lags
    configurations.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(100, 110))})

    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l2',
                                               lags = {'l1': 3, 'l2': [1, 5]}, 
                                               steps = 2)
    forecaster.fit(series=series)

    expected = pd.DataFrame({
        'l1': pd.Series(np.array([5, 6, 7, 8, 9])), 
        'l2': pd.Series(np.array([105, 106, 107, 108, 109]))
    })
    expected.index = pd.RangeIndex(start=5, stop=10, step=1)

    pd.testing.assert_frame_equal(forecaster.last_window, expected)
    assert forecaster.series_col_names == ['l1', 'l2']
    assert forecaster.series_X_train == ['l1', 'l2']


@pytest.mark.parametrize("level",
                         ['l1', 'l2'],
                         ids=lambda level: f'level: {level}')
def test_fit_last_window_stored_when_lags_dict_with_None(level):
    """
    Test that values of last window are stored after fitting when lags is a dict
    with None values.    
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(100, 110))})

    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level=level,
                                               lags = {'l1': 3, 'l2': None}, 
                                               steps = 2)
    forecaster.fit(series=series)

    expected = pd.DataFrame({'l1': pd.Series(np.array([7, 8, 9]))})
    expected.index = pd.RangeIndex(start=7, stop=10, step=1)

    pd.testing.assert_frame_equal(forecaster.last_window, expected)
    assert forecaster.series_col_names == ['l1'] if level == 'l1' else ['l1', 'l2']
    assert forecaster.series_X_train == ['l1']