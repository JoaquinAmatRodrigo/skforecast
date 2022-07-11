# Unit test _estimate_boot_interval ForecasterAutoregMultiSeries
# ==============================================================================
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.linear_model import LinearRegression


def test_estimate_boot_interval_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_True():
    '''
    Test output of _estimate_boot_interval when regressor is LinearRegression and
    1 step is predicted using in-sample residuals.
    '''
    series = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                           '2': pd.Series(np.arange(10))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=True)

    forecaster.in_sample_residuals['1'] = np.full_like(forecaster.in_sample_residuals['1'], fill_value=10)
    expected_1 = np.array([[20., 20.]])
    results_1 = forecaster._estimate_boot_interval(steps=1, level = '1', in_sample_residuals=True)

    forecaster.in_sample_residuals['2'] = np.full_like(forecaster.in_sample_residuals['2'], fill_value=20)
    expected_2 = np.array([[30., 30.]])
    results_2 = forecaster._estimate_boot_interval(steps=1, level = '2', in_sample_residuals=True)

    assert results_1 == approx(expected_1)
    assert results_2 == approx(expected_2)


def test_estimate_boot_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_True():
    '''
    Test output of _estimate_boot_interval when regressor is LinearRegression and
    2 steps are predicted using in-sample residuals.
    '''
    series = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                           '2': pd.Series(np.arange(10))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=True)

    forecaster.in_sample_residuals['1'] = np.full_like(forecaster.in_sample_residuals['1'], fill_value=10)
    expected_1 = np.array([[20.        , 20.],
                           [24.33333333, 24.33333333]])
    results_1 = forecaster._estimate_boot_interval(steps=2, level = '1', in_sample_residuals=True)

    forecaster.in_sample_residuals['2'] = np.full_like(forecaster.in_sample_residuals['2'], fill_value=20)
    expected_2 = np.array([[30.              , 30.],
                           [37.66666666666667, 37.66666666666667]])
    results_2 = forecaster._estimate_boot_interval(steps=2, level = '2', in_sample_residuals=True)

    assert results_1 == approx(expected_1)
    assert results_2 == approx(expected_2)


def test_estimate_boot_interval_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_False():
    '''
    Test output of _estimate_boot_interval when regressor is LinearRegression and
    1 step is predicted using out-sample residuals.
    '''
    series = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                           '2': pd.Series(np.arange(10))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=True)

    forecaster.out_sample_residuals = pd.Series(np.full_like(forecaster.in_sample_residuals['1'], fill_value=10))
    expected_1 = np.array([[20., 20.]])
    results_1 = forecaster._estimate_boot_interval(steps=1, level = '1', in_sample_residuals=False)

    forecaster.out_sample_residuals = pd.Series(np.full_like(forecaster.in_sample_residuals['2'], fill_value=20))
    expected_2 = np.array([[30., 30.]])
    results_2 = forecaster._estimate_boot_interval(steps=1, level = '2', in_sample_residuals=False)

    assert results_1 == approx(expected_1)
    assert results_2 == approx(expected_2)


def test_estimate_boot_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_False():
    '''
    Test output of _estimate_boot_interval when regressor is LinearRegression and
    2 steps are predicted using out-sample residuals.
    '''
    series = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                           '2': pd.Series(np.arange(10))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=True)

    forecaster.out_sample_residuals = pd.Series(np.full_like(forecaster.in_sample_residuals['1'], fill_value=10))
    expected_1 = np.array([[20.        , 20.        ],
                           [24.33333333, 24.33333333]])
    results_1 = forecaster._estimate_boot_interval(steps=2, level = '1', in_sample_residuals=False)

    forecaster.out_sample_residuals = pd.Series(np.full_like(forecaster.in_sample_residuals['2'], fill_value=20))
    expected_2 = np.array([[30.              , 30.],
                           [37.66666666666667, 37.66666666666667]])
    results_2 = forecaster._estimate_boot_interval(steps=2, level = '2', in_sample_residuals=False)

    assert results_1 == approx(expected_1)
    assert results_2 == approx(expected_2)