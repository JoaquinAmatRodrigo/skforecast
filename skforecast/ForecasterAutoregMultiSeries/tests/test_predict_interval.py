# Unit test set_out_sample_residuals ForecasterAutoregMultiSeries
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.linear_model import LinearRegression


def test_predict_interval_exception_when_forecaster_in_sample_residuals_are_not_stored():
    '''
    Test exception is raised when forecaster.in_sample_residuals are not stored 
    during fit method.
    '''
    series = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                           '2': pd.Series(np.arange(10))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=False)

    with pytest.raises(Exception):
        forecaster.predict_interval(steps=1, level='1', in_sample_residuals=True)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_True():
    '''
    Test output when regressor is LinearRegression and one step ahead is predicted
    using in sample residuals.
    '''
    series = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                           '2': pd.Series(np.arange(10))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=True)

    forecaster.in_sample_residuals['1'] = np.full_like(forecaster.in_sample_residuals['1'], fill_value=10)
    expected_1 = pd.DataFrame(
                    np.array([[10., 20., 20.]]),
                    columns = ['pred', 'lower_bound', 'upper_bound'],
                    index = pd.RangeIndex(start=10, stop=11, step=1)
                 )
    results_1 = forecaster.predict_interval(steps=1, level='1', in_sample_residuals=True)

    forecaster.in_sample_residuals['2'] = np.full_like(forecaster.in_sample_residuals['2'], fill_value=20)
    expected_2 = pd.DataFrame(
                    np.array([[10., 30., 30.]]),
                    columns = ['pred', 'lower_bound', 'upper_bound'],
                    index = pd.RangeIndex(start=10, stop=11, step=1)
                 )
    results_2 = forecaster.predict_interval(steps=1, level='2', in_sample_residuals=True)

    pd.testing.assert_frame_equal(results_1, expected_1)
    pd.testing.assert_frame_equal(results_2, expected_2)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_True():
    '''
    Test output when regressor is LinearRegression and two step ahead is predicted
    using in sample residuals.
    '''
    series = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                           '2': pd.Series(np.arange(10))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=True)

    forecaster.in_sample_residuals['1'] = np.full_like(forecaster.in_sample_residuals['1'], fill_value=10)
    expected_1 = pd.DataFrame(
                    np.array([[10., 20.        , 20.],
                              [11., 24.33333333, 24.33333333]
                             ]),
                    columns = ['pred', 'lower_bound', 'upper_bound'],
                    index = pd.RangeIndex(start=10, stop=12, step=1)
                 )
    results_1 = forecaster.predict_interval(steps=2, level='1', in_sample_residuals=True)

    forecaster.in_sample_residuals['2'] = np.full_like(forecaster.in_sample_residuals['2'], fill_value=20)
    expected_2 = pd.DataFrame(
                    np.array([[10., 30.              , 30.],
                              [11., 37.66666666666667, 37.66666666666667]
                             ]),
                    columns = ['pred', 'lower_bound', 'upper_bound'],
                    index = pd.RangeIndex(start=10, stop=12, step=1)
                 )
    results_2 = forecaster.predict_interval(steps=2, level='2', in_sample_residuals=True)

    pd.testing.assert_frame_equal(results_1, expected_1)
    pd.testing.assert_frame_equal(results_2, expected_2)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_False():
    '''
    Test output when regressor is LinearRegression and one step ahead is predicted
    using out sample residuals.
    '''
    series = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                           '2': pd.Series(np.arange(10))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=True)

    forecaster.out_sample_residuals = pd.Series(np.full_like(forecaster.in_sample_residuals['1'], fill_value=10))
    expected_1 = pd.DataFrame(
                    np.array([[10., 20., 20.]]),
                    columns = ['pred', 'lower_bound', 'upper_bound'],
                    index = pd.RangeIndex(start=10, stop=11, step=1)
                 )
    results_1 = forecaster.predict_interval(steps=1, level='1', in_sample_residuals=False)

    forecaster.out_sample_residuals = pd.Series(np.full_like(forecaster.in_sample_residuals['2'], fill_value=20))
    expected_2 = pd.DataFrame(
                    np.array([[10., 30., 30.]]),
                    columns = ['pred', 'lower_bound', 'upper_bound'],
                    index = pd.RangeIndex(start=10, stop=11, step=1)
                 )
    results_2 = forecaster.predict_interval(steps=1, level='2', in_sample_residuals=False)

    pd.testing.assert_frame_equal(results_1, expected_1)
    pd.testing.assert_frame_equal(results_2, expected_2)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_False():
    '''
    Test output when regressor is LinearRegression and two step ahead is predicted
    using out sample residuals.
    '''
    series = pd.DataFrame({'1': pd.Series(np.arange(10)), 
                           '2': pd.Series(np.arange(10))
                          })

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, store_in_sample_residuals=True)

    forecaster.out_sample_residuals = pd.Series(np.full_like(forecaster.in_sample_residuals['1'], fill_value=10))
    expected_1 = pd.DataFrame(
                    np.array([[10., 20.        , 20.],
                              [11., 24.33333333, 24.33333333]
                             ]),
                    columns = ['pred', 'lower_bound', 'upper_bound'],
                    index = pd.RangeIndex(start=10, stop=12, step=1)
                 )
    results_1 = forecaster.predict_interval(steps=2, level='1', in_sample_residuals=False)

    forecaster.out_sample_residuals = pd.Series(np.full_like(forecaster.in_sample_residuals['2'], fill_value=20))
    expected_2 = pd.DataFrame(
                    np.array([[10., 30.              , 30.],
                              [11., 37.66666666666667, 37.66666666666667]
                             ]),
                    columns = ['pred', 'lower_bound', 'upper_bound'],
                    index = pd.RangeIndex(start=10, stop=12, step=1)
                 )
    results_2 = forecaster.predict_interval(steps=2, level='2', in_sample_residuals=False)

    pd.testing.assert_frame_equal(results_1, expected_1)
    pd.testing.assert_frame_equal(results_2, expected_2)