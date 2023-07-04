# Unit test create_train_X_y ForecasterAutoregMultiVariate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from sklearn.linear_model import LinearRegression


@pytest.mark.parametrize("step", [0, 4], ids=lambda step: f'step: {step}')
def test_filter_train_X_y_for_step_exception_when_step_not_in_steps(step):
    """
    Test exception is raised when step not in steps.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(100, 110), dtype=float)})
    
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3)
    X_train, y_train = forecaster.create_train_X_y(series=series)

    err_msg = re.escape(
                (f"Invalid value `step`. For this forecaster, minimum value is 1 "
                 f"and the maximum step is {forecaster.steps}.")
            )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.filter_train_X_y_for_step(step=step, X_train=X_train, y_train=y_train)


def test_filter_train_X_y_for_step_output_when_lags_3_steps_2_exog_is_None_for_step_1():
    """
    Test output of filter_train_X_y_for_step when regressor is LinearRegression, 
    lags is 3 and steps is 2 for step 1.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(100, 110), dtype=float)})
    
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=2)
    X_train, y_train = forecaster.create_train_X_y(series=series)
    results = forecaster.filter_train_X_y_for_step(step=1, X_train=X_train, y_train=y_train)

    expected = (
        pd.DataFrame(
            data = np.array([[2., 1., 0., 102., 101., 100.],
                             [3., 2., 1., 103., 102., 101.],
                             [4., 3., 2., 104., 103., 102.],
                             [5., 4., 3., 105., 104., 103.],
                             [6., 5., 4., 106., 105., 104.],
                             [7., 6., 5., 107., 106., 105.]], dtype=float),
            index   = pd.RangeIndex(start=3, stop=9, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3',
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3']
        ),
        pd.Series(
            data  = np.array([3., 4., 5., 6., 7., 8.], dtype=float),
            index = pd.RangeIndex(start=3, stop=9, step=1),
            name  = 'l1_step_1'
        )
    )
 
    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_filter_train_X_y_for_step_output_when_lags_3_steps_2_and_exog_for_step_2():
    """
    Test output of filter_train_X_y_for_step when regressor is LinearRegression, 
    lags is 3 and steps is 2 with exog for step 2.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=int)

    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l2',
                                               lags=[1, 2, 3], steps=2)
    X_train, y_train = forecaster.create_train_X_y(series=series, exog=exog)
    results = forecaster.filter_train_X_y_for_step(step=2, X_train=X_train, y_train=y_train)

    expected = (
        pd.DataFrame(
            data = np.array([[2., 1., 0., 52., 51., 50., 104.],
                             [3., 2., 1., 53., 52., 51., 105.],
                             [4., 3., 2., 54., 53., 52., 106.],
                             [5., 4., 3., 55., 54., 53., 107.],
                             [6., 5., 4., 56., 55., 54., 108.],
                             [7., 6., 5., 57., 56., 55., 109.]], dtype=float),
            index   = pd.RangeIndex(start=4, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3',
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3',
                       'exog_step_2']
        ).astype({'exog_step_2': int}),
        pd.Series(
            data  = np.array([54., 55., 56., 57., 58., 59.]),
            index = pd.RangeIndex(start=4, stop=10, step=1),
            name  = 'l2_step_2', 
            dtype = float
        )
    )  
 
    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_filter_train_X_y_for_step_output_when_lags_3_steps_2_and_exog_for_step_2_remove_suffix():
    """
    Test output of filter_train_X_y_for_step when regressor is LinearRegression, 
    lags is 3 and steps is 2 with exog for step 2 with remove_suffix=True.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=int)

    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l2',
                                               lags=[1, 2, 3], steps=2)
    X_train, y_train = forecaster.create_train_X_y(series=series, exog=exog)
    results = forecaster.filter_train_X_y_for_step(
                  step          = 2, 
                  X_train       = X_train, 
                  y_train       = y_train,
                  remove_suffix = True
              )

    expected = (
        pd.DataFrame(
            data = np.array([[2., 1., 0., 52., 51., 50., 104.],
                             [3., 2., 1., 53., 52., 51., 105.],
                             [4., 3., 2., 54., 53., 52., 106.],
                             [5., 4., 3., 55., 54., 53., 107.],
                             [6., 5., 4., 56., 55., 54., 108.],
                             [7., 6., 5., 57., 56., 55., 109.]], dtype=float),
            index   = pd.RangeIndex(start=4, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3',
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3',
                       'exog']
        ).astype({'exog': int}),
        pd.Series(
            data  = np.array([54., 55., 56., 57., 58., 59.]),
            index = pd.RangeIndex(start=4, stop=10, step=1),
            name  = 'l2', 
            dtype = float
        )
    )  
 
    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])