# Unit test create_train_X_y ForecasterAutoregDirect
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from sklearn.linear_model import LinearRegression


@pytest.mark.parametrize("step", [0, 4], ids=lambda step: f'step: {step}')
def test_filter_train_X_y_for_step_ValueError_when_step_not_in_steps(step):
    """
    Test ValueError is raised when step not in steps.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)

    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=3)
    X_train, y_train = forecaster.create_train_X_y(y)

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
    y = pd.Series(np.arange(10), name='y', dtype=float)

    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=2)
    X_train, y_train = forecaster.create_train_X_y(y=y)
    results = forecaster.filter_train_X_y_for_step(step=1, X_train=X_train, y_train=y_train)

    expected = (
        pd.DataFrame(
            data = np.array([[2., 1., 0.],
                             [3., 2., 1.],
                             [4., 3., 2.],
                             [5., 4., 3.],
                             [6., 5., 4.],
                             [7., 6., 5.]], dtype=float),
            index   = pd.RangeIndex(start=3, stop=9, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3']
        ),
        pd.Series(
            data  = np.array([3., 4., 5., 6., 7., 8.], dtype=float),
            index = pd.RangeIndex(start=3, stop=9, step=1),
            name  = 'y_step_1'
        )
    )  
 
    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_filter_train_X_y_for_step_output_when_lags_3_steps_2_and_exog_for_step_2():
    """
    Test output of filter_train_X_y_for_step when regressor is LinearRegression, 
    lags is 3 and steps is 2 with exog for step 2.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=int)

    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=2)
    X_train, y_train = forecaster.create_train_X_y(y=y, exog=exog)
    results = forecaster.filter_train_X_y_for_step(step=2, X_train=X_train, y_train=y_train)

    expected = (
        pd.DataFrame(
            data = np.array([[2., 1., 0., 104],
                             [3., 2., 1., 105],
                             [4., 3., 2., 106],
                             [5., 4., 3., 107],
                             [6., 5., 4., 108],
                             [7., 6., 5., 109]], dtype=float),
            index   = pd.RangeIndex(start=4, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'exog_step_2']
        ).astype({'exog_step_2': int}),
        pd.Series(
            data  = np.array([4., 5., 6., 7., 8., 9.]),
            index = pd.RangeIndex(start=4, stop=10, step=1),
            name  = 'y_step_2', 
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
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=int)

    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=2)
    X_train, y_train = forecaster.create_train_X_y(y=y, exog=exog)
    results = forecaster.filter_train_X_y_for_step(
                  step          = 2, 
                  X_train       = X_train, 
                  y_train       = y_train,
                  remove_suffix = True
              )

    expected = (
        pd.DataFrame(
            data = np.array([[2., 1., 0., 104],
                             [3., 2., 1., 105],
                             [4., 3., 2., 106],
                             [5., 4., 3., 107],
                             [6., 5., 4., 108],
                             [7., 6., 5., 109]], dtype=float),
            index   = pd.RangeIndex(start=4, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'exog']
        ).astype({'exog': int}),
        pd.Series(
            data  = np.array([4., 5., 6., 7., 8., 9.]),
            index = pd.RangeIndex(start=4, stop=10, step=1),
            name  = 'y', 
            dtype = float
        )
    )  
 
    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])