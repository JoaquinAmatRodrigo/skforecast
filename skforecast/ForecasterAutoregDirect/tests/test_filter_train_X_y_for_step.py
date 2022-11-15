# Unit test create_train_X_y ForecasterAutoregDirect
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from sklearn.linear_model import LinearRegression


@pytest.mark.parametrize("step", [0, 4], ids=lambda step: f'step: {step}')
def test_filter_train_X_y_for_step_exception_when_step_not_in_steps(step):
    """
    Test exception is raised when step not in steps.
    """
    y = pd.Series(np.arange(10), name='y')
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=3)
    X_train, y_train = forecaster.create_train_X_y(y)

    err_msg = re.escape(
                f"Invalid value `step`. For this forecaster, minimum value is 1 "
                f"and the maximum step is {forecaster.steps}."
            )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.filter_train_X_y_for_step(step=step, X_train=X_train, y_train=y_train)


def test_filter_train_X_y_for_step_output_when_lags_3_steps_2_exog_is_None_for_step_1():
    """
    Test output of filter_train_X_y_for_step when regressor is LinearRegression, 
    lags is 3 and steps is 2 for step 1.
    """
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=2)
    X_train, y_train = forecaster.create_train_X_y(y=pd.Series(np.arange(10)))
    results = forecaster.filter_train_X_y_for_step(step=1, X_train=X_train, y_train=y_train)
    expected = (pd.DataFrame(
                    data = np.array([[2., 1., 0.],
                                     [3., 2., 1.],
                                     [4., 3., 2.],
                                     [5., 4., 3.],
                                     [6., 5., 4.],
                                     [7., 6., 5.]]),
                    index   = np.array([4, 5, 6, 7, 8, 9]),
                    columns = ['lag_1', 'lag_2', 'lag_3']
                ),
                pd.Series(
                    data  = np.array([3., 4., 5., 6., 7., 8.]),
                    index = np.array([4, 5, 6, 7, 8, 9]),
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
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=2)
    X_train, y_train = forecaster.create_train_X_y(
                           y    = pd.Series(np.arange(10)),
                           exog = pd.Series(np.arange(100, 110), name='exog')
                       )
    results = forecaster.filter_train_X_y_for_step(step=2, X_train=X_train, y_train=y_train)
    expected = (pd.DataFrame(
                    data = np.array([[2., 1., 0., 104.],
                                     [3., 2., 1., 105.],
                                     [4., 3., 2., 106.],
                                     [5., 4., 3., 107.],
                                     [6., 5., 4., 108.],
                                     [7., 6., 5., 109.]]),
                    index   = np.array([4, 5, 6, 7, 8, 9]),
                    columns = ['lag_1', 'lag_2', 'lag_3', 'exog_step_2']
                ),
                pd.Series(
                    data  = np.array([4., 5., 6., 7., 8., 9.]),
                    index = np.array([4, 5, 6, 7, 8, 9]),
                    name  = 'y_step_2'
                )
               )  
 
    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])