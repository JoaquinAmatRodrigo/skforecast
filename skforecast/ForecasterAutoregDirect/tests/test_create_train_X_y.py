# Unit test create_train_X_y ForecasterAutoregDirect
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from sklearn.linear_model import LinearRegression


def test_create_train_X_y_exception_when_len_of_y_is_lower_than_maximum_lag_plus_steps():
    """
    Test exception is raised when length of y is lower than maximum lag plus
    number of steps included in the forecaster.
    """
    y = pd.Series(np.arange(5), name='y')
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=3)
    err_msg = re.escape(
                (f'Minimum length of `y` for training this forecaster is '
                 f'{forecaster.max_lag + forecaster.steps}. Got {len(y)}.')
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.create_train_X_y(y=y)


def test_create_train_X_y_exception_when_len_y_is_different_from_len_exog():
    """
    Test exception is raised when length of y is not equal to length exog.
    """
    y = pd.Series(np.arange(10), name='y')
    exog = pd.Series(np.arange(11), name='exog')
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=3)
    err_msg = re.escape(
                (f'`exog` must have same number of samples as `y`. '
                 f'length `exog`: ({len(exog)}), length `y`: ({len(y)})')
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.create_train_X_y(y=y, exog=exog)


def test_create_train_X_y_output_when_lags_3_steps_1_and_exog_is_None():
    """
    Test output of create_train_X_y when regressor is LinearRegression, lags is 3
    and steps is 1.
    """
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=1)
    results = forecaster.create_train_X_y(y=pd.Series(np.arange(10)))
    expected = (pd.DataFrame(
                    data = np.array([[2., 1., 0.],
                                     [3., 2., 1.],
                                     [4., 3., 2.],
                                     [5., 4., 3.],
                                     [6., 5., 4.],
                                     [7., 6., 5.],
                                     [8., 7., 6.]]),
                    index   = np.array([3, 4, 5, 6, 7, 8, 9]),
                    columns = ['lag_1', 'lag_2', 'lag_3']
                ),
                pd.DataFrame(
                    np.array([3., 4., 5., 6., 7., 8., 9.]),
                    index = np.array([3, 4, 5, 6, 7, 8, 9]),
                    columns = ['y_step_1'])
               )  
 
    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            assert (results[i] == expected[i]).all()


def test_create_train_X_y_output_when_lags_3_steps_2_and_exog_is_None():
    """
    Test output of create_train_X_y when regressor is LinearRegression, lags is 3
    and steps is 2.
    """
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=2)
    results = forecaster.create_train_X_y(y=pd.Series(np.arange(10)))
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
                pd.DataFrame(
                    np.array([[3., 4.],
                              [4., 5.],
                              [5., 6.],
                              [6., 7.],
                              [7., 8.],
                              [8., 9.]]),
                    index = np.array([4, 5, 6, 7, 8, 9]),
                    columns = ['y_step_1', 'y_step_2'])
               )  
 
    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            assert (results[i] == expected[i]).all()
    
    

def test_create_train_X_y_output_when_lags_5_steps_1_and_exog_is_series():
    """
    Test output of create_train_X_y when regressor is LinearRegression, lags is 5,
    steps is 1 and exog is pandas Series.
    """
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=5, steps=1)
    results = forecaster.create_train_X_y(
                    y = pd.Series(np.arange(10)),
                    exog = pd.Series(np.arange(100, 110), name='exog')
                )
    expected = (pd.DataFrame(
                    data = np.array([[4., 3., 2., 1., 0., 105.],
                                     [5., 4., 3., 2., 1., 106.],
                                     [6., 5., 4., 3., 2., 107.],
                                     [7., 6., 5., 4., 3., 108.],
                                     [8., 7., 6., 5., 4., 109.]]),
                    index = np.array([5, 6, 7, 8, 9]),
                    columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_step_1']
                ),
                pd.DataFrame(
                    np.array([[5.],
                              [6.],
                              [7.],
                              [8.],
                              [9.]]),
                    index = np.array([5, 6, 7, 8, 9]),
                    columns = ['y_step_1'])
               )    

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            assert (results[i] == expected[i]).all()


def test_create_train_X_y_output_when_lags_5_steps_2_and_exog_is_series():
    """
    Test output of create_train_X_y when regressor is LinearRegression, lags is 5,
    steps is 2 and exog is pandas Series.
    """
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=5, steps=2)
    results = forecaster.create_train_X_y(
                    y = pd.Series(np.arange(10)),
                    exog = pd.Series(np.arange(100, 110), name='exog')
                )
    expected = (np.array([[4., 3., 2., 1., 0., 105., 106.],
                          [5., 4., 3., 2., 1., 106., 107.],
                          [6., 5., 4., 3., 2., 107., 108.],
                          [7., 6., 5., 4., 3., 108., 109.]]),
                np.array([[5., 6.],
                          [6., 7.],
                          [7., 8.],
                          [8., 9.]]))   
 
    expected = (pd.DataFrame(
                    data = np.array([[4., 3., 2., 1., 0., 105., 106.],
                                     [5., 4., 3., 2., 1., 106., 107.],
                                     [6., 5., 4., 3., 2., 107., 108.],
                                     [7., 6., 5., 4., 3., 108., 109.]]),
                    index = np.array([6, 7, 8, 9]),
                    columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
                               'exog_step_1', 'exog_step_2']
                ),
                pd.DataFrame(
                    np.array([[5., 6.],
                              [6., 7.],
                              [7., 8.],
                              [8., 9.]]),
                    index = np.array([6, 7, 8, 9]),
                    columns = ['y_step_1', 'y_step_2'])
               )    

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            assert (results[i] == expected[i]).all()
    

def test_create_train_X_y_output_when_lags_5_steps_1_y_is_range_10_and_exog_is_dataframe():
    """
    Test output of create_train_X_y when regressor is LinearRegression, lags is 5,
    steps is 2 and exog is pandas dataframe.
    """
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=5, steps=1)
    results = forecaster.create_train_X_y(
                y = pd.Series(np.arange(10)),
                exog = pd.DataFrame({
                           'exog_1' : np.arange(100, 110),
                           'exog_2' : np.arange(1000, 1010)
                       })
              )  

    expected = (pd.DataFrame(
                    data = np.array([[4., 3., 2., 1., 0., 105., 1005.],
                                     [5., 4., 3., 2., 1., 106., 1006.],
                                     [6., 5., 4., 3., 2., 107., 1007.],
                                     [7., 6., 5., 4., 3., 108., 1008.],
                                     [8., 7., 6., 5., 4., 109., 1009.]], dtype=float),
                    index = np.array([5, 6, 7, 8, 9]),
                    columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
                               'exog_1_step_1', 'exog_2_step_1']
                ),
                pd.DataFrame(
                    np.array([[5.],
                              [6.],
                              [7.],
                              [8.],
                              [9.]]),
                    index = np.array([5, 6, 7, 8, 9]),
                    columns = ['y_step_1'])
               )    

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            assert (results[i] == expected[i]).all()

    
def test_create_train_X_y_output_when_lags_5_steps_3_and_and_exog_is_dataframe():
    """
    Test output of create_train_X_y when regressor is LinearRegression, lags is 5,
    steps is 3 and exog is pandas dataframe.
    """
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=5, steps=3)
    results = forecaster.create_train_X_y(
                y = pd.Series(np.arange(10)),
                exog = pd.DataFrame({
                           'exog_1' : np.arange(100, 110),
                           'exog_2' : np.arange(1000, 1010)
                       })
              )

    expected = (pd.DataFrame(
                    data = np.array([[4., 3., 2., 1., 0., 105., 106., 107., 1005., 1006., 1007.],
                                     [5., 4., 3., 2., 1., 106., 107., 108., 1006., 1007., 1008.],
                                     [6., 5., 4., 3., 2., 107., 108., 109., 1007., 1008., 1009.]],
                                    dtype=float),
                    index = np.array([7, 8, 9]),
                    columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
                               'exog_1_step_1', 'exog_1_step_2', 'exog_1_step_3',
                               'exog_2_step_1', 'exog_2_step_2', 'exog_2_step_3']
                ),
                pd.DataFrame(
                    np.array([[5., 6., 7.],
                              [6., 7., 8.],
                              [7., 8., 9.]], dtype=float),
                    index = np.array([7, 8, 9]),
                    columns = ['y_step_1', 'y_step_2', 'y_step_3'])
               )
    
    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            assert (results[i] == expected[i]).all()