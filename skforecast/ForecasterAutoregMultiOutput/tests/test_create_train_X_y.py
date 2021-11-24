# Unit test create_train_X_y
# ==============================================================================

import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from sklearn.linear_model import LinearRegression

def test_create_train_X_y_output_when_lags_3_steps_1_and_exog_is_None():
    '''
    Test output of create_train_X_y when regressor is LinearRegression, lags is 3
    and steps is 1.
    '''
    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=3, steps=1)
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
                    columns = ['y_step_0'])
               )  
 
    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_frame_equal(results[1], expected[1])


def test_create_train_X_y_output_when_lags_3_steps_2_and_exog_is_None():
    '''
    Test output of create_train_X_y when regressor is LinearRegression, lags is 3
    and steps is 1.
    '''
    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=3, steps=2)
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
                    columns = ['y_step_0', 'y_step_1'])
               )  
 
    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_frame_equal(results[1], expected[1])
    
    

def test_create_train_X_y_output_when_lags_5_steps_1_y_is_range_10_and_exog_is_1d_array():

    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=5, steps=1)
    results = forecaster.create_train_X_y(y=np.arange(10), exog=np.arange(100, 110))
    expected = (np.array([[4.,   3.,   2.,   1.,   0., 105.],
                         [5.,   4.,   3.,   2.,   1., 106.],
                         [6.,   5.,   4.,   3.,   2., 107.],
                         [7.,   6.,   5.,   4.,   3., 108.],
                         [8.,   7.,   6.,   5.,   4., 109.]]),
                np.array([[5.],
                         [6.],
                         [7.],
                         [8.],
                         [9.]]))     

    assert (results[0] == expected[0]).all()
    assert (results[1] == expected[1]).all()


def test_create_train_X_y_output_when_lags_5_steps_2_y_is_range_10_and_exog_is_1d_array():

    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=5, steps=2)
    results = forecaster.create_train_X_y(y=np.arange(10), exog=np.arange(100, 110))
    expected = (np.array([[  4.,   3.,   2.,   1.,   0., 105., 106.],
                        [  5.,   4.,   3.,   2.,   1., 106., 107.],
                        [  6.,   5.,   4.,   3.,   2., 107., 108.],
                        [  7.,   6.,   5.,   4.,   3., 108., 109.]]),
                np.array([[5., 6.],
                        [6., 7.],
                        [7., 8.],
                        [8., 9.]]))   

    assert (results[0] == expected[0]).all()
    assert (results[1] == expected[1]).all()


def test_create_train_X_y_output_when_lags_5_steps_1_y_is_range_10_and_exog_is_2d_array():

    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=5, steps=1)
    results = forecaster.create_train_X_y(
            y=np.arange(10),
            exog=np.column_stack([np.arange(100, 110), np.arange(1000, 1010)])
          )
    expected = (np.array([[4,    3,    2,    1,    0,  105, 1005],
                        [5,    4,    3,    2,    1,  106, 1006],
                        [6,    5,    4,    3,    2,  107, 1007],
                        [7,    6,    5,    4,    3,  108, 1008],
                        [8,    7,    6,    5,    4,  109, 1009]]),
                np.array([[5.],
                         [6.],
                         [7.],
                         [8.],
                         [9.]]))     

    assert (results[0] == expected[0]).all()
    assert (results[1] == expected[1]).all()

def test_create_train_X_y_output_when_lags_5_steps_3_y_is_range_10_and_exog_is_2d_array():

    forecaster = ForecasterAutoregMultiOutput(LinearRegression(), lags=5, steps=3)
    results = forecaster.create_train_X_y(
            y=np.arange(10),
            exog=np.column_stack([np.arange(100, 110), np.arange(1000, 1010)])
          )
    expected = (np.array([[   4,    3,    2,    1,    0,  105,  106,  107, 1005, 1006, 1007],
                        [   5,    4,    3,    2,    1,  106,  107,  108, 1006, 1007, 1008],
                        [   6,    5,    4,    3,    2,  107,  108,  109, 1007, 1008, 1009]]),
                np.array([[5, 6, 7],
                         [6, 7, 8],
                         [7, 8, 9]]))     

    assert (results[0] == expected[0]).all()
    assert (results[1] == expected[1]).all()