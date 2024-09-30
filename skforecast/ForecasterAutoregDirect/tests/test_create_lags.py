# Unit test _create_lags ForecasterAutoregDirect
# ==============================================================================
import numpy as np
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from sklearn.linear_model import LinearRegression

  
def test_create_lags_when_lags_is_3_steps_1_and_y_is_numpy_arange_10():
    """
    Test matrix of lags created properly when lags is 3, steps is 1 and y is
    np.arange(10).
    """
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=1)
    results = forecaster._create_lags(y=np.arange(10))
    expected = (np.array([[2., 1., 0.],
                          [3., 2., 1.],
                          [4., 3., 2.],
                          [5., 4., 3.],
                          [6., 5., 4.],
                          [7., 6., 5.],
                          [8., 7., 6.]]),
                np.array([[3., 4., 5., 6., 7., 8., 9.]])
                )

    assert (results[0] == expected[0]).all()
    assert (results[1] == expected[1]).all()

  
def test_create_lags_when_lags_is_list_interspersed_lags_steps_1_and_y_is_numpy_arange_10():
    """
    Test matrix of lags created properly when lags is a list with interspersed 
    lags, steps is 1 and y is np.arange(10).
    """
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=[1, 5], steps=1)
    results = forecaster._create_lags(y=np.arange(10))
    expected = (np.array([[4., 0.],
                          [5., 1.],
                          [6., 2.],
                          [7., 3.],
                          [8., 4.]]),
                np.array([[5., 6., 7., 8., 9.]])
                )

    assert (results[0] == expected[0]).all()
    assert (results[1] == expected[1]).all()


def test_create_lags_when_lags_is_3_steps_2_and_y_is_numpy_arange_10():
    """
    Test matrix of lags created properly when lags is 3, steps is 2 and y is
    np.arange(10).
    """
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=2)
    results = forecaster._create_lags(y=np.arange(10))
    expected = (np.array([[2., 1., 0.],
                          [3., 2., 1.],
                          [4., 3., 2.],
                          [5., 4., 3.],
                          [6., 5., 4.],
                          [7., 6., 5.]]),
                np.array([[3., 4., 5., 6., 7., 8.],
                          [4., 5., 6., 7., 8., 9.]])
                )

    assert (results[0] == expected[0]).all()
    assert (results[1] == expected[1]).all()


def test_create_lags_when_lags_is_3_steps_5_and_y_is_numpy_arange_10():
    """
    Test matrix of lags created properly when lags is 3, steps is 5 and y is
    np.arange(10).
    """
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=5)
    results = forecaster._create_lags(y=np.arange(10))
    expected = (np.array([[2., 1., 0.],
                          [3., 2., 1.],
                          [4., 3., 2.]]),
                np.array([[3., 4., 5.],
                          [4., 5., 6.],
                          [5., 6., 7.],
                          [6., 7., 8.],
                          [7., 8., 9.]])
                )

    assert (results[0] == expected[0]).all()
    assert (results[1] == expected[1]).all()