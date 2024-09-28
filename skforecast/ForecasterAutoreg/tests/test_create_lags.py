# Unit test _create_lags ForecasterAutoreg
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression


def test_create_lags_output():
    """
    Test matrix of lags is created properly when lags=3 and y=np.arange(10).
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    results = forecaster._create_lags(y=np.arange(10))
    expected = (
        np.array([[2., 1., 0.],
                  [3., 2., 1.],
                  [4., 3., 2.],
                  [5., 4., 3.],
                  [6., 5., 4.],
                  [7., 6., 5.],
                  [8., 7., 6.]]),
        np.array([3., 4., 5., 6., 7., 8., 9.])
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], expected[1])


def test_create_lags_output_interspersed_lags():
    """
    Test matrix of lags is is a list with interspersed lags.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=[2, 3])
    results = forecaster._create_lags(y=np.arange(10))
    expected = (
        np.array([[1., 0.],
                  [2., 1.],
                  [3., 2.],
                  [4., 3.],
                  [5., 4.],
                  [6., 5.],
                  [7., 6.]]),
        np.array([3., 4., 5., 6., 7., 8., 9.])
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], expected[1])


def test_create_lags_output_pandas():
    """
    Test matrix of lags is created properly when X_as_pandas=True.
    """
    forecaster = ForecasterAutoreg(LinearRegression(), lags=3)
    results = forecaster._create_lags(
        y=np.arange(10), X_as_pandas=True, 
        train_index=pd.date_range('2020-01-03', periods=7, freq='D')
    )
    expected = (
        pd.DataFrame(
            data = np.array([
                       [2., 1., 0.],
                       [3., 2., 1.],
                       [4., 3., 2.],
                       [5., 4., 3.],
                       [6., 5., 4.],
                       [7., 6., 5.],
                       [8., 7., 6.]]
                   ),
            columns = ["lag_1", "lag_2", "lag_3"],
            index = pd.date_range('2020-01-03', periods=7, freq='D')
        ),
        np.array([3., 4., 5., 6., 7., 8., 9.])
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], expected[1])
