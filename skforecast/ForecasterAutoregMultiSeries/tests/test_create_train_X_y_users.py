# Unit test create_train_X_y ForecasterAutoregMultiSeries
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.linear_model import LinearRegression


def test_create_train_X_y_output_when_series_and_exog_is_None():
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is None.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(7, dtype=float)), 
                           '2': pd.Series(np.arange(7, dtype=float))})
    forecaster = ForecasterAutoregMultiSeries(
        regressor = LinearRegression(),
        lags      = 3,
        encoding  = "onehot"
    )

    results = forecaster.create_train_X_y(series=series)
    expected = (
        pd.DataFrame(
            data = np.array([[-0.5, -1. , -1.5, 1., 0.],
                             [ 0. , -0.5, -1. , 1., 0.],
                             [ 0.5,  0. , -0.5, 1., 0.],
                             [ 1. ,  0.5,  0. , 1., 0.],
                             [-0.5, -1. , -1.5, 0., 1.],
                             [ 0. , -0.5, -1. , 0., 1.],
                             [ 0.5,  0. , -0.5, 0., 1.],
                             [ 1. ,  0.5,  0. , 0., 1.]]),
            index   = pd.Index([3, 4, 5, 6, 3, 4, 5, 6]),
            columns = ['lag_1', 'lag_2', 'lag_3', '1', '2']
        ).astype({'1': int, '2': int}),
        pd.Series(
            data  = np.array([0., 0.5, 1., 1.5, 0., 0.5, 1., 1.5]),
            index = pd.Index([3, 4, 5, 6, 3, 4, 5, 6]),
            name  = 'y',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])