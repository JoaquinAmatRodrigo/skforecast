# Unit test create_sample_weights ForecasterAutoreg
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.linear_model import LinearRegression


def custom_weights(index):
    """
    Return 0 if index is one of '2022-01-08', '2022-01-10', 1 otherwise.
    """
    weights = np.where(
                (index >= '2022-01-08') & (index <= '2022-01-10'),
                0,
                1
              )
    return weights


def custom_weights_nan(index):
    """
    Return np.nan if index is one of '2022-01-05', '2022-01-06', 1 otherwise.
    """
    weights = np.where(
                (index >= '2022-01-08') & (index <= '2022-01-10'),
                np.nan,
                1
              )
    return weights


def custom_weights_zeros(index):
    """
    Return 0 for all elements in index
    """
    weights = np.zeros_like(index, dtype=int)
    return weights


series = pd.DataFrame(
            data = np.array([[0.12362923, 0.51328688],
                            [0.65138268, 0.11599708],
                            [0.58142898, 0.72350895],
                            [0.72969992, 0.10305721],
                            [0.97790567, 0.20581485],
                            [0.56924731, 0.41262027],
                            [0.85369084, 0.82107767],
                            [0.75425194, 0.0107816 ],
                            [0.08167939, 0.94951918],
                            [0.00249297, 0.55583355]]),
            columns= ['series_1', 'series_2'],
            index = pd.DatetimeIndex(
                        ['2022-01-04', '2022-01-05', '2022-01-06', '2022-01-07',
                        '2022-01-08', '2022-01-09', '2022-01-10', '2022-01-11',
                        '2022-01-12', '2022-01-13'],
                        dtype='datetime64[ns]', freq='D'
                    ),
         )

X_train = pd.DataFrame(
          data = np.array(
                    [[0.58142898, 0.65138268, 0.12362923, 1., 0.],
                    [0.72969992, 0.58142898, 0.65138268, 1., 0.],
                    [0.97790567, 0.72969992, 0.58142898, 1., 0.],
                    [0.56924731, 0.97790567, 0.72969992, 1., 0.],
                    [0.85369084, 0.56924731, 0.97790567, 1., 0.],
                    [0.75425194, 0.85369084, 0.56924731, 1., 0.],
                    [0.08167939, 0.75425194, 0.85369084, 1., 0.],
                    [0.72350895, 0.11599708, 0.51328688, 0., 1.],
                    [0.10305721, 0.72350895, 0.11599708, 0., 1.],
                    [0.20581485, 0.10305721, 0.72350895, 0., 1.],
                    [0.41262027, 0.20581485, 0.10305721, 0., 1.],
                    [0.82107767, 0.41262027, 0.20581485, 0., 1.],
                    [0.0107816 , 0.82107767, 0.41262027, 0., 1.],
                    [0.94951918, 0.0107816 , 0.82107767, 0., 1.]]
                 ),
            columns = ['lag_1', 'lag_2', 'lag_3', 'series_1', 'series_2']
          )

y_train_index = pd.DatetimeIndex(
                    ['2022-01-07', '2022-01-07', '2022-01-08', '2022-01-08',
                    '2022-01-09', '2022-01-09', '2022-01-10', '2022-01-10',
                    '2022-01-11', '2022-01-11', '2022-01-12', '2022-01-12',
                    '2022-01-13', '2022-01-13'],
                    dtype='datetime64[ns]', freq=None
                )

def test_create_sample_weights_output(X_train=X_train):
    """
    Test sample_weights creation
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor   = LinearRegression,
                    weight_func = custom_weights,
                    lags        = 3
                )

    expected = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    results = forecaster.create_sample_weights(series=series, X_train=X_train, y_train_index=y_train_index)
    assert np.array_equal(results, expected)


def test_create_sample_weights_exceptions_when_weights_has_nan(X_train=X_train):
    """
    Test sample_weights exception when weights contains NaNs.
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor   = LinearRegression,
                    weight_func = custom_weights_nan,
                    lags        = 3
                )

    err_msg = re.escape("NaN values in in Weights")
    with pytest.raises(Exception, match=err_msg):
        forecaster.create_sample_weights(series=series, X_train=X_train, y_train_index=y_train_index)
    

def test_create_sample_weights_exceptions_when_weights_all_zeros(X_train=X_train):
    """
    Test sample_weights exception when all weights are zeros.
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor   = LinearRegression,
                    weight_func = custom_weights_zeros,
                    lags        = 3
                )
    
    err_msg = re.escape("Weights sum to zero, can't be normalized")
    with pytest.raises(Exception, match=err_msg):
        forecaster.create_sample_weights(series=series, X_train=X_train, y_train_index=y_train_index)
