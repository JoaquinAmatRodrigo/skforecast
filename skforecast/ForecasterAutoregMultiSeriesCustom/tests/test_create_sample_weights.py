# Unit test create_sample_weights ForecasterAutoregMultiSeriesCustom
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeriesCustom import ForecasterAutoregMultiSeriesCustom
from sklearn.linear_model import LinearRegression


def custom_weights(index): # pragma: no cover
    """
    Return 0 if index is between '2022-01-08' and '2022-01-10', 1 otherwise.
    """
    weights = np.where(
                  (index >= '2022-01-08') & (index <= '2022-01-10'),
                   0,
                   1
              )
    
    return weights
    

def custom_weights_2(index): # pragma: no cover
    """
    Return 2 if index is between '2022-01-11' and '2022-01-13', 3 otherwise.
    """
    weights = np.where(
                  (index >= '2022-01-11') & (index <= '2022-01-13'),
                   2,
                   3
              )
    
    return weights


def custom_weights_nan(index): # pragma: no cover
    """
    Return np.nan if index is between '2022-01-08' and '2022-01-10', 1 otherwise.
    """
    weights = np.where(
                  (index >= '2022-01-08') & (index <= '2022-01-10'),
                   np.nan,
                   1
              )
    
    return weights


def custom_weights_negative(index): # pragma: no cover
    """
    Return -1 if index is between '2022-01-08' and '2022-01-10', 1 otherwise.
    """
    weights = np.where(
                  (index >= '2022-01-08') & (index <= '2022-01-10'),
                   -1,
                   1
              )
    
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
                    ['2022-01-07', '2022-01-08', '2022-01-09', '2022-01-10',
                     '2022-01-11', '2022-01-12', '2022-01-13',
                     '2022-01-07', '2022-01-08', '2022-01-09', '2022-01-10',
                     '2022-01-11', '2022-01-12', '2022-01-13'],
                     dtype='datetime64[ns]', freq=None
                )


def create_predictors(y): # pragma: no cover
    """
    Create first 3 lags of a time series.
    """
    lags = y[-1:-4:-1]

    return lags


def test_create_sample_weights_output_using_series_weights():
    """
    Test `sample_weights` creation using `series_weights`.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3,
                     series_weights  = {'series_1': 1., 'series_2': 2.}
                 )

    expected = np.array([1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2.])
    results = forecaster.create_sample_weights(series=series, X_train=X_train, 
                                               y_train_index=y_train_index)
    
    assert np.array_equal(results, expected)


def test_create_sample_weights_output_using_weight_func():
    """
    Test `sample_weights` creation using `weight_func`.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3,
                     weight_func     = custom_weights
                 )

    expected = np.array([1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1])
    results = forecaster.create_sample_weights(series=series, X_train=X_train, 
                                               y_train_index=y_train_index)
    
    assert np.array_equal(results, expected)


@pytest.mark.parametrize("weight_func, expected", 
                         [({'series_1': custom_weights}, 
                           np.array([1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])),                            
                          ({'series_2': custom_weights_2}, 
                           np.array([1., 1., 1., 1., 1., 1., 1., 3., 3., 3., 3., 2., 2., 2.])), 
                          ({'series_1': custom_weights, 
                            'series_2': custom_weights_2}, 
                           np.array([1, 0, 0, 0, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2]))], 
                         ids = lambda values : f'levels: {values}'
                        )
def test_create_sample_weights_output_using_weight_func_dict(weight_func, expected):
    """
    Test `sample_weights` creation using `weight_func`.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3,
                     weight_func     = weight_func
                 )

    results = forecaster.create_sample_weights(series=series, X_train=X_train, 
                                               y_train_index=y_train_index)
    
    assert np.array_equal(results, expected)


@pytest.mark.parametrize("weight_func, expected", 
                         [({'series_1': custom_weights}, 
                           np.array([1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1.,])), 
                          ({'series_2': custom_weights_2}, 
                           np.array([1., 1., 1., 1., 1., 1., 1., 3., 2., 2., 2.])), 
                          ({'series_1': custom_weights, 
                            'series_2': custom_weights_2}, 
                           np.array([1, 0, 0, 0, 1, 1, 1, 3, 2, 2, 2]))], 
                         ids = lambda values : f'levels: {values}')
def test_create_sample_weights_output_using_weight_func_dict_different_series_lengths(weight_func, expected):
    """
    Test `sample_weights` creation using `weight_func` with series of different lengths.
    """
    new_series = series.copy()
    new_series['series_2'].iloc[:3] = np.nan
    new_X_train = X_train.drop([7, 8, 9]).reset_index(drop=True)
    new_y_train_index = pd.DatetimeIndex(
                            ['2022-01-07', '2022-01-08', '2022-01-09', '2022-01-10',
                             '2022-01-11', '2022-01-12', '2022-01-13',
                             '2022-01-10', '2022-01-11', '2022-01-12', '2022-01-13'],
                            dtype='datetime64[ns]', freq=None
                        )
    
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3,
                     weight_func     = weight_func
                 )
    results = forecaster.create_sample_weights(series=new_series, X_train=new_X_train, 
                                               y_train_index=new_y_train_index)
    
    assert np.array_equal(results, expected)


def test_create_sample_weights_output_using_series_weights_and_weight_func():
    """
    Test `sample_weights` creation using `series_weights` and `weight_func`.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3,
                     series_weights  = {'series_1': 1., 'series_2': 2.},
                     weight_func     = custom_weights
                 )

    expected = np.array([1, 0, 0, 0, 1, 1, 1, 2, 0, 0, 0, 2, 2, 2], dtype=float)
    results = forecaster.create_sample_weights(series=series, X_train=X_train, 
                                               y_train_index=y_train_index)
    
    assert np.array_equal(results, expected)


def test_create_sample_weights_output_using_series_weights_and_weight_func_different_series_lengths():
    """
    Test `sample_weights` creation using `series_weights` and `weight_func` 
    with series of different lengths.
    """
    new_series = series.copy()
    new_series['series_2'].iloc[:3] = np.nan
    new_X_train = X_train.drop([7, 8, 9]).reset_index(drop=True)
    new_y_train_index = pd.DatetimeIndex(
                            ['2022-01-07', '2022-01-08', '2022-01-09', '2022-01-10',
                             '2022-01-11', '2022-01-12', '2022-01-13',
                             '2022-01-10', '2022-01-11', '2022-01-12', '2022-01-13'],
                            dtype='datetime64[ns]', freq=None
                        )
    
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3,
                     series_weights  = {'series_1': 1., 'series_2': 2.},
                     weight_func     = custom_weights
                 )
    results = forecaster.create_sample_weights(series=new_series, X_train=new_X_train, 
                                               y_train_index=new_y_train_index)

    expected = np.array([1, 0, 0, 0, 1, 1, 1, 0, 2, 2, 2], dtype=float)
    
    assert np.array_equal(results, expected)


def test_create_sample_weights_exceptions_when_weights_has_nan():
    """
    Test sample_weights exception when sample_weight contains NaNs.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3,
                     weight_func     = custom_weights_nan
                 )

    err_msg = re.escape("The resulting `weights` cannot have NaN values.")
    with pytest.raises(ValueError, match=err_msg):
        forecaster.create_sample_weights(series=series, X_train=X_train, 
                                         y_train_index=y_train_index)


def test_create_sample_weights_exceptions_when_weights_has_negative_values():
    """
    Test sample_weights exception when sample_weight contains negative values.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3,
                     weight_func     = custom_weights_negative
                 )

    err_msg = re.escape("The resulting `weights` cannot have negative values.")
    with pytest.raises(ValueError, match=err_msg):
        forecaster.create_sample_weights(series=series, X_train=X_train, 
                                         y_train_index=y_train_index)
   

def test_create_sample_weights_exceptions_when_weights_all_zeros():
    """
    Test sample_weights exception when the sum of the weights is zero.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors,
                     window_size     = 3,
                     series_weights  = {'series_1': 0, 'series_2': 0}
                 )
    
    err_msg = re.escape(
                    ("The resulting `weights` cannot be normalized because "
                     "the sum of the weights is zero.")
                )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.create_sample_weights(series=series, X_train=X_train, 
                                         y_train_index=y_train_index)