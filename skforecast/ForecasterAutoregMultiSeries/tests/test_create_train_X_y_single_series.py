# Unit test _create_train_X_y_single_series ForecasterAutoregMultiSeries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.exceptions import MissingValuesExogWarning
from skforecast.exceptions import IgnoredArgumentWarning
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


@pytest.mark.parametrize("ignore_exog", 
                         [True, False], 
                         ids = lambda ie : f'ignore_exog: {ie}')
def test_create_train_X_y_single_series_output_when_series_and_exog_is_None(ignore_exog):
    """
    Test the output of _create_train_X_y_single_series when exog is None. 
    Check cases `ignore_exog` True and False.
    """
    y = pd.Series(np.arange(7, dtype=float), name='l1')
    
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.transformer_series_ = {'l1': StandardScaler()}
    results = forecaster._create_train_X_y_single_series(
                  y           = y,
                  ignore_exog = ignore_exog,
                  exog        = None
              )
    
    expected = (
        pd.DataFrame(
            data = np.array([[-0.5, -1. , -1.5, 'l1'],
                             [ 0. , -0.5, -1. , 'l1'],
                             [ 0.5,  0. , -0.5, 'l1'],
                             [ 1. ,  0.5,  0. , 'l1']]),
            index   = pd.RangeIndex(start=3, stop=7, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', '_level_skforecast']
        ).astype({'lag_1': float, 'lag_2': float, 'lag_3': float}),
        pd.DataFrame(
            data    = np.nan,
            columns = ['_dummy_exog_col_to_keep_shape'],
            index   = pd.RangeIndex(start=3, stop=7, step=1)
        ),
        pd.Series(
            data  = np.array([0., 0.5, 1., 1.5]),
            index = pd.RangeIndex(start=3, stop=7, step=1),
            name  = 'y',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    if ignore_exog:
        assert isinstance(results[1], type(None))
    else:
        pd.testing.assert_frame_equal(results[1], expected[1])
    pd.testing.assert_series_equal(results[2], expected[2])
        

def test_create_train_X_y_single_series_output_when_series_and_exog():
    """
    Test the output of _create_train_X_y_single_series when exog is a 
    pandas DataFrame with one column.
    """
    y = pd.Series(np.arange(10, dtype=float), name='l1')
    exog = pd.DataFrame(np.arange(100, 110, dtype=float), columns=['exog'])
    
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5,
                                              transformer_series=None)
    forecaster.transformer_series_ = {'l1': None}
    results = forecaster._create_train_X_y_single_series(
                  y           = y,
                  ignore_exog = False,
                  exog        = exog
              )

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 'l1'],
                             [5., 4., 3., 2., 1., 'l1'],
                             [6., 5., 4., 3., 2., 'l1'],
                             [7., 6., 5., 4., 3., 'l1'],
                             [8., 7., 6., 5., 4., 'l1']]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       '_level_skforecast']
        ).astype({'lag_1': float, 'lag_2': float, 'lag_3': float,
                  'lag_4': float, 'lag_5': float}),
        pd.DataFrame(
            data  = np.array([105., 106., 107., 108., 109.], dtype=float),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['exog']
        ),
        pd.Series(
            data  = np.array([5., 6., 7., 8., 9.], dtype=float),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y'
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_frame_equal(results[1], expected[1])
    pd.testing.assert_series_equal(results[2], expected[2])


def test_create_train_X_y_single_series_output_when_series_10_and_exog_is_dataframe_of_category():
    """
    Test the output of _create_train_X_y_single_series when exog is a 
    pandas DataFrame with one column float and one column of cateogry.
    """
    y = pd.Series(np.arange(10, dtype=float), name='l1')
    exog = pd.DataFrame({'exog_1': np.arange(10, 20, dtype=float),
                         'exog_2': pd.Categorical(range(100, 110))})

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5,
                                              transformer_series=None)
    forecaster.transformer_series_ = {'l1': None}
    results = forecaster._create_train_X_y_single_series(
                  y           = y,
                  ignore_exog = False,
                  exog        = exog
              )

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 'l1'],
                             [5., 4., 3., 2., 1., 'l1'],
                             [6., 5., 4., 3., 2., 'l1'],
                             [7., 6., 5., 4., 3., 'l1'],
                             [8., 7., 6., 5., 4., 'l1']]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       '_level_skforecast']
        ).astype({'lag_1': float, 'lag_2': float, 'lag_3': float,
                  'lag_4': float, 'lag_5': float}),
        pd.DataFrame(
            data  = np.array([15., 16., 17., 18., 19.], dtype=float),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['exog_1']
        ).assign(exog_2 = pd.Categorical([105, 106, 107, 108, 109], categories=range(100, 110))),
        pd.Series(
            data  = np.array([5., 6., 7., 8., 9.], dtype=float),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y'
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_frame_equal(results[1], expected[1])
    pd.testing.assert_series_equal(results[2], expected[2])


def test_create_train_X_y_single_series_output_when_series_and_exog_is_DataFrame_datetime_index():
    """
    Test the output of _create_train_X_y_single_series when series and 
    exog is a pandas dataframe with two columns and datetime index.
    """
    y = pd.Series(np.arange(7, dtype=float), name='l1')
    y.index = pd.date_range("1990-01-01", periods=7, freq='D')
    exog = pd.DataFrame({
               'exog_1' : np.arange(100, 107, dtype=float),
               'exog_2' : np.arange(1000, 1007, dtype=float)},
               index = pd.date_range("1990-01-01", periods=7, freq='D')
           )

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.transformer_series_ = {'l1': None}
    results = forecaster._create_train_X_y_single_series(
                  y           = y,
                  ignore_exog = False,
                  exog        = exog
              )
    
    expected = (
        pd.DataFrame(
            data = np.array([[2.0, 1.0, 0.0, 'l1'],
                             [3.0, 2.0, 1.0, 'l1'],
                             [4.0, 3.0, 2.0, 'l1'],
                             [5.0, 4.0, 3.0, 'l1']]),
            index   = pd.date_range("1990-01-04", periods=4, freq='D'),
            columns = ['lag_1', 'lag_2', 'lag_3', '_level_skforecast']
        ).astype({'lag_1': float, 'lag_2': float, 'lag_3': float}),
        pd.DataFrame(
            data = np.array([[103., 1003.],
                             [104., 1004.],
                             [105., 1005.],
                             [106., 1006.]]),
            index   = pd.date_range("1990-01-04", periods=4, freq='D'),
            columns = ['exog_1', 'exog_2']
        ),
        pd.Series(
            data  = np.array([3., 4., 5., 6.], dtype=float),
            index = pd.date_range("1990-01-04", periods=4, freq='D'),
            name  = 'y'
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_frame_equal(results[1], expected[1])
    pd.testing.assert_series_equal(results[2], expected[2])


# TODO: CONTINUE FROM HERE
# ==============================================================================


def test_create_train_X_y_output_when_series_10_and_transformer_series_is_StandardScaler():
    """
    Test the output of create_train_X_y when exog is None and transformer_series
    is StandardScaler.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    forecaster = ForecasterAutoregMultiSeries(
                    regressor          = LinearRegression(),
                    lags               = 5,
                    transformer_series = StandardScaler()
                )
    results = forecaster.create_train_X_y(series=series)
    expected = (
        pd.DataFrame(
            data = np.array([
                       [-0.17407766, -0.52223297, -0.87038828, -1.21854359, -1.5666989 , 1.,  0.],
                       [ 0.17407766, -0.17407766, -0.52223297, -0.87038828, -1.21854359, 1.,  0.],
                       [ 0.52223297,  0.17407766, -0.17407766, -0.52223297, -0.87038828, 1.,  0.],
                       [ 0.87038828,  0.52223297,  0.17407766, -0.17407766, -0.52223297, 1.,  0.],
                       [ 1.21854359,  0.87038828,  0.52223297,  0.17407766, -0.17407766, 1.,  0.],
                       [-0.17407766, -0.52223297, -0.87038828, -1.21854359, -1.5666989 , 0.,  1.],
                       [ 0.17407766, -0.17407766, -0.52223297, -0.87038828, -1.21854359, 0.,  1.],
                       [ 0.52223297,  0.17407766, -0.17407766, -0.52223297, -0.87038828, 0.,  1.],
                       [ 0.87038828,  0.52223297,  0.17407766, -0.17407766, -0.52223297, 0.,  1.],
                       [ 1.21854359,  0.87038828,  0.52223297,  0.17407766, -0.17407766, 0.,  1.]]),
            index   = pd.RangeIndex(start=0, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'l1', 'l2']
        ),
        pd.Series(
            data  = np.array([0.17407766, 0.52223297, 0.87038828, 1.21854359, 1.5666989 ,
                              0.17407766, 0.52223297, 0.87038828, 1.21854359, 1.5666989 ]),
            index = pd.RangeIndex(start=0, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        pd.RangeIndex(start=0, stop=len(series), step=1),
        pd.Index(np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]))
    )

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            np.testing.assert_array_equal(results[i], expected[i])


def test_create_train_X_y_output_when_exog_is_None_and_transformer_exog_is_not_None():
    """
    Test the output of create_train_X_y when exog is None and transformer_exog
    is not None.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(7, dtype=float)), 
                           '2': pd.Series(np.arange(7, dtype=float))})
    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     transformer_series = None,
                     transformer_exog   = StandardScaler()
                 )

    results = forecaster.create_train_X_y(series=series)
    expected = (
        pd.DataFrame(
            data = np.array([[2.0, 1.0, 0.0, 1., 0.],
                             [3.0, 2.0, 1.0, 1., 0.],
                             [4.0, 3.0, 2.0, 1., 0.],
                             [5.0, 4.0, 3.0, 1., 0.],
                             [2.0, 1.0, 0.0, 0., 1.],
                             [3.0, 2.0, 1.0, 0., 1.],
                             [4.0, 3.0, 2.0, 0., 1.],
                             [5.0, 4.0, 3.0, 0., 1.]]),
            index   = pd.RangeIndex(start=0, stop=8, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', '1', '2']
        ),
        pd.Series(
            data  = np.array([3., 4., 5., 6., 3., 4., 5., 6.]),
            index = pd.RangeIndex(start=0, stop=8, step=1),
            name  = 'y',
            dtype = float
        ),
        pd.RangeIndex(start=0, stop=len(series), step=1),
        pd.Index(np.array([3., 4., 5., 6., 3., 4., 5., 6.]))
    )

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            np.testing.assert_array_equal(results[i], expected[i])


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(),
                          {'1': StandardScaler(), '2': StandardScaler()}], 
                         ids = lambda tr : f'transformer_series type: {type(tr)}')
def test_create_train_X_y_output_when_transformer_series_and_transformer_exog(transformer_series):
    """
    Test the output of create_train_X_y when using transformer_series and 
    transformer_exog.
    """
    series = pd.DataFrame({'1': np.arange(10, dtype=float), 
                           '2': np.arange(10, dtype=float)},
                           index = pd.date_range("1990-01-01", periods=10, freq='D'))
    exog = pd.DataFrame({
               'exog_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 24.4, 87.2, 47.4, 23.8],
               'exog_2': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b']},
                index = pd.date_range("1990-01-01", periods=10, freq='D'))

    transformer_exog = ColumnTransformer(
                            [('scale', StandardScaler(), ['exog_1']),
                             ('onehot', OneHotEncoder(), ['exog_2'])],
                            remainder = 'passthrough',
                            verbose_feature_names_out = False
                        )

    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     transformer_series = transformer_series,
                     transformer_exog   = transformer_exog
                 )
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([
                       [-0.87038828, -1.21854359, -1.5666989 ,  0.67431975, 1., 0., 1., 0.],
                       [-0.52223297, -0.87038828, -1.21854359,  0.37482376, 1., 0., 1., 0.],
                       [-0.17407766, -0.52223297, -0.87038828, -0.04719331, 0., 1., 1., 0.],
                       [ 0.17407766, -0.17407766, -0.52223297, -0.81862236, 0., 1., 1., 0.],
                       [ 0.52223297,  0.17407766, -0.17407766,  2.03112731, 0., 1., 1., 0.],
                       [ 0.87038828,  0.52223297,  0.17407766,  0.22507577, 0., 1., 1., 0.],
                       [ 1.21854359,  0.87038828,  0.52223297, -0.84584926, 0., 1., 1., 0.],
                       [-0.87038828, -1.21854359, -1.5666989 ,  0.67431975, 1., 0., 0., 1.],
                       [-0.52223297, -0.87038828, -1.21854359,  0.37482376, 1., 0., 0., 1.],
                       [-0.17407766, -0.52223297, -0.87038828, -0.04719331, 0., 1., 0., 1.],
                       [ 0.17407766, -0.17407766, -0.52223297, -0.81862236, 0., 1., 0., 1.],
                       [ 0.52223297,  0.17407766, -0.17407766,  2.03112731, 0., 1., 0., 1.],
                       [ 0.87038828,  0.52223297,  0.17407766,  0.22507577, 0., 1., 0., 1.],
                       [ 1.21854359,  0.87038828,  0.52223297, -0.84584926, 0., 1., 0., 1.]]),
            index   = pd.RangeIndex(start=0, stop=14, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'exog_1',
                       'exog_2_a', 'exog_2_b', '1', '2']
        ),
        pd.Series(
            data  = np.array([-0.52223297, -0.17407766,  0.17407766,  0.52223297,  0.87038828,
                               1.21854359,  1.5666989 , -0.52223297, -0.17407766,  0.17407766,
                               0.52223297,  0.87038828,  1.21854359,  1.5666989 ]),
            index = pd.RangeIndex(start=0, stop=14, step=1),
            name  = 'y',
            dtype = float
        ),
        pd.date_range("1990-01-01", periods=10, freq='D'),
        pd.Index(pd.DatetimeIndex(['1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07', 
                                   '1990-01-08', '1990-01-09', '1990-01-10', '1990-01-04',
                                   '1990-01-05', '1990-01-06', '1990-01-07', '1990-01-08',
                                   '1990-01-09', '1990-01-10']))
    )

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            np.testing.assert_array_equal(results[i], expected[i])


def test_create_train_X_y_output_when_series_different_length_and_exog_is_dataframe_of_float_int_category():
    """
    Test the output of create_train_X_y when series has 2 columns with different 
    lengths and exog is a pandas dataframe with two columns of float, int, category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series([np.nan, np.nan, 2., 3., 4., 5., 6., 7., 8., 9.])})
    series.index = pd.date_range("1990-01-01", periods=10, freq='D')
    exog = pd.DataFrame({'exog_1': pd.Series(np.arange(100, 110), dtype=float),
                         'exog_2': pd.Series(np.arange(1000, 1010), dtype=int),
                         'exog_3': pd.Categorical(range(100, 110))})
    exog.index = pd.date_range("1990-01-01", periods=10, freq='D')

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5,
                                              transformer_series=None)
    results = forecaster.create_train_X_y(series=series, exog=exog)   

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105., 1005.],
                             [5., 4., 3., 2., 1., 106., 1006.],
                             [6., 5., 4., 3., 2., 107., 1007.],
                             [7., 6., 5., 4., 3., 108., 1008.],
                             [8., 7., 6., 5., 4., 109., 1009.],
                             [6., 5., 4., 3., 2., 107., 1007.],
                             [7., 6., 5., 4., 3., 108., 1008.],
                             [8., 7., 6., 5., 4., 109., 1009.]],
                             dtype=float),
            index   = pd.RangeIndex(start=0, stop=8, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'exog_1', 'exog_2']
        ).assign(exog_3 = pd.Categorical([105, 106, 107, 108, 109, 
                                          107, 108, 109], categories=range(100, 110)), 
                 l1     = [1.]*5 + [0.]*3, 
                 l2     = [0.]*5 + [1.]*3
        ).astype({'exog_1': float, 
                  'exog_2': int}
        ),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 7, 8, 9]),
            index = pd.RangeIndex(start=0, stop=8, step=1),
            name  = 'y',
            dtype = float
        ),
        pd.date_range("1990-01-01", periods=10, freq='D'),
        pd.DatetimeIndex(['1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10',
                          '1990-01-08', '1990-01-09', '1990-01-10'],
                         dtype='datetime64[ns]', freq=None
        )
    )

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            np.testing.assert_array_equal(results[i], expected[i])


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(),
                          {'l1': StandardScaler(), 'l2': StandardScaler(), 'l3': StandardScaler()}], 
                         ids = lambda tr : f'transformer_series type: {type(tr)}')
def test_create_train_X_y_output_when_transformer_series_and_transformer_exog_with_different_series_lengths(transformer_series):
    """
    Test the output of create_train_X_y when using transformer_series and 
    transformer_exog with series with different lengths.
    """
    series = pd.DataFrame({'l1': np.arange(10, dtype=float), 
                           'l2': pd.Series([np.nan, np.nan, 
                                            2., 3., 4., 5., 6., 7., 8., 9.]), 
                           'l3': pd.Series([np.nan, np.nan, np.nan, np.nan, 
                                            4., 5., 6., 7., 8., 9.])})
    series.index = pd.date_range("1990-01-01", periods=10, freq='D')
    exog = pd.DataFrame({
               'exog_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 24.4, 87.2, 47.4, 23.8],
               'exog_2': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b']},
                index = pd.date_range("1990-01-01", periods=10, freq='D'))

    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )

    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     transformer_series = transformer_series,
                     transformer_exog   = transformer_exog
                 )
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([
                       [-0.8703882797784892,  -1.2185435916898848,  -1.5666989036012806,   0.6743197452466179,  0.0, 1.0, 1.0, 0.0, 0.0],
                       [-0.5222329678670935,  -0.8703882797784892,  -1.2185435916898848,   0.3748237614897084,  1.0, 0.0, 1.0, 0.0, 0.0],
                       [-0.17407765595569785, -0.5222329678670935,  -0.8703882797784892,  -0.04719330653139179, 0.0, 1.0, 1.0, 0.0, 0.0],
                       [ 0.17407765595569785, -0.17407765595569785, -0.5222329678670935,  -0.8186223556022197,  1.0, 0.0, 1.0, 0.0, 0.0],
                       [ 0.5222329678670935,   0.17407765595569785, -0.17407765595569785,  2.0311273080241334,  0.0, 1.0, 1.0, 0.0, 0.0],
                       [ 0.8703882797784892,   0.5222329678670935,   0.17407765595569785,  0.2250757696112534,  1.0, 0.0, 1.0, 0.0, 0.0],
                       [ 1.2185435916898848,   0.8703882797784892,   0.5222329678670935,  -0.8458492632164842,  0.0, 1.0, 1.0, 0.0, 0.0],
                       [-0.6546536707079772,  -1.091089451179962,   -1.5275252316519468,  -0.04719330653139179, 0.0, 1.0, 0.0, 1.0, 0.0],
                       [-0.2182178902359924,  -0.6546536707079772,  -1.091089451179962,   -0.8186223556022197,  1.0, 0.0, 0.0, 1.0, 0.0],
                       [ 0.2182178902359924,  -0.2182178902359924,  -0.6546536707079772,   2.0311273080241334,  0.0, 1.0, 0.0, 1.0, 0.0],
                       [ 0.6546536707079772,   0.2182178902359924,  -0.2182178902359924,   0.2250757696112534,  1.0, 0.0, 0.0, 1.0, 0.0],
                       [ 1.091089451179962,    0.6546536707079772,   0.2182178902359924,  -0.8458492632164842,  0.0, 1.0, 0.0, 1.0, 0.0],
                       [-0.29277002188455997, -0.8783100656536799,  -1.4638501094227998,   2.0311273080241334,  0.0, 1.0, 0.0, 0.0, 1.0],
                       [ 0.29277002188455997, -0.29277002188455997, -0.8783100656536799,   0.2250757696112534,  1.0, 0.0, 0.0, 0.0, 1.0],
                       [ 0.8783100656536799,   0.29277002188455997, -0.29277002188455997, -0.8458492632164842,  0.0, 1.0, 0.0, 0.0, 1.0]]),
            index   = pd.RangeIndex(start=0, stop=15, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'exog_1',
                       'exog_2_a', 'exog_2_b', 'l1', 'l2', 'l3']
        ),
        pd.Series(
            data  = np.array([-0.5222329678670935, -0.17407765595569785, 0.17407765595569785, 0.5222329678670935, 0.8703882797784892, 1.2185435916898848, 1.5666989036012806, 
                              -0.2182178902359924, 0.2182178902359924, 0.6546536707079772, 1.091089451179962, 1.5275252316519468, 
                              0.29277002188455997, 0.8783100656536799, 1.4638501094227998]),
            index = pd.RangeIndex(start=0, stop=15, step=1),
            name  = 'y',
            dtype = float
        ),
        pd.date_range("1990-01-01", periods=10, freq='D'),
        pd.DatetimeIndex(['1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10',
                          '1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10',
                          '1990-01-08', '1990-01-09', '1990-01-10'],
                         dtype='datetime64[ns]', freq=None
        )
    )

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            np.testing.assert_array_equal(results[i], expected[i])