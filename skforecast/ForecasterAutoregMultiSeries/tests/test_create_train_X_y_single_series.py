# Unit test _create_train_X_y_single_series ForecasterAutoregMultiSeries
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries


@pytest.mark.parametrize("ignore_exog", 
                         [True, False], 
                         ids = lambda ie: f'ignore_exog: {ie}')
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
               'exog_1': np.arange(100, 107, dtype=float),
               'exog_2': np.arange(1000, 1007, dtype=float)},
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


def test_create_train_X_y_single_series_output_when_series_and_exog_is_DataFrame_with_NaNs():
    """
    Test the output of _create_train_X_y_single_series when series and 
    exog is a pandas dataframe with two columns and NaNs in between.
    """
    y = pd.Series(np.arange(10, dtype=float), name='l1')
    y.iloc[6:8] = np.nan
    exog = pd.DataFrame({
               'exog_1': np.arange(100, 110, dtype=float),
               'exog_2': np.arange(1000, 1010, dtype=float)}
           )
    exog.iloc[2:7, 0] = np.nan

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
                             [np.nan, 5., 4., 3., 2., 'l1'],
                             [np.nan, np.nan, 5., 4., 3., 'l1'],
                             [8., np.nan, np.nan, 5., 4., 'l1']]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       '_level_skforecast']
        ).astype({'lag_1': float, 'lag_2': float, 'lag_3': float,
                  'lag_4': float, 'lag_5': float}),
        pd.DataFrame(
            data = np.array([[np.nan, 1005.],
                             [np.nan, 1006.],
                             [107., 1007.],
                             [108., 1008.],
                             [109., 1009.]]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['exog_1', 'exog_2']
        ),
        pd.Series(
            data  = np.array([5., np.nan, np.nan, 8., 9.], dtype=float),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y'
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_frame_equal(results[1], expected[1])
    pd.testing.assert_series_equal(results[2], expected[2])


def test_create_train_X_y_single_series_output_when_transformer_and_fitted():
    """
    Test the output of _create_train_X_y_single_series when Forecaster as
    already been fitted, transformer is MinMaxScaler() and has a different 
    series as input.
    """
    y = pd.Series(np.arange(9, dtype=float), name='l1')
    transformer = MinMaxScaler()
    transformer.fit(y.values.reshape(-1, 1))

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.transformer_series_ = {'l1': transformer}
    forecaster.fitted = True

    new_y = pd.Series(np.arange(10, 19, dtype=float), name='l1')
    results = forecaster._create_train_X_y_single_series(y=new_y, ignore_exog=True)
    
    expected = (
        pd.DataFrame(
            data = np.array([[1.5  , 1.375, 1.25 , 'l1'],
                             [1.625, 1.5  , 1.375, 'l1'],
                             [1.75 , 1.625, 1.5  , 'l1'],
                             [1.875, 1.75 , 1.625, 'l1'],
                             [2.   , 1.875, 1.75 , 'l1'],
                             [2.125, 2.   , 1.875, 'l1']]),
            index   = pd.RangeIndex(start=3, stop=9, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', '_level_skforecast']
        ).astype({'lag_1': float, 'lag_2': float, 'lag_3': float}),
        None,
        pd.Series(
            data  = np.array([1.625, 1.75, 1.875, 2., 2.125, 2.25]),
            index = pd.RangeIndex(start=3, stop=9, step=1),
            name  = 'y',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], type(None))
    pd.testing.assert_series_equal(results[2], expected[2])


@pytest.mark.parametrize("fitted", 
                         [True, False], 
                         ids = lambda fitted : f'fitted: {fitted}')
def test_create_train_X_y_single_series_output_when_series_and_exog_and_differentitation_1(fitted):
    """
    Test the output of _create_train_X_y_single_series when exog is a 
    pandas DataFrame with one column and differentiation=1.
    """
    y = pd.Series(np.arange(10, dtype=float), name='l1')
    exog = pd.DataFrame(np.arange(100, 110, dtype=float), columns=['exog'])
    
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5,
                                              transformer_series = None,
                                              differentiation    = 1)
    forecaster.transformer_series_ = {'l1': None}
    forecaster.differentiator_ = {'l1': clone(forecaster.differentiator)}
    forecaster.fitted = fitted

    results = forecaster._create_train_X_y_single_series(
                  y           = y,
                  ignore_exog = False,
                  exog        = exog
              )

    expected = (
        pd.DataFrame(
            data = np.array([[1., 1., 1., 1., 1., 'l1'],
                             [1., 1., 1., 1., 1., 'l1'],
                             [1., 1., 1., 1., 1., 'l1'],
                             [1., 1., 1., 1., 1., 'l1']]),
            index   = pd.RangeIndex(start=6, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       '_level_skforecast']
        ).astype({'lag_1': float, 'lag_2': float, 'lag_3': float,
                  'lag_4': float, 'lag_5': float}),
        pd.DataFrame(
            data  = np.array([106., 107., 108., 109.], dtype=float),
            index = pd.RangeIndex(start=6, stop=10, step=1),
            columns = ['exog']
        ),
        pd.Series(
            data  = np.array([1., 1., 1., 1.], dtype=float),
            index = pd.RangeIndex(start=6, stop=10, step=1),
            name  = 'y'
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_frame_equal(results[1], expected[1])
    pd.testing.assert_series_equal(results[2], expected[2])


@pytest.mark.parametrize("fitted", 
                         [True, False], 
                         ids = lambda fitted: f'fitted: {fitted}')
def test_create_train_X_y_single_series_output_when_series_and_exog_and_differentitation_2(fitted):
    """
    Test the output of _create_train_X_y_single_series when exog is a 
    pandas DataFrame with one column and differentiation=2.
    """
    y = pd.Series(np.arange(10, dtype=float), name='l1')
    exog = pd.DataFrame(np.arange(100, 110, dtype=float), columns=['exog'])
    
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5,
                                              transformer_series = None,
                                              differentiation    = 2)
    forecaster.transformer_series_ = {'l1': None}
    forecaster.differentiator_ = {'l1': clone(forecaster.differentiator)}
    forecaster.fitted = fitted

    results = forecaster._create_train_X_y_single_series(
                  y           = y,
                  ignore_exog = False,
                  exog        = exog
              )

    expected = (
        pd.DataFrame(
            data = np.array([[0., 0., 0., 0., 0., 'l1'],
                             [0., 0., 0., 0., 0., 'l1'],
                             [0., 0., 0., 0., 0., 'l1']]),
            index   = pd.RangeIndex(start=7, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       '_level_skforecast']
        ).astype({'lag_1': float, 'lag_2': float, 'lag_3': float,
                  'lag_4': float, 'lag_5': float}),
        pd.DataFrame(
            data  = np.array([107., 108., 109.], dtype=float),
            index = pd.RangeIndex(start=7, stop=10, step=1),
            columns = ['exog']
        ),
        pd.Series(
            data  = np.array([0., 0., 0.], dtype=float),
            index = pd.RangeIndex(start=7, stop=10, step=1),
            name  = 'y'
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_frame_equal(results[1], expected[1])
    pd.testing.assert_series_equal(results[2], expected[2])
