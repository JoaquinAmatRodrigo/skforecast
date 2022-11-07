# Unit test create_train_X_y ForecasterAutoregMultiSeries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def test_create_train_X_y_exception_when_series_not_dataframe():
    """
    Test exception is raised when series is not a pandas DataFrame.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    series = pd.Series(np.arange(7))

    err_msg = re.escape(f'`series` must be a pandas DataFrame. Got {type(series)}.')
    with pytest.raises(TypeError, match = err_msg):
        forecaster.create_train_X_y(series=series)


def test_create_train_X_y_exception_when_levels_of_transformer_series_not_equal_to_series_levels():
    """
    Test exception is raised when `transformer_series` is a dict and its keys are
    not the same as forecaster.series_levels.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)),  
                           '2': pd.Series(np.arange(5))
                           })
    series_levels = list(series.columns)

    dict_transformers = {'1': StandardScaler(), 
                         '3': StandardScaler()
                        }
    forecaster = ForecasterAutoregMultiSeries(regressor          = LinearRegression(), 
                                              lags               = 3,
                                              transformer_series = dict_transformers)
    
    err_msg = re.escape(
                    (f'When `transformer_series` parameter is a `dict`, its keys '
                     f'must be the same as `series` column names.\n'
                     f'    `transformer_series` keys : {list(forecaster.transformer_series_.keys())}.\n'
                     f'    `series` columns          : {series_levels}.')
                )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.create_train_X_y(series=series)


@pytest.mark.parametrize("exog", [pd.Series(np.arange(10)), pd.DataFrame(np.arange(50).reshape(25, 2))])
def test_create_train_X_y_exception_when_series_and_exog_have_different_length(exog):
    """
    Test exception is raised when length of series is not equal to length exog.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(7)), 
                           '2': pd.Series(np.arange(7))
                          })
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5)
    
    err_msg = re.escape(
                (f'`exog` must have same number of samples as `series`. '
                 f'length `exog`: ({len(exog)}), length `series`: ({len(series)})')
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.create_train_X_y(series=series, exog=exog)


def test_create_train_X_y_exception_when_series_and_exog_have_different_index():
    """
    Test exception is raised when series and exog have different index.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5)
    series = pd.DataFrame({'1': pd.Series(np.arange(7)), 
                           '2': pd.Series(np.arange(7))
                          })

    series.index = pd.date_range(start='2022-01-01', periods=7, freq='1D')

    err_msg = re.escape(
                ('Different index for `series` and `exog`. They must be equal '
                 'to ensure the correct alignment of values.')
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.fit(
            series = series,
            exog   = pd.Series(np.arange(7), index=pd.RangeIndex(start=0, stop=7, step=1))
        )


def test_create_train_X_y_output_when_series_and_exog_is_None():
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is None.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    series = pd.DataFrame({'1': pd.Series(np.arange(7, dtype=float)), 
                           '2': pd.Series(np.arange(7, dtype=float))
                          })

    results = forecaster.create_train_X_y(series=series)
    expected = (pd.DataFrame(
                    data = np.array([[2.0, 1.0, 0.0, 1., 0.],
                                     [3.0, 2.0, 1.0, 1., 0.],
                                     [4.0, 3.0, 2.0, 1., 0.],
                                     [5.0, 4.0, 3.0, 1., 0.],
                                     [2.0, 1.0, 0.0, 0., 1.],
                                     [3.0, 2.0, 1.0, 0., 1.],
                                     [4.0, 3.0, 2.0, 0., 1.],
                                     [5.0, 4.0, 3.0, 0., 1.]]),
                    index   = np.array([0, 1, 2, 3, 4, 5, 6, 7]),
                    columns = ['lag_1', 'lag_2', 'lag_3', '1', '2']
                ),
                pd.Series(
                    data  = np.array([3., 4., 5., 6., 3., 4., 5., 6.]),
                    index = np.array([0, 1, 2, 3, 4, 5, 6, 7]),
                    name  = 'y'
                ),
                pd.RangeIndex(start=0, stop=len(series), step=1
                ),
                pd.Index(np.array([3., 4., 5., 6., 3., 4., 5., 6.])
                )
               )

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            assert (results[i] == expected[i]).all()


def test_create_train_X_y_output_when_series_and_exog_is_pandas_series():
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is a pandas series.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    series = pd.DataFrame({'1': pd.Series(np.arange(7, dtype=float)), 
                           '2': pd.Series(np.arange(7, dtype=float))
                          })

    results = forecaster.create_train_X_y(
                  series = series,
                  exog   = pd.Series(np.arange(100, 107, dtype=float), name='exog')            
              )

    expected = (pd.DataFrame(
                    data = np.array([[2.0, 1.0, 0.0, 103., 1., 0.],
                                     [3.0, 2.0, 1.0, 104., 1., 0.],
                                     [4.0, 3.0, 2.0, 105., 1., 0.],
                                     [5.0, 4.0, 3.0, 106., 1., 0.],
                                     [2.0, 1.0, 0.0, 103., 0., 1.],
                                     [3.0, 2.0, 1.0, 104., 0., 1.],
                                     [4.0, 3.0, 2.0, 105., 0., 1.],
                                     [5.0, 4.0, 3.0, 106., 0., 1.]]),
                    index   = np.array([0, 1, 2, 3, 4, 5, 6, 7]),
                    columns = ['lag_1', 'lag_2', 'lag_3', 'exog', '1', '2']
                ),
                pd.Series(
                    data  = np.array([3., 4., 5., 6., 3., 4., 5., 6.]),
                    index = np.array([0, 1, 2, 3, 4, 5, 6, 7]),
                    name  = 'y'),
                pd.RangeIndex(start=0, stop=len(series), step=1
                ),
                pd.Index(np.array([3, 4, 5, 6, 3, 4, 5, 6])
                )
               )

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            assert (results[i] == expected[i]).all()


def test_create_train_X_y_output_when_series_and_exog_is_dataframe():
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is a pandas dataframe with two columns.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    series = pd.DataFrame({'1': pd.Series(np.arange(7, dtype=float)), 
                           '2': pd.Series(np.arange(7, dtype=float))
                          })

    results = forecaster.create_train_X_y(
                            series = series,
                            exog = pd.DataFrame({
                                      'exog_1' : np.arange(100, 107, dtype=float),
                                      'exog_2' : np.arange(1000, 1007, dtype=float)
                                   })           
              )

    expected = (pd.DataFrame(
                    data = np.array([[2.0, 1.0, 0.0, 103., 1003., 1., 0.],
                                     [3.0, 2.0, 1.0, 104., 1004., 1., 0.],
                                     [4.0, 3.0, 2.0, 105., 1005., 1., 0.],
                                     [5.0, 4.0, 3.0, 106., 1006., 1., 0.],
                                     [2.0, 1.0, 0.0, 103., 1003., 0., 1.],
                                     [3.0, 2.0, 1.0, 104., 1004., 0., 1.],
                                     [4.0, 3.0, 2.0, 105., 1005., 0., 1.],
                                     [5.0, 4.0, 3.0, 106., 1006., 0., 1.]]),
                    index   = np.array([0, 1, 2, 3, 4, 5, 6, 7]),
                    columns = ['lag_1', 'lag_2', 'lag_3', 'exog_1', 'exog_2', '1', '2']
                ),
                pd.Series(
                    data  = np.array([3., 4., 5., 6., 3., 4., 5., 6.]),
                    index = np.array([0, 1, 2, 3, 4, 5, 6, 7]),
                    name  = 'y'),
                pd.RangeIndex(start=0, stop=len(series), step=1
                ),
                pd.Index(np.array([3, 4, 5, 6, 3, 4, 5, 6])
                )
               )

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            assert (results[i] == expected[i]).all()


def test_create_train_X_y_output_when_series_and_exog_is_dataframe_datetime_index():
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is a pandas dataframe with two columns and datetime index.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    series = pd.DataFrame({'1': np.arange(7, dtype=float), 
                           '2': np.arange(7, dtype=float)},
                           index = pd.date_range("1990-01-01", periods=7, freq='D')
                          )

    results = forecaster.create_train_X_y(
                            series = series,
                            exog = pd.DataFrame(
                                       {'exog_1' : np.arange(100, 107, dtype=float),
                                        'exog_2' : np.arange(1000, 1007, dtype=float)},
                                       index = pd.date_range("1990-01-01", periods=7, freq='D')
                                   )           
              )

    expected = (pd.DataFrame(
                    data = np.array([[2.0, 1.0, 0.0, 103., 1003., 1., 0.],
                                     [3.0, 2.0, 1.0, 104., 1004., 1., 0.],
                                     [4.0, 3.0, 2.0, 105., 1005., 1., 0.],
                                     [5.0, 4.0, 3.0, 106., 1006., 1., 0.],
                                     [2.0, 1.0, 0.0, 103., 1003., 0., 1.],
                                     [3.0, 2.0, 1.0, 104., 1004., 0., 1.],
                                     [4.0, 3.0, 2.0, 105., 1005., 0., 1.],
                                     [5.0, 4.0, 3.0, 106., 1006., 0., 1.]]),
                    index   = np.array([0, 1, 2, 3, 4, 5, 6, 7]),
                    columns = ['lag_1', 'lag_2', 'lag_3', 'exog_1', 'exog_2', '1', '2']
                ),
                pd.Series(
                    data  = np.array([3., 4., 5., 6., 3., 4., 5., 6.]),
                    index = np.array([0, 1, 2, 3, 4, 5, 6, 7]),
                    name  = 'y'),
                pd.date_range("1990-01-01", periods=7, freq='D'
                ),
                pd.Index(pd.DatetimeIndex(['1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07', 
                                           '1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07'])
                )
               )

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            assert (results[i] == expected[i]).all()