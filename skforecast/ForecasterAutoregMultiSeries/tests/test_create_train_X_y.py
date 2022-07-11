# Unit test create_train_X_y ForecasterAutoregMultiSeries
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.linear_model import LinearRegression


def test_create_train_X_y_exception_when_series_not_dataframe():
    '''
    Test exception is raised when series is not a pandas DataFrame.
    '''
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    serie = pd.Series(np.arange(7))

    with pytest.raises(Exception):
        forecaster.create_train_X_y(series=serie)


def test_create_train_X_y_output_when_series_and_exog_is_None():
    '''
    Test the output of create_train_X_y when series has 2 columns and 
    exog is None.
    '''
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    series = pd.DataFrame({'1': pd.Series(np.arange(7)), 
                           '2': pd.Series(np.arange(7))
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
                    np.array([3., 4., 5., 6., 3., 4., 5., 6.]),
                    index = np.array([0, 1, 2, 3, 4, 5, 6, 7]),
                    name = 'y')
               )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert (results[1] == expected[1]).all()


def test_create_train_X_y_output_when_series_10_and_exog_is_series():
    '''
    Test the output of create_train_X_y when series has 2 columns and 
    exog is a pandas series.
    '''
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    series = pd.DataFrame({'1': pd.Series(np.arange(7)), 
                           '2': pd.Series(np.arange(7))
                          })

    results = forecaster.create_train_X_y(
                            series = series,
                            exog   = pd.Series(np.arange(100, 107), name='exog')            
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
                    np.array([3., 4., 5., 6., 3., 4., 5., 6.]),
                    index = np.array([0, 1, 2, 3, 4, 5, 6, 7]),
                    name = 'y')
               )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert (results[1] == expected[1]).all()


def test_create_train_X_y_output_when_series_10_and_exog_is_dataframe():
    '''
    Test the output of create_train_X_y when series has 2 columns and 
    exog is a pandas dataframe with two columns.
    '''
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    series = pd.DataFrame({'1': pd.Series(np.arange(7)), 
                           '2': pd.Series(np.arange(7))
                          })

    results = forecaster.create_train_X_y(
                            series = series,
                            exog = pd.DataFrame({
                                    'exog_1' : np.arange(100, 107),
                                    'exog_2' : np.arange(1000, 1007)
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
                    np.array([3., 4., 5., 6., 3., 4., 5., 6.]),
                    index = np.array([0, 1, 2, 3, 4, 5, 6, 7]),
                    name = 'y')
               )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert (results[1] == expected[1]).all()


def test_create_train_X_y_exception_when_series_and_exog_have_different_length():
    '''
    Test exception is raised when length of series and length of exog are different.
    '''
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5)
    series = pd.DataFrame({'1': pd.Series(np.arange(7)), 
                           '2': pd.Series(np.arange(7))
                          })

    with pytest.raises(Exception):
        forecaster.fit(series=series, exog=pd.Series(np.arange(10)))
    with pytest.raises(Exception):
        forecaster.fit(
            series = series,
            exog   = pd.DataFrame(np.arange(50).reshape(25,2))
        )


def test_create_train_X_y_exception_when_series_and_exog_have_different_index():
    '''
    Test exception is raised when series and exog have different index.
    '''
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5)
    series = pd.DataFrame({'1': pd.Series(np.arange(7)), 
                           '2': pd.Series(np.arange(7))
                          })

    series.index = pd.date_range(start='2022-01-01', periods=7, freq='1D')

    with pytest.raises(Exception):
        forecaster.fit(
            series = series,
            exog   = pd.Series(np.arange(7), index=pd.RangeIndex(start=0, stop=7, step=1))
        )   