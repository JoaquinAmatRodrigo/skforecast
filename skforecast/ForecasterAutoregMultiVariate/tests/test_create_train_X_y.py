# Unit test create_train_X_y ForecasterAutoregMultiVariate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def test_create_train_X_y_exception_when_series_not_dataframe():
    """
    Test exception is raised when series is not a pandas DataFrame.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1', lags=3, steps=2)
    series = pd.Series(np.arange(7))

    err_msg = re.escape(f'`series` must be a pandas DataFrame. Got {type(series)}.')
    with pytest.raises(TypeError, match = err_msg):
        forecaster.create_train_X_y(series=series)


def test_create_train_X_y_exception_when_level_not_in_series():
    """
    Test exception is raised when level not in series column names.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1', lags=3, steps=2)
    series = pd.DataFrame({'1': pd.Series(np.arange(5)),  
                           'l2': pd.Series(np.arange(5))
                           })
    series_col_names = list(series.columns)

    err_msg = re.escape(
                (f'One of the `series` columns must be named as the `level` of the forecaster.\n'
                 f'    forecaster `level` : {forecaster.level}.\n'
                 f'    `series` columns   : {series_col_names}.')
            )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.create_train_X_y(series=series)


def test_create_train_X_y_exception_when_lags_keys_not_in_series():
    """
    Test exception is raised when lags keys are not in series column names.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags={'l1': 2, '2': 3}, steps=2)
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)),  
                           'l2': pd.Series(np.arange(10))
                           })
    series_col_names = list(series.columns)

    err_msg = re.escape(
                    (f'When `lags` parameter is a `dict`, its keys must be the '
                     f'same as `series` column names.\n'
                     f'    Lags keys        : {list(forecaster.lags_.keys())}.\n'
                     f'    `series` columns : {series_col_names}.')
                )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.create_train_X_y(series=series)


def test_create_train_X_y_exception_when_len_series_is_lower_than_maximum_lag_plus_steps():
    """
    Test exception is raised when length of y is lower than maximum lag plus
    number of steps included in the forecaster.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3)
    series = pd.DataFrame({'l1': pd.Series(np.arange(5)),  
                           'l2': pd.Series(np.arange(5))
                           })

    err_msg = re.escape(
                (f'Minimum length of `series` for training this forecaster is '
                 f'{forecaster.max_lag + forecaster.steps}. Got {len(series)}.')
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.create_train_X_y(series=series)


def test_create_train_X_y_warning_when_levels_of_transformer_series_not_equal_to_series_col_names():
    """
    Test warning is raised when `transformer_series` is a dict and its keys 
    are not the same as series column names.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)),  
                           'l2': pd.Series(np.arange(10))
                           })
    series_col_names = list(series.columns)

    dict_transformers = {'l1': StandardScaler(), 
                         'l3': StandardScaler()
                        }
    forecaster = ForecasterAutoregMultiVariate(regressor          = LinearRegression(), 
                                               level              = 'l1',
                                               lags               = 3,
                                               steps              = 3,
                                               transformer_series = dict_transformers)
    
    series_not_in_transformer_series = set(series.columns) - set(forecaster.transformer_series.keys())
    warn_msg = re.escape(
                    (f"{series_not_in_transformer_series} not present in `transformer_series`."
                     f" No transformation is applied to these series.")
                )
    with pytest.warns(UserWarning, match = warn_msg):
        forecaster.create_train_X_y(series=series)


@pytest.mark.parametrize("exog", [pd.Series(np.arange(15)), pd.DataFrame(np.arange(50).reshape(25, 2))])
def test_create_train_X_y_exception_when_series_and_exog_have_different_length(exog):
    """
    Test exception is raised when length of series is not equal to length exog.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(10))
                          })
    forecaster = forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                                            lags=3, steps=3)
    
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
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3)
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(10))
                          })

    series.index = pd.date_range(start='2022-01-01', periods=10, freq='1D')

    err_msg = re.escape(
                ('Different index for `series` and `exog`. They must be equal '
                 'to ensure the correct alignment of values.')
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.fit(
            series = series,
            exog   = pd.Series(np.arange(10), index=pd.RangeIndex(start=0, stop=10, step=1))
        )


def test_create_train_X_y_output_when_lags_3_steps_1_and_exog_is_None():
    """
    Test output of create_train_X_y when regressor is LinearRegression, 
    lags is 3 and steps is 1.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=1)
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(100, 110))
                          })
    results = forecaster.create_train_X_y(series=series)
    expected = (pd.DataFrame(
                    data = np.array([[2., 1., 0., 102., 101., 100.],
                                     [3., 2., 1., 103., 102., 101.],
                                     [4., 3., 2., 104., 103., 102.],
                                     [5., 4., 3., 105., 104., 103.],
                                     [6., 5., 4., 106., 105., 104.],
                                     [7., 6., 5., 107., 106., 105.],
                                     [8., 7., 6., 108., 107., 106.]]),
                    index   = np.array([3, 4, 5, 6, 7, 8, 9]),
                    columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3',
                               'l2_lag_1', 'l2_lag_2', 'l2_lag_3']
                ),
                pd.DataFrame(
                    data    = np.array([3., 4., 5., 6., 7., 8., 9.]),
                    index   = np.array([3, 4, 5, 6, 7, 8, 9]),
                    columns = ['l1_step_1'])
               )  
 
    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            assert (results[i] == expected[i]).all()


def test_create_train_X_y_output_when_interspersed_lags_steps_2_and_exog_is_None():
    """
    Test output of create_train_X_y when regressor is LinearRegression, 
    interspersed lags and steps is 2.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=[1, 3], steps=2)
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(100, 110))
                          })
    results = forecaster.create_train_X_y(series=series)
    expected = (pd.DataFrame(
                    data = np.array([[2., 0., 102., 100.],
                                     [3., 1., 103., 101.],
                                     [4., 2., 104., 102.],
                                     [5., 3., 105., 103.],
                                     [6., 4., 106., 104.],
                                     [7., 5., 107., 105.]]),
                    index   = np.array([4, 5, 6, 7, 8, 9]),
                    columns = ['l1_lag_1', 'l1_lag_3',
                               'l2_lag_1', 'l2_lag_3']
                ),
                pd.DataFrame(
                    np.array([[3., 4.],
                              [4., 5.],
                              [5., 6.],
                              [6., 7.],
                              [7., 8.],
                              [8., 9.]]),
                    index = np.array([4, 5, 6, 7, 8, 9]),
                    columns = ['l1_step_1', 'l1_step_2'])
               )  
 
    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            assert (results[i] == expected[i]).all()


def test_create_train_X_y_output_when_different_lags_steps_2_and_exog_is_None():
    """
    Test output of create_train_X_y when regressor is LinearRegression, 
    different lags and steps is 2.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags={'l1': 3, 'l2': [1, 5]}, 
                                               steps=2)
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(100, 110))
                          })
    results = forecaster.create_train_X_y(series=series)
    expected = (pd.DataFrame(
                    data = np.array([[4., 3., 2., 104., 100.],
                                     [5., 4., 3., 105., 101.],
                                     [6., 5., 4., 106., 102.],
                                     [7., 6., 5., 107., 103.]]),
                    index   = np.array([6, 7, 8, 9]),
                    columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3',
                               'l2_lag_1', 'l2_lag_5']
                ),
                pd.DataFrame(
                    np.array([[5., 6.],
                              [6., 7.],
                              [7., 8.],
                              [8., 9.]]),
                    index = np.array([6, 7, 8, 9]),
                    columns = ['l1_step_1', 'l1_step_2'])
               )  
 
    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            assert (results[i] == expected[i]).all()
    
    

def test_create_train_X_y_output_when_lags_5_steps_1_and_exog_is_series():
    """
    Test output of create_train_X_y when regressor is LinearRegression, lags is 5,
    steps is 1 and exog is pandas Series.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=5, steps=1)
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(50, 60))
                          })
    results = forecaster.create_train_X_y(
                  series = series,
                  exog   = pd.Series(np.arange(100, 110), name='exog')
              )
    expected = (pd.DataFrame(
                    data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50., 105.],
                                     [5., 4., 3., 2., 1., 55., 54., 53., 52., 51., 106.],
                                     [6., 5., 4., 3., 2., 56., 55., 54., 53., 52., 107.],
                                     [7., 6., 5., 4., 3., 57., 56., 55., 54., 53., 108.],
                                     [8., 7., 6., 5., 4., 58., 57., 56., 55., 54., 109.]]),
                    index = np.array([5, 6, 7, 8, 9]),
                    columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                               'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
                               'exog_step_1']
                ),
                pd.DataFrame(
                    np.array([[5.],
                              [6.],
                              [7.],
                              [8.],
                              [9.]]),
                    index = np.array([5, 6, 7, 8, 9]),
                    columns = ['l1_step_1'])
               )    

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            assert (results[i] == expected[i]).all()


def test_create_train_X_y_output_when_lags_5_steps_2_and_exog_is_series():
    """
    Test output of create_train_X_y when regressor is LinearRegression, lags is 5,
    steps is 2 and exog is pandas Series.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=5, steps=2)
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(50, 60))
                          })
    results = forecaster.create_train_X_y(
                  series = series,
                  exog   = pd.Series(np.arange(100, 110), name='exog')
              )
 
    expected = (pd.DataFrame(
                    data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50., 105., 106.],
                                     [5., 4., 3., 2., 1., 55., 54., 53., 52., 51., 106., 107.],
                                     [6., 5., 4., 3., 2., 56., 55., 54., 53., 52., 107., 108.],
                                     [7., 6., 5., 4., 3., 57., 56., 55., 54., 53., 108., 109.]]),
                    index = np.array([6, 7, 8, 9]),
                    columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                               'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
                               'exog_step_1', 'exog_step_2']
                ),
                pd.DataFrame(
                    np.array([[5., 6.],
                              [6., 7.],
                              [7., 8.],
                              [8., 9.]]),
                    index = np.array([6, 7, 8, 9]),
                    columns = ['l1_step_1', 'l1_step_2'])
               )    

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            assert (results[i] == expected[i]).all()
    

def test_create_train_X_y_output_when_lags_5_steps_1_and_exog_is_dataframe():
    """
    Test output of create_train_X_y when regressor is LinearRegression, lags is 5,
    steps is 2 and exog is pandas dataframe.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l2',
                                               lags=5, steps=1)
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(50, 60))
                          })
    results = forecaster.create_train_X_y(
                series = series,
                exog   = pd.DataFrame({
                             'exog_1' : np.arange(100, 110),
                             'exog_2' : np.arange(1000, 1010)
                         })
              )  

    expected = (pd.DataFrame(
                    data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50., 105., 1005.],
                                     [5., 4., 3., 2., 1., 55., 54., 53., 52., 51., 106., 1006.],
                                     [6., 5., 4., 3., 2., 56., 55., 54., 53., 52., 107., 1007.],
                                     [7., 6., 5., 4., 3., 57., 56., 55., 54., 53., 108., 1008.],
                                     [8., 7., 6., 5., 4., 58., 57., 56., 55., 54., 109., 1009.]], dtype=float),
                    index = np.array([5, 6, 7, 8, 9]),
                    columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                               'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
                               'exog_1_step_1', 'exog_2_step_1']
                ),
                pd.DataFrame(
                    np.array([[55.],
                              [56.],
                              [57.],
                              [58.],
                              [59.]]),
                    index = np.array([5, 6, 7, 8, 9]),
                    columns = ['l2_step_1'])
               )    

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            assert (results[i] == expected[i]).all()

    
def test_create_train_X_y_output_when_lags_5_steps_3_and_and_exog_is_dataframe():
    """
    Test output of create_train_X_y when regressor is LinearRegression, lags is 5,
    steps is 3 and exog is pandas dataframe.
    """
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l2',
                                               lags=5, steps=3)
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(50, 60))
                          })
    results = forecaster.create_train_X_y(
                series = series,
                exog   = pd.DataFrame({
                             'exog_1' : np.arange(100, 110),
                             'exog_2' : np.arange(1000, 1010)
                         })
              )

    expected = (pd.DataFrame(
                    data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50., 105., 106., 107., 1005., 1006., 1007.],
                                     [5., 4., 3., 2., 1., 55., 54., 53., 52., 51., 106., 107., 108., 1006., 1007., 1008.],
                                     [6., 5., 4., 3., 2., 56., 55., 54., 53., 52., 107., 108., 109., 1007., 1008., 1009.]],
                                    dtype=float),
                    index = np.array([7, 8, 9]),
                    columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                               'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
                               'exog_1_step_1', 'exog_1_step_2', 'exog_1_step_3',
                               'exog_2_step_1', 'exog_2_step_2', 'exog_2_step_3']
                ),
                pd.DataFrame(
                    np.array([[55., 56., 57.],
                              [56., 57., 58.],
                              [57., 58., 59.]], dtype=float),
                    index = np.array([7, 8, 9]),
                    columns = ['l2_step_1', 'l2_step_2', 'l2_step_3'])
               )
    
    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            assert (results[i] == expected[i]).all()