# Unit test create_train_X_y ForecasterAutoregMultiVariate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from skforecast.exceptions import MissingValuesExogWarning
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def test_create_train_X_y_TypeError_when_exog_is_categorical_of_no_int():
    """
    Test TypeError is raised when exog is categorical with no int values.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(6)),  
                           'l2': pd.Series(np.arange(6))})
    exog = pd.Series(['A', 'B', 'C', 'A', 'B', 'C'], name='exog', dtype='category')
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1', lags=2, steps=2)

    err_msg = re.escape(
                ("If exog is of type category, it must contain only integer values. "
                 "See skforecast docs for more info about how to include categorical "
                 "features https://skforecast.org/"
                 "latest/user_guides/categorical-features.html")
              )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.create_train_X_y(series=series, exog=exog)


def test_create_train_X_y_MissingValuesExogWarning_when_exog_has_missing_values():
    """
    Test create_train_X_y is issues a MissingValuesExogWarning when exog has missing values.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(6)),  
                           'l2': pd.Series(np.arange(6))})
    exog = pd.Series([1, 2, 3, np.nan, 5, 6], name='exog')
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l2', lags=2, steps=2)

    warn_msg = re.escape(
                ("`exog` has missing values. Most machine learning models do "
                 "not allow missing values. Fitting the forecaster may fail.")  
              )
    with pytest.warns(MissingValuesExogWarning, match = warn_msg):
        forecaster.create_train_X_y(series=series, exog=exog)


def test_create_train_X_y_TypeError_when_series_not_dataframe():
    """
    Test TypeError is raised when series is not a pandas DataFrame.
    """
    series = pd.Series(np.arange(7))
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1', lags=3, steps=2)

    err_msg = re.escape(f"`series` must be a pandas DataFrame. Got {type(series)}.")
    with pytest.raises(TypeError, match = err_msg):
        forecaster.create_train_X_y(series=series)


def test_create_train_X_y_ValueError_when_level_not_in_series():
    """
    Test ValueError is raised when level not in series column names.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)),  
                           'l2': pd.Series(np.arange(5))})
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1', lags=3, steps=2)

    series_col_names = list(series.columns)

    err_msg = re.escape(
                (f"One of the `series` columns must be named as the `level` of the forecaster.\n"
                 f"    forecaster `level` : {forecaster.level}.\n"
                 f"    `series` columns   : {series_col_names}.")
            )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.create_train_X_y(series=series)


def test_create_train_X_y_ValueError_when_lags_keys_not_in_series():
    """
    Test ValueError is raised when lags keys are not in series column names.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)),  
                           'l2': pd.Series(np.arange(10))})
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags={'l1': 2, '2': 3}, steps=2)
    
    series_col_names = list(series.columns)

    err_msg = re.escape(
                    (f"When `lags` parameter is a `dict`, its keys must be the "
                     f"same as `series` column names.\n"
                     f"    Lags keys        : {list(forecaster.lags_.keys())}.\n"
                     f"    `series` columns : {series_col_names}.")
                )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.create_train_X_y(series=series)


def test_create_train_X_y_ValueError_when_len_series_is_lower_than_maximum_lag_plus_steps():
    """
    Test ValueError is raised when length of y is lower than maximum lag plus
    number of steps included in the forecaster.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(5)),  
                           'l2': pd.Series(np.arange(5))})
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3)

    err_msg = re.escape(
                (f"Minimum length of `series` for training this forecaster is "
                 f"{forecaster.max_lag + forecaster.steps}. Got {len(series)}. Reduce the "
                 f"number of predicted steps, {forecaster.steps}, or the maximum "
                 f"lag, {forecaster.max_lag}, if no more data is available.")
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.create_train_X_y(series=series)


def test_create_train_X_y_UserWarning_when_levels_of_transformer_series_not_equal_to_series_col_names():
    """
    Test UserWarning is raised when `transformer_series` is a dict and its keys 
    are not the same as series column names.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)),  
                           'l2': pd.Series(np.arange(10))})
    dict_transformers = {'l1': StandardScaler(), 
                         'l3': StandardScaler()}
    
    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = LinearRegression(), 
                     level              = 'l1',
                     lags               = 3,
                     steps              = 3,
                     transformer_series = dict_transformers
                 )
    series_not_in_transformer_series = set(series.columns) - set(forecaster.transformer_series.keys())

    warn_msg = re.escape(
                    (f"{series_not_in_transformer_series} not present in `transformer_series`."
                     f" No transformation is applied to these series.")
                )
    with pytest.warns(UserWarning, match = warn_msg):
        forecaster.create_train_X_y(series=series)


@pytest.mark.parametrize("exog", 
                         [pd.Series(np.arange(15)), 
                          pd.DataFrame(np.arange(50).reshape(25, 2))])
def test_create_train_X_y_ValueError_when_series_and_exog_have_different_length(exog):
    """
    Test ValueError is raised when length of series is not equal to length exog.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(10))})
    forecaster = forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                                            lags=3, steps=3)
    
    err_msg = re.escape(
                (f"`exog` must have same number of samples as `series`. "
                 f"length `exog`: ({len(exog)}), length `series`: ({len(series)})")
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.create_train_X_y(series=series, exog=exog)


def test_create_train_X_y_ValueError_when_series_and_exog_have_different_index():
    """
    Test ValueError is raised when series and exog have different index.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(10))})
    series.index = pd.date_range(start='2022-01-01', periods=10, freq='1D')
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3)
    
    exog = pd.Series(np.arange(10), index=pd.RangeIndex(start=0, stop=10, step=1))

    err_msg = re.escape(
                ("Different index for `series` and `exog`. They must be equal "
                 "to ensure the correct alignment of values.")
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.fit(series=series, exog=exog)


@pytest.mark.parametrize("level, expected_y_values", 
                         [('l1', [3., 4., 5., 6., 7., 8., 9.]), 
                          ('l2', [103., 104., 105., 106., 107., 108., 109.])])
def test_create_train_X_y_output_when_lags_3_steps_1_and_exog_is_None(level, expected_y_values):
    """
    Test output of create_train_X_y when regressor is LinearRegression, 
    lags is 3 and steps is 1.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(100, 110), dtype=float)})
    exog = None

    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level=level,
                                               lags=3, steps=1)
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[2., 1., 0., 102., 101., 100.],
                             [3., 2., 1., 103., 102., 101.],
                             [4., 3., 2., 104., 103., 102.],
                             [5., 4., 3., 105., 104., 103.],
                             [6., 5., 4., 106., 105., 104.],
                             [7., 6., 5., 107., 106., 105.],
                             [8., 7., 6., 108., 107., 106.]], dtype=float),
            index   = pd.RangeIndex(start=3, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3',
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3']
        ),
        {1: pd.Series(
                data  = np.array(expected_y_values, dtype=float), 
                index = pd.RangeIndex(start=3, stop=10, step=1),
                name  = f'{level}_step_1'
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


@pytest.mark.parametrize("level, expected_y_values", 
                         [('l1', ([3., 4., 5., 6., 7., 8.], 
                                  [4., 5., 6., 7., 8., 9.])), 
                          ('l2', ([103., 104., 105., 106., 107., 108.], 
                                  [104., 105., 106., 107., 108., 109.]))])
def test_create_train_X_y_output_when_interspersed_lags_steps_2_and_exog_is_None(level, expected_y_values):
    """
    Test output of create_train_X_y when regressor is LinearRegression, 
    interspersed lags and steps is 2.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(100, 110), dtype=float)})
    exog = None

    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level=level,
                                               lags=[1, 3], steps=2)
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[2., 0., 102., 100.],
                             [3., 1., 103., 101.],
                             [4., 2., 104., 102.],
                             [5., 3., 105., 103.],
                             [6., 4., 106., 104.],
                             [7., 5., 107., 105.]], dtype=float),
            index   = pd.RangeIndex(start=4, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_3',
                       'l2_lag_1', 'l2_lag_3']
        ),
        {1: pd.Series(
                data  = np.array(expected_y_values[0], dtype=float), 
                index = pd.RangeIndex(start=3, stop=9, step=1),
                name  = f'{level}_step_1'
            ),
         2: pd.Series(
                data  = np.array(expected_y_values[1], dtype=float), 
                index = pd.RangeIndex(start=4, stop=10, step=1),
                name  = f'{level}_step_2'
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


@pytest.mark.parametrize("level, expected_y_values", 
                         [('l1', ([5., 6., 7., 8.], 
                                  [6., 7., 8., 9.])), 
                          ('l2', ([105., 106., 107., 108.], 
                                  [106., 107., 108., 109.]))])
def test_create_train_X_y_output_when_different_lags_steps_2_and_exog_is_None(level, expected_y_values):
    """
    Test output of create_train_X_y when regressor is LinearRegression, 
    different lags and steps is 2.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(100, 110))})
    exog = None

    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level=level,
                                               lags={'l1': 3, 'l2': [1, 5]}, 
                                               steps=2)
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 104., 100.],
                             [5., 4., 3., 105., 101.],
                             [6., 5., 4., 106., 102.],
                             [7., 6., 5., 107., 103.]], dtype=float),
            index   = pd.RangeIndex(start=6, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3',
                       'l2_lag_1', 'l2_lag_5']
        ),
        {1: pd.Series(
                data  = np.array(expected_y_values[0], dtype=float), 
                index = pd.RangeIndex(start=5, stop=9, step=1),
                name  = f'{level}_step_1'
            ),
         2: pd.Series(
                data  = np.array(expected_y_values[1], dtype=float), 
                index = pd.RangeIndex(start=6, stop=10, step=1),
                name  = f'{level}_step_2'
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


def test_create_train_X_y_output_when_y_and_exog_no_pandas_index():
    """
    Test the output of create_train_X_y when y and exog have no pandas index 
    that doesn't start at 0.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    series.index = np.arange(6, 16)
    exog = pd.Series(np.arange(100, 110), name='exog', 
                     index=np.arange(6, 16), dtype=float)

    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=5, steps=1)
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50., 105.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51., 106.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52., 107.],
                             [7., 6., 5., 4., 3., 57., 56., 55., 54., 53., 108.],
                             [8., 7., 6., 5., 4., 58., 57., 56., 55., 54., 109.]], 
                             dtype=float),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
                       'exog_step_1']
        ),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "l1_step_1"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt : f'dtype: {dt}')
def test_create_train_X_y_output_when_lags_5_steps_1_and_exog_is_series_of_float_int(dtype):
    """
    Test output of create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 1 and exog is pandas Series of floats or ints.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=dtype)

    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=5, steps=1)
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50., 105.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51., 106.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52., 107.],
                             [7., 6., 5., 4., 3., 57., 56., 55., 54., 53., 108.],
                             [8., 7., 6., 5., 4., 58., 57., 56., 55., 54., 109.]], 
                             dtype=float),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
                       'exog_step_1']
        ).astype({'exog_step_1': dtype}),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "l1_step_1"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt : f'dtype: {dt}')
def test_create_train_X_y_output_when_lags_5_steps_2_and_exog_is_series_of_float_int(dtype):
    """
    Test output of create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 2 and exog is pandas Series of floats or ints.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=dtype)

    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=5, steps=2)
    results = forecaster.create_train_X_y(series=series, exog=exog)
 
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50., 105., 106.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51., 106., 107.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52., 107., 108.],
                             [7., 6., 5., 4., 3., 57., 56., 55., 54., 53., 108., 109.]], 
                             dtype=float),
            index = pd.RangeIndex(start=6, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
                       'exog_step_1', 'exog_step_2']
        ).astype({'exog_step_1': dtype, 'exog_step_2': dtype}),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=9, step=1),
                name  = "l1_step_1"
            ),
         2: pd.Series(
                data  = np.array([6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=6, stop=10, step=1),
                name  = "l1_step_2"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt : f'dtype: {dt}')
def test_create_train_X_y_output_when_lags_5_steps_1_and_exog_is_dataframe_of_float_int(dtype):
    """
    Test output of create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 1 and exog is pandas DataFrame of floats or ints.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog   = pd.DataFrame({'exog_1' : np.arange(100, 110, dtype=dtype),
                           'exog_2' : np.arange(1000, 1010, dtype=dtype)})
    
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l2',
                                               lags=5, steps=1)
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50., 105., 1005.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51., 106., 1006.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52., 107., 1007.],
                             [7., 6., 5., 4., 3., 57., 56., 55., 54., 53., 108., 1008.],
                             [8., 7., 6., 5., 4., 58., 57., 56., 55., 54., 109., 1009.]], 
                             dtype=float),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
                       'exog_1_step_1', 'exog_2_step_1']
        ).astype({'exog_1_step_1': dtype, 'exog_2_step_1': dtype}),
        {1: pd.Series(
                data  = np.array([55., 56., 57., 58., 59.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "l2_step_1"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt : f'dtype: {dt}')
def test_create_train_X_y_output_when_lags_5_steps_3_and_exog_is_dataframe_of_float_int(dtype):
    """
    Test output of create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 3 and exog is pandas DataFrame of floats or ints.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.DataFrame({'exog_1': np.arange(100, 110, dtype=dtype),
                         'exog_2': np.arange(1000, 1010, dtype=dtype)})
    
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l2',
                                               lags=5, steps=3)
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50., 105., 1005., 106., 1006., 107., 1007.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51., 106., 1006., 107., 1007., 108., 1008.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52., 107., 1007., 108., 1008., 109., 1009.]],
                            dtype=float),
            index = pd.RangeIndex(start=7, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
                       'exog_1_step_1', 'exog_2_step_1', 
                       'exog_1_step_2', 'exog_2_step_2', 
                       'exog_1_step_3', 'exog_2_step_3']
        ).astype({'exog_1_step_1': dtype, 'exog_2_step_1': dtype, 
                  'exog_1_step_2': dtype, 'exog_2_step_2': dtype, 
                  'exog_1_step_3': dtype, 'exog_2_step_3': dtype}),
        {1: pd.Series(
                data  = np.array([55., 56., 57.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=8, step=1),
                name  = "l2_step_1"
            ),
         2: pd.Series(
                data  = np.array([56., 57., 58.], dtype=float), 
                index = pd.RangeIndex(start=6, stop=9, step=1),
                name  = "l2_step_2"
            ),
         3: pd.Series(
                data  = np.array([57., 58., 59.], dtype=float), 
                index = pd.RangeIndex(start=7, stop=10, step=1),
                name  = "l2_step_3"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


@pytest.mark.parametrize("exog_values, dtype", 
                         [([True]    , bool), 
                          (['string'], str)], 
                         ids = lambda dt : f'values, dtype: {dt}')
def test_create_train_X_y_output_when_lags_5_steps_1_and_exog_is_series_of_bool_str(exog_values, dtype):
    """
    Test output of create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 1 and exog is pandas Series of bool or str.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.Series(exog_values*10, name='exog', dtype=dtype)

    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=5, steps=1)
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52.],
                             [7., 6., 5., 4., 3., 57., 56., 55., 54., 53.],
                             [8., 7., 6., 5., 4., 58., 57., 56., 55., 54.]], 
                             dtype=float),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5']
        ).assign(exog_step_1=exog_values*5).astype({'exog_step_1': dtype}),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "l1_step_1"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


@pytest.mark.parametrize("exog_values, dtype", 
                         [([True]    , bool), 
                          (['string'], str)], 
                         ids = lambda dt : f'values, dtype: {dt}')
def test_create_train_X_y_output_when_lags_5_steps_2_and_exog_is_series_of_bool_str(exog_values, dtype):
    """
    Test output of create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 2 and exog is pandas Series of bool or str.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.Series(exog_values*10, name='exog', dtype=dtype)

    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=5, steps=2)
    results = forecaster.create_train_X_y(series=series, exog=exog)
 
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52.],
                             [7., 6., 5., 4., 3., 57., 56., 55., 54., 53.]], 
                             dtype=float),
            index = pd.RangeIndex(start=6, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5']
        ).assign(exog_step_1=exog_values*4,
                 exog_step_2=exog_values*4).astype({'exog_step_1': dtype, 
                                                    'exog_step_2': dtype}),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=9, step=1),
                name  = "l1_step_1"
            ),
         2: pd.Series(
                data  = np.array([6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=6, stop=10, step=1),
                name  = "l1_step_2"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


@pytest.mark.parametrize("v_exog_1   , v_exog_2  , dtype", 
                         [([True]    , [False]   , bool), 
                          (['string'], ['string'], str)], 
                         ids = lambda dt : f'values, dtype: {dt}')
def test_create_train_X_y_output_when_lags_5_steps_1_and_exog_is_dataframe_of_bool_str(v_exog_1, v_exog_2, dtype):
    """
    Test output of create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 1 and exog is pandas DataFrame of bool or str.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.DataFrame({'exog_1': v_exog_1*10,
                         'exog_2': v_exog_2*10})
    
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l2',
                                               lags=5, steps=1)
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52.],
                             [7., 6., 5., 4., 3., 57., 56., 55., 54., 53.],
                             [8., 7., 6., 5., 4., 58., 57., 56., 55., 54.]], 
                             dtype=float),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5']
        ).assign(exog_1_step_1=v_exog_1*5,
                 exog_2_step_1=v_exog_2*5).astype({'exog_1_step_1': dtype, 
                                                   'exog_2_step_1': dtype}),
        {1: pd.Series(
                data  = np.array([55., 56., 57., 58., 59.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "l2_step_1"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


@pytest.mark.parametrize("v_exog_1   , v_exog_2  , dtype", 
                         [([True]    , [False]   , bool), 
                          (['string'], ['string'], str)], 
                         ids = lambda dt : f'values, dtype: {dt}')
def test_create_train_X_y_output_when_lags_5_steps_3_and_exog_is_dataframe_of_bool_str(v_exog_1, v_exog_2, dtype):
    """
    Test output of create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 3 and exog is pandas DataFrame of bool or str.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.DataFrame({'exog_1': v_exog_1*10,
                         'exog_2': v_exog_2*10})
    
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l2',
                                               lags=5, steps=3)
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52.]],
                            dtype=float),
            index = pd.RangeIndex(start=7, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5']
        ).assign(exog_1_step_1=v_exog_1*3,
                 exog_2_step_1=v_exog_2*3,
                 exog_1_step_2=v_exog_1*3,
                 exog_2_step_2=v_exog_2*3,
                 exog_1_step_3=v_exog_1*3,
                 exog_2_step_3=v_exog_2*3
        ).astype({'exog_1_step_1': dtype, 'exog_2_step_1': dtype, 
                  'exog_1_step_2': dtype, 'exog_2_step_2': dtype, 
                  'exog_1_step_3': dtype, 'exog_2_step_3': dtype}
        ),
        {1: pd.Series(
                data  = np.array([55., 56., 57.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=8, step=1),
                name  = "l2_step_1"
            ),
         2: pd.Series(
                data  = np.array([56., 57., 58.], dtype=float), 
                index = pd.RangeIndex(start=6, stop=9, step=1),
                name  = "l2_step_2"
            ),
         3: pd.Series(
                data  = np.array([57., 58., 59.], dtype=float), 
                index = pd.RangeIndex(start=7, stop=10, step=1),
                name  = "l2_step_3"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


def test_create_train_X_y_output_when_lags_5_steps_1_and_exog_is_series_of_category():
    """
    Test output of create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 1 and exog is pandas Series of category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.Series(range(10), name='exog', dtype='category')

    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=5, steps=1)
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52.],
                             [7., 6., 5., 4., 3., 57., 56., 55., 54., 53.],
                             [8., 7., 6., 5., 4., 58., 57., 56., 55., 54.]], 
                             dtype=float),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5']
        ).assign(exog_step_1=pd.Categorical(range(5, 10), categories=range(10))),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "l1_step_1"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


def test_create_train_X_y_output_when_lags_5_steps_2_and_exog_is_series_of_category():
    """
    Test output of create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 2 and exog is pandas Series of category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.Series(range(10), name='exog', dtype='category')

    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=5, steps=2)
    results = forecaster.create_train_X_y(series=series, exog=exog)
 
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52.],
                             [7., 6., 5., 4., 3., 57., 56., 55., 54., 53.]], 
                             dtype=float),
            index = pd.RangeIndex(start=6, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5']
        ).assign(exog_step_1=pd.Categorical(range(5, 9), categories=range(10)),
                 exog_step_2=pd.Categorical(range(6, 10), categories=range(10))),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=9, step=1),
                name  = "l1_step_1"
            ),
         2: pd.Series(
                data  = np.array([6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=6, stop=10, step=1),
                name  = "l1_step_2"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


def test_create_train_X_y_output_when_lags_5_steps_1_and_exog_is_dataframe_of_category():
    """
    Test output of create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 1 and exog is pandas DataFrame of category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.DataFrame({'exog_1': pd.Categorical(range(10)),
                         'exog_2': pd.Categorical(range(100, 110))})
    
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l2',
                                               lags=5, steps=1)
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52.],
                             [7., 6., 5., 4., 3., 57., 56., 55., 54., 53.],
                             [8., 7., 6., 5., 4., 58., 57., 56., 55., 54.]], 
                             dtype=float),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5']
        ).assign(
            exog_1_step_1=pd.Categorical(range(5, 10), categories=range(10)),
            exog_2_step_1=pd.Categorical(range(105, 110), categories=range(100, 110))
        ),
        {1: pd.Series(
                data  = np.array([55., 56., 57., 58., 59.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "l2_step_1"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


def test_create_train_X_y_output_when_lags_5_steps_3_and_exog_is_dataframe_of_category():
    """
    Test output of create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 3 and exog is pandas DataFrame of category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.DataFrame({'exog_1': pd.Categorical(range(10)),
                         'exog_2': pd.Categorical(range(100, 110))})
    
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l2',
                                               lags=5, steps=3)
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52.]],
                            dtype=float),
            index = pd.RangeIndex(start=7, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5']
        ).assign(
            exog_1_step_1=pd.Categorical(range(5, 8), categories=range(10)),
            exog_2_step_1=pd.Categorical(range(105, 108), categories=range(100, 110)),
            exog_1_step_2=pd.Categorical(range(6, 9), categories=range(10)),
            exog_2_step_2=pd.Categorical(range(106, 109), categories=range(100, 110)),
            exog_1_step_3=pd.Categorical(range(7, 10), categories=range(10)),
            exog_2_step_3=pd.Categorical(range(107, 110), categories=range(100, 110))
        ),
        {1: pd.Series(
                data  = np.array([55., 56., 57.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=8, step=1),
                name  = "l2_step_1"
            ),
         2: pd.Series(
                data  = np.array([56., 57., 58.], dtype=float), 
                index = pd.RangeIndex(start=6, stop=9, step=1),
                name  = "l2_step_2"
            ),
         3: pd.Series(
                data  = np.array([57., 58., 59.], dtype=float), 
                index = pd.RangeIndex(start=7, stop=10, step=1),
                name  = "l2_step_3"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_dataframe_of_float_int_category_steps_1():
    """
    Test output of create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 1 and exog is pandas DataFrame of float, int and category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.DataFrame({'exog_1': pd.Series(np.arange(100, 110), dtype=float),
                         'exog_2': pd.Series(np.arange(1000, 1010), dtype=int),
                         'exog_3': pd.Categorical(range(100, 110))})
    
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=5, steps=1)
    results = forecaster.create_train_X_y(series=series, exog=exog)        
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50., 105., 1005.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51., 106., 1006.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52., 107., 1007.],
                             [7., 6., 5., 4., 3., 57., 56., 55., 54., 53., 108., 1008.],
                             [8., 7., 6., 5., 4., 58., 57., 56., 55., 54., 109., 1009.]], 
                             dtype=float),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
                       'exog_1_step_1', 'exog_2_step_1']
        ).astype({'exog_1_step_1': float, 
                  'exog_2_step_1': int}).assign(exog_3_step_1=pd.Categorical(range(105, 110), categories=range(100, 110))
        ),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "l1_step_1"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_dataframe_of_float_int_category_steps_3():
    """
    Test output of create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 3 and exog is pandas DataFrame of float, int and category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.DataFrame({'exog_1': pd.Series(np.arange(100, 110), dtype=float),
                         'exog_2': pd.Series(np.arange(1000, 1010), dtype=int),
                         'exog_3': pd.Categorical(range(100, 110))})
    
    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level='l1',
                                               lags=5, steps=3)
    results = forecaster.create_train_X_y(series=series, exog=exog)        
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50., 105, 1005, 105, 106, 1006, 106, 107, 1007, 107],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51., 106, 1006, 106, 107, 1007, 107, 108, 1008, 108],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52., 107, 1007, 107, 108, 1008, 108, 109, 1009, 109]],
                            dtype=float),
            index = pd.RangeIndex(start=7, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
                       'exog_1_step_1', 'exog_2_step_1', 'exog_3_step_1', 
                       'exog_1_step_2', 'exog_2_step_2', 'exog_3_step_2', 
                       'exog_1_step_3', 'exog_2_step_3', 'exog_3_step_3']
        ).astype({'exog_1_step_1': float, 'exog_2_step_1': int,
                  'exog_1_step_2': float, 'exog_2_step_2': int,
                  'exog_1_step_3': float, 'exog_2_step_3': int}
        ).assign(exog_3_step_1=pd.Categorical(range(105, 108), categories=range(100, 110)),
                 exog_3_step_2=pd.Categorical(range(106, 109), categories=range(100, 110)),
                 exog_3_step_3=pd.Categorical(range(107, 110), categories=range(100, 110))
        ),
        {1: pd.Series(
                data  = np.array([5., 6., 7.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=8, step=1),
                name  = "l1_step_1"
            ),
         2: pd.Series(
                data  = np.array([6., 7., 8.], dtype=float), 
                index = pd.RangeIndex(start=6, stop=9, step=1),
                name  = "l1_step_2"
            ),
         3: pd.Series(
                data  = np.array([7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=7, stop=10, step=1),
                name  = "l1_step_3"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(),
                          {'l1': StandardScaler(), 'l2': StandardScaler()}], 
                         ids = lambda tr : f'transformer_series: {tr}')
def test_create_train_X_y_output_when_lags_5_steps_1_and_transformer_series_StandardScaler(transformer_series):
    """
    Test the output of create_train_X_y when exog is None and transformer_series
    is StandardScaler with steps=1.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    
    forecaster = ForecasterAutoregMultiVariate(
                    regressor          = LinearRegression(),
                    lags               = 5,
                    level              = 'l1',
                    steps              = 1,
                    transformer_series = transformer_series
                )
    results = forecaster.create_train_X_y(series=series)

    expected = (
        pd.DataFrame(
            data = np.array([[-0.17407766, -0.52223297, -0.87038828, -1.21854359, -1.5666989 ,
                              -0.17407766, -0.52223297, -0.87038828, -1.21854359, -1.5666989 ],
                             [ 0.17407766, -0.17407766, -0.52223297, -0.87038828, -1.21854359,
                               0.17407766, -0.17407766, -0.52223297, -0.87038828, -1.21854359],
                             [ 0.52223297,  0.17407766, -0.17407766, -0.52223297, -0.87038828,
                               0.52223297,  0.17407766, -0.17407766, -0.52223297, -0.87038828],
                             [ 0.87038828,  0.52223297,  0.17407766, -0.17407766, -0.52223297,
                               0.87038828,  0.52223297,  0.17407766, -0.17407766, -0.52223297],
                             [ 1.21854359,  0.87038828,  0.52223297,  0.17407766, -0.17407766,
                               1.21854359,  0.87038828,  0.52223297,  0.17407766, -0.17407766]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5']
        ),
        {1: pd.Series(
                data  = np.array([0.17407766, 0.52223297, 0.87038828, 
                                  1.21854359, 1.5666989]), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "l1_step_1"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


@pytest.mark.parametrize("level, expected_y_values", 
                         [('l1', [3., 4., 5., 6., 7., 8., 9.]), 
                          ('l2', [103., 104., 105., 106., 107., 108., 109.])])
def test_create_train_X_y_output_when_lags_3_steps_1_and_exog_is_None_and_transformer_exog_is_not_None(level, expected_y_values):
    """
    Test output of create_train_X_y when regressor is LinearRegression, 
    lags is 3 and steps is 1 and transformer_exog is not None.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(100, 110), dtype=float)})
    exog = None

    forecaster = ForecasterAutoregMultiVariate(LinearRegression(), level=level,
                                               lags=3, steps=1, transformer_exog = StandardScaler())
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[2., 1., 0., 102., 101., 100.],
                             [3., 2., 1., 103., 102., 101.],
                             [4., 3., 2., 104., 103., 102.],
                             [5., 4., 3., 105., 104., 103.],
                             [6., 5., 4., 106., 105., 104.],
                             [7., 6., 5., 107., 106., 105.],
                             [8., 7., 6., 108., 107., 106.]], dtype=float),
            index   = pd.RangeIndex(start=3, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3',
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3']
        ),
        {1: pd.Series(
                data  = np.array(expected_y_values, dtype=float), 
                index = pd.RangeIndex(start=3, stop=10, step=1),
                name  = f'{level}_step_1'
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(),
                          {'l1': StandardScaler(), 'l2': StandardScaler()}], 
                         ids = lambda tr : f'transformer_series: {tr}')
def test_create_train_X_y_output_when_transformer_series_and_transformer_exog_steps_2(transformer_series):
    """
    Test the output of create_train_X_y when using transformer_series and 
    transformer_exog with steps=2.
    """
    series = pd.DataFrame({'l1': np.arange(10, dtype=float), 
                           'l2': np.arange(10, dtype=float)},
                           index = pd.date_range("1990-01-01", periods=10, freq='D'))
    exog = pd.DataFrame({
               'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 24.4, 87.2, 47.4, 23.8],
               'col_2': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b']},
                index = pd.date_range("1990-01-01", periods=10, freq='D'))

    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['col_1']),
                            ('onehot', OneHotEncoder(), ['col_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )

    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = LinearRegression(),
                     lags               = 5,
                     level              = 'l1',
                     steps              = 2,
                     transformer_series = transformer_series,
                     transformer_exog   = transformer_exog
                 )
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[-0.17407766, -0.52223297, -0.87038828, -1.21854359, -1.5666989 ,
                              -0.17407766, -0.52223297, -0.87038828, -1.21854359, -1.5666989 ,
                              -0.04719331,  0.        ,   1.        ,-0.81862236,  1.        ,
                               0.        ],
                             [ 0.17407766, -0.17407766, -0.52223297, -0.87038828, -1.21854359,
                               0.17407766, -0.17407766, -0.52223297, -0.87038828, -1.21854359,
                              -0.81862236,  1.        ,  0.        ,  2.03112731,  0.        ,
                               1.        ],
                             [ 0.52223297,  0.17407766, -0.17407766, -0.52223297, -0.87038828,
                               0.52223297,  0.17407766, -0.17407766, -0.52223297, -0.87038828,
                               2.03112731,  0.        ,  1.        ,  0.22507577,  1.        ,
                               0.        ],
                             [ 0.87038828,  0.52223297,  0.17407766, -0.17407766, -0.52223297,
                               0.87038828,  0.52223297,  0.17407766, -0.17407766, -0.52223297,
                               0.22507577,  1.        ,  0.        , -0.84584926,  0.        ,
                               1.        ]]),
            index   = pd.date_range("1990-01-07", periods=4, freq='D'),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5', 
                       'col_1_step_1', 'col_2_a_step_1', 'col_2_b_step_1',
                       'col_1_step_2', 'col_2_a_step_2', 'col_2_b_step_2']
        ),
        {1: pd.Series(
                data  = np.array([0.17407766, 0.52223297, 0.87038828, 1.21854359]), 
                index = pd.date_range("1990-01-06", periods=4, freq='D'),
                name  = "l1_step_1"
            ),
         2: pd.Series(
                data  = np.array([0.52223297, 0.87038828, 1.21854359, 1.5666989]), 
                index = pd.date_range("1990-01-07", periods=4, freq='D'),
                name  = "l1_step_2"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 