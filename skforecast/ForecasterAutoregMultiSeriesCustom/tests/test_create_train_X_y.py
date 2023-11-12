# Unit test create_train_X_y ForecasterAutoregMultiSeriesCustom
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeriesCustom import ForecasterAutoregMultiSeriesCustom
from skforecast.exceptions import MissingValuesExogWarning
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

def create_predictors_3(y): # pragma: no cover
    """
    Create first 3 lags of a time series.
    """
    lags = y[-1:-4:-1]

    return lags

def create_predictors_5(y): # pragma: no cover
    """
    Create first 5 lags of a time series.
    """
    lags = y[-1:-6:-1]

    return lags

def create_predictors_nan(y): # pragma: no cover
    """
    Create first 3 lags of a time series.
    """
    lags = np.array([1, 2, np.nan], dtype=float)

    return lags


def test_create_train_X_y_TypeError_when_exog_is_categorical_of_no_int():
    """
    Test TypeError is raised when exog is categorical with no int values.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)),  
                           '2': pd.Series(np.arange(5))})
    exog = pd.Series(['A', 'B', 'C', 'D', 'F'], name='exog', dtype='category')
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors_3,
                     window_size    = 3
                 )

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
    series = pd.DataFrame({'1': pd.Series(np.arange(5)),  
                           '2': pd.Series(np.arange(5))})
    exog = pd.Series([1, 2, 3, np.nan, 5], name='exog')
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors_3,
                     window_size    = 3
                 )

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
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors_3,
                     window_size    = 3
                 )
    series = pd.Series(np.arange(7))

    err_msg = re.escape(f'`series` must be a pandas DataFrame. Got {type(series)}.')
    with pytest.raises(TypeError, match = err_msg):
        forecaster.create_train_X_y(series=series)


def test_create_train_X_y_ValueError_when_series_length_is_lower_than_windows_size():
    """
    Test ValueError is raised when length of series is less than self.window_size + 1.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(2)),  
                           '2': pd.Series(np.arange(2))})

    forecaster = ForecasterAutoregMultiSeriesCustom(
                    regressor       = LinearRegression(),
                    fun_predictors  = create_predictors_nan,
                    name_predictors = ['lag_1', 'lag_2', 'lag_3'],
                    window_size     = 3
                 )

    err_msg = re.escape(
                (f'`series` must have as many values as the windows_size needed by '
                 f'{forecaster.fun_predictors.__name__}. For this Forecaster the '
                 f'minimum length is {forecaster.window_size + 1}')      
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.create_train_X_y(series = series)


def test_create_train_X_y_UserWarning_when_levels_of_transformer_series_not_equal_to_series_col_names():
    """
    Test UserWarning is raised when `transformer_series` is a dict and its keys are
    not the same as forecaster.series_col_names.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)),  
                           '2': pd.Series(np.arange(5))})
    dict_transformers = {'1': StandardScaler(), 
                         '3': StandardScaler()}
    
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = LinearRegression(),
                     fun_predictors     = create_predictors_3,
                     window_size        = 3,
                     transformer_series = dict_transformers
                 )
    series_not_in_transformer_series = set(series.columns) - set(forecaster.transformer_series.keys())
    
    warn_msg = re.escape(
                    (f"{series_not_in_transformer_series} not present in `transformer_series`."
                     f" No transformation is applied to these series.")
                )
    with pytest.warns(UserWarning, match = warn_msg):
        forecaster.create_train_X_y(series=series)


def test_create_train_X_y_ValueError_when_all_series_values_are_missing():
    """
    Test ValueError is raised when all series values are missing.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(7)), 
                           '2': pd.Series([np.nan]*7)})
    series.index = pd.date_range(start='2022-01-01', periods=7, freq='1D')

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors_5,
                     window_size     = 5
                 )

    err_msg = re.escape("All values of series '2' are NaN.")
    with pytest.raises(ValueError, match = err_msg):
        forecaster.create_train_X_y(series=series)


@pytest.mark.parametrize("values", 
                         [[0, 1, 2, 3, 4, 5, np.nan], 
                          [0, 1]+[np.nan]*5, 
                          [np.nan, 1, 2, 3, 4, 5, np.nan],
                          [0, 1, np.nan, 3, np.nan, 5, 6], 
                          [np.nan, np.nan, np.nan, 3, np.nan, 5, 6]])
def test_create_train_X_y_ValueError_when_series_values_are_missing(values):
    """
    Test ValueError is raised when series values are missing in different
    locations.
    """
    series = pd.DataFrame({'1': pd.Series(values), 
                           '2': pd.Series(np.arange(7))})
    series.index = pd.date_range(start='2022-01-01', periods=7, freq='1D')

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors_5,
                     window_size     = 5
                 )

    err_msg = re.escape(
                ("'1' Time series has missing values in between or "
                 "at the end of the time series. When working with series "
                 "of different lengths, all series must be complete after "
                 "the first non-null value.")
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.create_train_X_y(series=series)


def test_create_train_X_y_ValueError_when_fun_predictors_return_NaN():
    """
    Test ValueError is raised when `fun_predictors()` return NaN values.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)),  
                           '2': pd.Series(np.arange(5))
                           })

    forecaster = ForecasterAutoregMultiSeriesCustom(
                    regressor       = LinearRegression(),
                    fun_predictors  = create_predictors_nan,
                    name_predictors = ['lag_1', 'lag_2', 'lag_3'],
                    window_size     = 3
                 )

    err_msg = re.escape("`fun_predictors()` is returning `NaN` values for series '1'.")
    with pytest.raises(ValueError, match = err_msg):
        forecaster.create_train_X_y(series = series)


def test_create_train_X_y_ValueError_when_len_name_predictors_not_match_X_train_columns():
    """
    Test ValueError is raised when argument `name_predictors` has less values than the number of
    columns of X_train.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)),  
                           '2': pd.Series(np.arange(5))})

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors_3,
                     name_predictors = ['lag_1', 'lag_2'],
                     window_size     = 3
                 )

    err_msg = re.escape(
                (f"The length of provided predictors names (`name_predictors`) do not "
                 f"match the number of columns created by `fun_predictors()`.")  
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.fit(series=series)


def test_create_train_X_y_ValueError_when_forecaster_window_size_does_not_match_with_fun_predictors():
    """
    Test ValueError is raised when the window needed by `fun_predictors()` does 
    not correspond with the forecaster.window_size.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)),  
                           '2': pd.Series(np.arange(5))})

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors_3,
                     window_size    = 2
                 )
    
    err_msg = re.escape(
                (f"The `window_size` argument ({forecaster.window_size}), declared when "
                 f"initializing the forecaster, does not correspond to the window "
                 f"used by `fun_predictors()`.")
            )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.fit(series=series)


@pytest.mark.parametrize("exog", 
                         [pd.Series(np.arange(10)), 
                          pd.DataFrame(np.arange(50).reshape(25, 2))])
def test_create_train_X_y_ValueError_when_series_and_exog_have_different_length(exog):
    """
    Test ValueError is raised when length of series is not equal to length exog.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(7)), 
                           '2': pd.Series(np.arange(7))})
    
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors_3,
                     window_size    = 3
                 )
    
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
    series = pd.DataFrame({'1': pd.Series(np.arange(7)), 
                           '2': pd.Series(np.arange(7))})
    series.index = pd.date_range(start='2022-01-01', periods=7, freq='1D')
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors_3,
                     window_size    = 3
                 )
    
    exog = pd.Series(np.arange(7), index=pd.RangeIndex(start=0, stop=7, step=1))

    err_msg = re.escape(
                ("Different index for `series` and `exog`. They must be equal "
                 "to ensure the correct alignment of values.")
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.fit(series=series, exog=exog)


def test_create_train_X_y_column_names_match_name_predictors():
    """
    Check column names in X_train match the ones in argument `name_predictors`.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)),  
                           '2': pd.Series(np.arange(5))})

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors_3,
                     name_predictors = ['lag_1', 'lag_2', 'lag_3'],
                     window_size     = 3
                 )
    train_X = forecaster.create_train_X_y(series = series)[0]

    assert train_X.columns.to_list() == ['lag_1', 'lag_2', 'lag_3', '1', '2']


def test_create_train_X_y_output_when_series_and_exog_is_None():
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is None.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(7, dtype=float)), 
                           '2': pd.Series(np.arange(7, dtype=float))})
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors_3,
                     window_size    = 3
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
            columns = ['custom_predictor_0', 'custom_predictor_1', 'custom_predictor_2', 
                       '1', '2']
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


def test_create_train_X_y_output_when_series_and_exog_no_pandas_index():
    """
    Test the output of create_train_X_y when series and exog have no pandas index 
    that doesn't start at 0.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    series.index = np.arange(6, 16)
    exog = pd.Series(np.arange(100, 110), index=np.arange(6, 16), 
                     name='exog', dtype=float)

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors_5,
                     window_size    = 5
                 )
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105., 1., 0.],
                             [5., 4., 3., 2., 1., 106., 1., 0.],
                             [6., 5., 4., 3., 2., 107., 1., 0.],
                             [7., 6., 5., 4., 3., 108., 1., 0.],
                             [8., 7., 6., 5., 4., 109., 1., 0.],
                             [4., 3., 2., 1., 0., 105., 0., 1.],
                             [5., 4., 3., 2., 1., 106., 0., 1.],
                             [6., 5., 4., 3., 2., 107., 0., 1.],
                             [7., 6., 5., 4., 3., 108., 0., 1.],
                             [8., 7., 6., 5., 4., 109., 0., 1.]]),
            index   = pd.RangeIndex(start=0, stop=10, step=1),
            columns = ['custom_predictor_0', 'custom_predictor_1', 'custom_predictor_2', 
                       'custom_predictor_3', 'custom_predictor_4', 'exog', 'l1', 'l2']
        ),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
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


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt : f'dtype: {dt}')
def test_create_train_X_y_output_when_series_10_and_exog_is_series_of_float_int(dtype):
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is a pandas series of floats or ints.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(10, dtype=float)), 
                           '2': pd.Series(np.arange(10, dtype=float))})
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=dtype)

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors_5,
                     window_size    = 5
                 )

    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105., 1., 0.],
                             [5., 4., 3., 2., 1., 106., 1., 0.],
                             [6., 5., 4., 3., 2., 107., 1., 0.],
                             [7., 6., 5., 4., 3., 108., 1., 0.],
                             [8., 7., 6., 5., 4., 109., 1., 0.],
                             [4., 3., 2., 1., 0., 105., 0., 1.],
                             [5., 4., 3., 2., 1., 106., 0., 1.],
                             [6., 5., 4., 3., 2., 107., 0., 1.],
                             [7., 6., 5., 4., 3., 108., 0., 1.],
                             [8., 7., 6., 5., 4., 109., 0., 1.]]),
            index   = pd.RangeIndex(start=0, stop=10, step=1),
            columns = ['custom_predictor_0', 'custom_predictor_1', 'custom_predictor_2', 
                       'custom_predictor_3', 'custom_predictor_4', 'exog', '1', '2']
        ).astype({'exog': dtype}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
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


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt : f'dtype: {dt}')
def test_create_train_X_y_output_when_series_10_and_exog_is_dataframe_of_float_int(dtype):
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is a pandas dataframe with two columns of floats or ints.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(10, dtype=float)), 
                           '2': pd.Series(np.arange(10, dtype=float))})
    exog = pd.DataFrame({'exog_1': np.arange(100, 110, dtype=dtype),
                         'exog_2': np.arange(1000, 1010, dtype=dtype)})

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors_5,
                     name_predictors = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5'],
                     window_size     = 5
                 )
    results = forecaster.create_train_X_y(series=series, exog=exog)    

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105., 1005., 1., 0.],
                             [5., 4., 3., 2., 1., 106., 1006., 1., 0.],
                             [6., 5., 4., 3., 2., 107., 1007., 1., 0.],
                             [7., 6., 5., 4., 3., 108., 1008., 1., 0.],
                             [8., 7., 6., 5., 4., 109., 1009., 1., 0.],
                             [4., 3., 2., 1., 0., 105., 1005., 0., 1.],
                             [5., 4., 3., 2., 1., 106., 1006., 0., 1.],
                             [6., 5., 4., 3., 2., 107., 1007., 0., 1.],
                             [7., 6., 5., 4., 3., 108., 1008., 0., 1.],
                             [8., 7., 6., 5., 4., 109., 1009., 0., 1.]]),
            index   = pd.RangeIndex(start=0, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'exog_1', 'exog_2', '1', '2']
        ).astype({'exog_1': dtype, 'exog_2': dtype}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
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


@pytest.mark.parametrize("exog_values, dtype", 
                         [([True]    , bool), 
                          (['string'], str)], 
                         ids = lambda dt : f'values, dtype: {dt}')
def test_create_train_X_y_output_when_series_10_and_exog_is_series_of_bool_str(exog_values, dtype):
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is a pandas series of bool or str.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    exog = pd.Series(exog_values*10, name='exog', dtype=dtype)

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors_5,
                     window_size    = 5
                 )
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.],
                             [4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=0, stop=10, step=1),
            columns = ['custom_predictor_0', 'custom_predictor_1', 'custom_predictor_2', 
                       'custom_predictor_3', 'custom_predictor_4']
        ).assign(exog = exog_values*5 + exog_values*5, 
                 l1   = [1.]*5 + [0.]*5, 
                 l2   = [0.]*5 + [1.]*5).astype({'exog': dtype}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
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


@pytest.mark.parametrize("v_exog_1   , v_exog_2  , dtype", 
                         [([True]    , [False]   , bool), 
                          (['string'], ['string'], str)], 
                         ids = lambda dt : f'values, dtype: {dt}')
def test_create_train_X_y_output_when_series_10_and_exog_is_dataframe_of_bool_str(v_exog_1, v_exog_2, dtype):
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is a pandas dataframe with two columns of bool or str.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    exog = pd.DataFrame({'exog_1': v_exog_1*10,
                         'exog_2': v_exog_2*10})

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors_5,
                     name_predictors = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5'],
                     window_size     = 5
                 )
    results = forecaster.create_train_X_y(series=series, exog=exog)    

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.],
                             [4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=0, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(exog_1 = v_exog_1*5 + v_exog_1*5, 
                 exog_2 = v_exog_2*5 + v_exog_2*5, 
                 l1     = [1.]*5 + [0.]*5, 
                 l2     = [0.]*5 + [1.]*5).astype({'exog_1': dtype, 
                                                   'exog_2': dtype}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
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


def test_create_train_X_y_output_when_series_10_and_exog_is_series_of_category():
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is a pandas series of category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    exog = pd.Series(range(10), name='exog', dtype='category')

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors_5,
                     name_predictors = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5'],
                     window_size     = 5
                 )
    results = forecaster.create_train_X_y(series=series, exog=exog)   

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.],
                             [4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=0, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(exog = pd.Categorical([5, 6, 7, 8, 9]*2, categories=range(10)), 
                 l1   = [1.]*5 + [0.]*5, 
                 l2   = [0.]*5 + [1.]*5),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
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


def test_create_train_X_y_output_when_series_10_and_exog_is_dataframe_of_category():
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is a pandas dataframe with two columns of category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    exog = pd.DataFrame({'exog_1': pd.Categorical(range(10)),
                         'exog_2': pd.Categorical(range(100, 110))})

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors_5,
                     window_size    = 5
                 )
    results = forecaster.create_train_X_y(series=series, exog=exog)   

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.],
                             [4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=0, stop=10, step=1),
            columns = ['custom_predictor_0', 'custom_predictor_1', 'custom_predictor_2', 
                       'custom_predictor_3', 'custom_predictor_4']
        ).assign(exog_1 = pd.Categorical([5, 6, 7, 8, 9]*2, categories=range(10)),
                 exog_2 = pd.Categorical([105, 106, 107, 108, 109]*2, categories=range(100, 110)), 
                 l1     = [1.]*5 + [0.]*5, 
                 l2     = [0.]*5 + [1.]*5),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
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


def test_create_train_X_y_output_when_series_10_and_exog_is_dataframe_of_float_int_category():
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is a pandas dataframe with two columns of float, int, category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    exog = pd.DataFrame({'exog_1': pd.Series(np.arange(100, 110), dtype=float),
                         'exog_2': pd.Series(np.arange(1000, 1010), dtype=int),
                         'exog_3': pd.Categorical(range(100, 110))})

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors_5,
                     window_size    = 5
                 )
    results = forecaster.create_train_X_y(series=series, exog=exog)   

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105., 1005.],
                             [5., 4., 3., 2., 1., 106., 1006.],
                             [6., 5., 4., 3., 2., 107., 1007.],
                             [7., 6., 5., 4., 3., 108., 1008.],
                             [8., 7., 6., 5., 4., 109., 1009.],
                             [4., 3., 2., 1., 0., 105., 1005.],
                             [5., 4., 3., 2., 1., 106., 1006.],
                             [6., 5., 4., 3., 2., 107., 1007.],
                             [7., 6., 5., 4., 3., 108., 1008.],
                             [8., 7., 6., 5., 4., 109., 1009.]],
                             dtype=float),
            index   = pd.RangeIndex(start=0, stop=10, step=1),
            columns = ['custom_predictor_0', 'custom_predictor_1',
                       'custom_predictor_2', 'custom_predictor_3',
                       'custom_predictor_4', 'exog_1', 'exog_2']
        ).assign(exog_3 = pd.Categorical([105, 106, 107, 108, 109]*2, categories=range(100, 110)), 
                 l1     = [1.]*5 + [0.]*5, 
                 l2     = [0.]*5 + [1.]*5).astype({'exog_1': float, 
                                                   'exog_2': int}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
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


def test_create_train_X_y_output_when_series_and_exog_is_dataframe_datetime_index():
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is a pandas dataframe with two columns and datetime index.
    """
    series = pd.DataFrame({'1': np.arange(7, dtype=float), 
                           '2': np.arange(7, dtype=float)},
                           index = pd.date_range("1990-01-01", periods=7, freq='D'))
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors_3,
                     window_size    = 3
                 )

    results = forecaster.create_train_X_y(
                  series = series,
                  exog = pd.DataFrame(
                             {'exog_1' : np.arange(100, 107, dtype=float),
                              'exog_2' : np.arange(1000, 1007, dtype=float)},
                             index = pd.date_range("1990-01-01", periods=7, freq='D')
                         )           
              )

    expected = (
        pd.DataFrame(
            data = np.array([[2.0, 1.0, 0.0, 103., 1003., 1., 0.],
                             [3.0, 2.0, 1.0, 104., 1004., 1., 0.],
                             [4.0, 3.0, 2.0, 105., 1005., 1., 0.],
                             [5.0, 4.0, 3.0, 106., 1006., 1., 0.],
                             [2.0, 1.0, 0.0, 103., 1003., 0., 1.],
                             [3.0, 2.0, 1.0, 104., 1004., 0., 1.],
                             [4.0, 3.0, 2.0, 105., 1005., 0., 1.],
                             [5.0, 4.0, 3.0, 106., 1006., 0., 1.]]),
            index   = pd.RangeIndex(start=0, stop=8, step=1),
            columns = ['custom_predictor_0', 'custom_predictor_1', 'custom_predictor_2', 
                       'exog_1', 'exog_2', '1', '2']
        ),
        pd.Series(
            data  = np.array([3., 4., 5., 6., 3., 4., 5., 6.]),
            index = pd.RangeIndex(start=0, stop=8, step=1),
            name  = 'y',
            dtype = float),
        pd.date_range("1990-01-01", periods=7, freq='D'),
        pd.Index(pd.DatetimeIndex(['1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07', 
                                    '1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07']))
    )

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            np.testing.assert_array_equal(results[i], expected[i])


def test_create_train_X_y_output_when_series_10_and_transformer_series_is_StandardScaler():
    """
    Test the output of create_train_X_y when exog is None and transformer_series
    is StandardScaler.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = LinearRegression(),
                     fun_predictors     = create_predictors_5,
                     name_predictors    = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5'],
                     window_size        = 5,
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
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor        = LinearRegression(),
                     fun_predictors   = create_predictors_3,
                     window_size      = 3,
                     transformer_exog = StandardScaler()
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
            columns = ['custom_predictor_0', 'custom_predictor_1', 'custom_predictor_2', 
                       '1', '2']
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
                         ids = lambda tr : f'transformer_series: {tr}')
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
    
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = LinearRegression(),
                     fun_predictors     = create_predictors_3,
                     name_predictors    = ['lag_1', 'lag_2', 'lag_3'],
                     window_size        = 3,
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

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor       = LinearRegression(),
                     fun_predictors  = create_predictors_5,
                     name_predictors = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5'],
                     window_size     = 5
                 )
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

    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = LinearRegression(),
                     fun_predictors     = create_predictors_3,
                     window_size        = 3,
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
            columns = ['custom_predictor_0', 'custom_predictor_1', 'custom_predictor_2',
                       'exog_1', 'exog_2_a', 'exog_2_b', 'l1', 'l2', 'l3']
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