# Unit test create_train_X_y ForecasterAutoregMultiSeries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.exceptions import MissingValuesExogWarning
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def test_create_train_X_y_TypeError_when_exog_is_categorical_of_no_int():
    """
    Test TypeError is raised when exog is categorical with no int values.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(3)),  
                           '2': pd.Series(np.arange(3))})
    exog = pd.Series(['A', 'B', 'C'], name='exog', dtype='category')
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=2)

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
    series = pd.DataFrame({'1': pd.Series(np.arange(4)),  
                           '2': pd.Series(np.arange(4))})
    exog = pd.Series([1, 2, 3, np.nan], name='exog')
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=2)

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
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    series = pd.Series(np.arange(7))

    err_msg = re.escape(f"`series` must be a pandas DataFrame. Got {type(series)}.")
    with pytest.raises(TypeError, match = err_msg):
        forecaster.create_train_X_y(series=series)


def test_create_train_X_y_UserWarning_when_levels_of_transformer_series_not_equal_to_series_col_names():
    """
    Test UserWarning is raised when `transformer_series` is a dict and its keys are
    not the same as forecaster.series_col_names.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)),  
                           '2': pd.Series(np.arange(5))})
    dict_transformers = {'1': StandardScaler(), 
                         '3': StandardScaler()}
    
    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = LinearRegression(), 
                     lags               = 3,
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
                         [pd.Series(np.arange(10)), 
                          pd.DataFrame(np.arange(50).reshape(25, 2))])
def test_create_train_X_y_ValueError_when_series_and_exog_have_different_length(exog):
    """
    Test ValueError is raised when length of series is not equal to length exog.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(7)), 
                           '2': pd.Series(np.arange(7))})
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5)
    
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
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5)
    
    exog = pd.Series(np.arange(7), index=pd.RangeIndex(start=0, stop=7, step=1))

    err_msg = re.escape(
                ("Different index for `series` and `exog`. They must be equal "
                 "to ensure the correct alignment of values.")
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.fit(series=series, exog=exog)


def test_create_train_X_y_output_when_series_and_exog_is_None():
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is None.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(7, dtype=float)), 
                           '2': pd.Series(np.arange(7, dtype=float))})
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)

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
            index   = np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            columns = ['lag_1', 'lag_2', 'lag_3', '1', '2']
        ),
        pd.Series(
            data  = np.array([3., 4., 5., 6., 3., 4., 5., 6.]),
            index = np.array([0, 1, 2, 3, 4, 5, 6, 7]),
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

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5)
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
            index   = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'exog', '1', '2']
        ).astype({'exog': dtype}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
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

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5)
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
            index   = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'exog_1', 'exog_2', '1', '2']
        ).astype({'exog_1': dtype, 'exog_2': dtype}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
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

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5)
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
            index   = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(exog = exog_values*5 + exog_values*5, 
                 l1   = [1.]*5 + [0.]*5, 
                 l2   = [0.]*5 + [1.]*5).astype({'exog': dtype}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
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

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5)
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
            index   = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(exog_1 = v_exog_1*5 + v_exog_1*5, 
                 exog_2 = v_exog_2*5 + v_exog_2*5, 
                 l1     = [1.]*5 + [0.]*5, 
                 l2     = [0.]*5 + [1.]*5).astype({'exog_1': dtype, 
                                                   'exog_2': dtype}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
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

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5)
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
            index   = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(exog = pd.Categorical([5, 6, 7, 8, 9]*2, categories=range(10)), 
                 l1   = [1.]*5 + [0.]*5, 
                 l2   = [0.]*5 + [1.]*5),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
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

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5)
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
            index   = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(exog_1 = pd.Categorical([5, 6, 7, 8, 9]*2, categories=range(10)),
                 exog_2 = pd.Categorical([105, 106, 107, 108, 109]*2, categories=range(100, 110)), 
                 l1     = [1.]*5 + [0.]*5, 
                 l2     = [0.]*5 + [1.]*5),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
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
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)

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
            index   = np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'exog_1', 'exog_2', '1', '2']
        ),
        pd.Series(
            data  = np.array([3., 4., 5., 6., 3., 4., 5., 6.]),
            index = np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            name  = 'y',
            dtype = float
        ),
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
                     regressor        = LinearRegression(),
                     lags             = 3,
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
            index   = np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            columns = ['lag_1', 'lag_2', 'lag_3', '1', '2']
        ),
        pd.Series(
            data  = np.array([3., 4., 5., 6., 3., 4., 5., 6.]),
            index = np.array([0, 1, 2, 3, 4, 5, 6, 7]),
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
               'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 24.4, 87.2, 47.4, 23.8],
               'col_2': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b']},
                index = pd.date_range("1990-01-01", periods=10, freq='D'))

    transformer_exog = ColumnTransformer(
                            [('scale', StandardScaler(), ['col_1']),
                             ('onehot', OneHotEncoder(), ['col_2'])],
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
            columns = ['lag_1', 'lag_2', 'lag_3', 'col_1',
                       'col_2_a', 'col_2_b', '1', '2']
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