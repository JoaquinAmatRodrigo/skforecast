# Unit test create_train_X_y ForecasterAutoregCustom
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.preprocessing import TimeSeriesDifferentiator
from skforecast.exceptions import MissingValuesExogWarning
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Fixtures
from .fixtures_ForecasterAutoregCustom import data # to test results when using differentiation


def create_predictors(y): # pragma: no cover
    """
    Create first 5 lags of a time series.
    """
    lags = y[-1:-6:-1]

    return lags


def test_create_train_X_y_TypeError_when_exog_is_categorical_of_no_int():
    """
    Test TypeError is raised when exog is categorical with no int values.
    """
    y = pd.Series(np.arange(10))
    exog = pd.Series(['A', 'B']*5, name='exog', dtype='category')
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )
    
    err_msg = re.escape(
                ("If exog is of type category, it must contain only integer values. "
                 "See skforecast docs for more info about how to include categorical "
                 "features https://skforecast.org/"
                 "latest/user_guides/categorical-features.html")
              )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.create_train_X_y(y=y, exog=exog)


def test_create_train_X_y_MissingValuesExogWarning_when_exog_has_missing_values():
    """
    Test create_train_X_y is issues a MissingValuesExogWarning when exog has missing values.
    """
    y = pd.Series(np.arange(10))
    exog = pd.Series([1, 2, 3, np.nan, 5]*2, name='exog')
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )
    
    warn_msg = re.escape(
                ("`exog` has missing values. Most machine learning models do "
                 "not allow missing values. Fitting the forecaster may fail.")  
              )
    with pytest.warns(MissingValuesExogWarning, match = warn_msg):
        forecaster.create_train_X_y(y=y, exog=exog)


def test_create_train_X_y_ValueError_when_len_y_is_less_than_window_size():
    """
    Test ValueError is raised when length of y is less than self.window_size + 1.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 10
                 )
                 
    err_msg = re.escape(
                (f'`y` must have as many values as the windows_size needed by '
                 f'{forecaster.fun_predictors.__name__}. For this Forecaster the '
                 f'minimum length is {forecaster.window_size + 1}')
            )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.create_train_X_y(y=pd.Series(np.arange(5)))


@pytest.mark.parametrize("y                        , exog", 
                         [(pd.Series(np.arange(50)), pd.Series(np.arange(10))), 
                          (pd.Series(np.arange(10)), pd.Series(np.arange(50))), 
                          (pd.Series(np.arange(10)), pd.DataFrame(np.arange(50).reshape(25,2)))])
def test_create_train_X_y_ValueError_when_len_y_is_different_from_len_exog(y, exog):
    """
    Test ValueError is raised when length of y is not equal to length exog.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )

    err_msg = re.escape(
                (f'`exog` must have same number of samples as `y`. '
                 f'length `exog`: ({len(exog)}), length `y`: ({len(y)})')
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.create_train_X_y(y=y, exog=exog)


def test_create_train_X_y_ValueError_when_y_and_exog_have_different_index():
    """
    Test ValueError is raised when y and exog have different index.
    """
    forecaster = ForecasterAutoregCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5
                )

    err_msg = re.escape(
                    ('Different index for `y` and `exog`. They must be equal '
                     'to ensure the correct alignment of values.')      
                )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.fit(
            y=pd.Series(np.arange(10), index=pd.date_range(start='2022-01-01', periods=10, freq='1D')),
            exog=pd.Series(np.arange(10), index=pd.RangeIndex(start=0, stop=10, step=1))
        )


def test_create_train_X_y_ValueError_when_len_name_predictors_not_match_X_train_columns():
    """
    Test ValueError is raised when argument `name_predictors` has less values than the number of
    columns of X_train.
    """
    forecaster = ForecasterAutoregCustom(
                    regressor       = LinearRegression(),
                    fun_predictors  = create_predictors,
                    name_predictors = ['lag_1', 'lag_2'],
                    window_size     = 5
                 )

    err_msg = re.escape(
                    (f"The length of provided predictors names (`name_predictors`) do not "
                     f"match the number of columns created by `fun_predictors()`.")
                )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.fit(
            y = pd.Series(np.arange(10), index=pd.date_range(start='2022-01-01', periods=10, freq='1D'))
        )
        
        
def test_create_train_X_y_ValueError_fun_predictors_return_nan_values():
    """
    Test ValueError is raised when fun_predictors returns `NaN` values.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = lambda y: np.array([np.nan, np.nan]),
                     window_size    = 5
                 )
    
    err_msg = re.escape("`fun_predictors()` is returning `NaN` values.")
    with pytest.raises(ValueError, match = err_msg):
        forecaster.fit(y=pd.Series(np.arange(50)))


def test_create_train_X_y_ValueError_when_forecaster_window_size_does_not_match_with_fun_predictors():
    """
    Test ValueError is raised when the window needed by `fun_predictors()` does 
    not correspond with the forecaster.window_size.
    """
    forecaster = ForecasterAutoregCustom(
                    regressor       = LinearRegression(),
                    fun_predictors  = create_predictors,
                    window_size     = 4
                 )

    err_msg = re.escape(
                (f"The `window_size` argument ({forecaster.window_size}), declared when "
                 f"initializing the forecaster, does not correspond to the window "
                 f"used by `fun_predictors()`.")
            )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.fit(
            y = pd.Series(np.arange(10), index=pd.date_range(start='2022-01-01', periods=10, freq='1D'))
        )


def test_create_train_X_y_column_names_match_name_predictors():
    """
    Check column names in X_train match the ones in argument `name_predictors`.
    """
    forecaster = ForecasterAutoregCustom(
                    regressor       = LinearRegression(),
                    fun_predictors  = create_predictors,
                    name_predictors = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5'],
                    window_size     = 5
                 )

    X_train = forecaster.create_train_X_y(
                  y = pd.Series(np.arange(10), index=pd.date_range(start='2022-01-01', periods=10, freq='1D'))
              )[0]
    
    assert X_train.columns.to_list() == ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
    

def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_None():
    """
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is None.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )
    results = forecaster.create_train_X_y(y=pd.Series(np.arange(10, dtype=float)))
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['custom_predictor_0', 'custom_predictor_1',
                       'custom_predictor_2', 'custom_predictor_3',
                       'custom_predictor_4']
        ),
        pd.Series(
            data  = np.array([5., 6., 7., 8., 9.]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_create_train_X_y_output_when_y_and_exog_no_pandas_index():
    """
    Test the output of create_train_X_y when y and exog have no pandas index 
    that doesn't start at 0.
    """
    y = pd.Series(np.arange(10), index=np.arange(3, 13), dtype=float)
    exog = pd.Series(np.arange(100, 110), index=np.arange(3, 13) , 
                     name='exog', dtype=float)
    
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )
    results = forecaster.create_train_X_y(y=y, exog=exog)
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105.],
                             [5., 4., 3., 2., 1., 106.],
                             [6., 5., 4., 3., 2., 107.],
                             [7., 6., 5., 4., 3., 108.],
                             [8., 7., 6., 5., 4., 109.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['custom_predictor_0', 'custom_predictor_1',
                       'custom_predictor_2', 'custom_predictor_3',
                       'custom_predictor_4', 'exog']
        ).astype({'exog': float}),
        pd.Series(
            data  = np.array([5., 6., 7., 8., 9.]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt : f'dtype: {dt}')
def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_series_of_float_int(dtype):
    """
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas series of floats or ints.
    """
    y = pd.Series(np.arange(10), dtype=float)
    exog = pd.Series(np.arange(100, 110), name='exog',  dtype=dtype)
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )
    results = forecaster.create_train_X_y(y=y, exog=exog)
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105.],
                             [5., 4., 3., 2., 1., 106.],
                             [6., 5., 4., 3., 2., 107.],
                             [7., 6., 5., 4., 3., 108.],
                             [8., 7., 6., 5., 4., 109.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['custom_predictor_0', 'custom_predictor_1',
                       'custom_predictor_2', 'custom_predictor_3',
                       'custom_predictor_4', 'exog']
        ).astype({'exog': dtype}),
        pd.Series(
            data  = np.array([5., 6., 7., 8., 9.]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt : f'dtype: {dt}')
def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_dataframe_of_float_int(dtype):
    """
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas dataframe with two columns of floats or ints.
    """
    y = pd.Series(np.arange(10), dtype=float)
    exog = pd.DataFrame({
               'exog_1': np.arange(100, 110, dtype=dtype),
               'exog_2': np.arange(1000, 1010, dtype=dtype)
           })
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )
    results = forecaster.create_train_X_y(y=y, exog=exog)     
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105., 1005.],
                             [5., 4., 3., 2., 1., 106., 1006.],
                             [6., 5., 4., 3., 2., 107., 1007.],
                             [7., 6., 5., 4., 3., 108., 1008.],
                             [8., 7., 6., 5., 4., 109., 1009.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['custom_predictor_0', 'custom_predictor_1',
                       'custom_predictor_2', 'custom_predictor_3',
                       'custom_predictor_4', 'exog_1', 'exog_2']
        ).astype({'exog_1': dtype, 'exog_2': dtype}),
        pd.Series(
            data  = np.array([5., 6., 7., 8., 9.]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


@pytest.mark.parametrize("exog_values, dtype", 
                         [([True]    , bool), 
                          (['string'], str)], 
                         ids = lambda dt : f'values, dtype: {dt}')
def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_series_of_bool_str(exog_values, dtype):
    """
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas series of bool or str.
    """
    y = pd.Series(np.arange(10), dtype=float)
    exog = pd.Series(exog_values*10, name='exog', dtype=dtype)
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )
    results = forecaster.create_train_X_y(y=y, exog=exog)
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['custom_predictor_0', 'custom_predictor_1',
                       'custom_predictor_2', 'custom_predictor_3',
                       'custom_predictor_4']
        ).assign(exog=exog_values*5).astype({'exog': dtype}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


@pytest.mark.parametrize("v_exog_1   , v_exog_2  , dtype", 
                         [([True]    , [False]   , bool), 
                          (['string'], ['string'], str)], 
                         ids = lambda dt : f'values, dtype: {dt}')
def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_dataframe_of_bool_str(v_exog_1, v_exog_2, dtype):
    """
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas dataframe with two columns of bool or str.
    """
    y = pd.Series(np.arange(10), dtype=float)
    exog = pd.DataFrame({
               'exog_1': v_exog_1*10,
               'exog_2': v_exog_2*10,
           })
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )
    results = forecaster.create_train_X_y(y=y, exog=exog)
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['custom_predictor_0', 'custom_predictor_1',
                       'custom_predictor_2', 'custom_predictor_3',
                       'custom_predictor_4']
        ).assign(exog_1=v_exog_1*5, exog_2=v_exog_2*5).astype({'exog_1': dtype, 'exog_2': dtype}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_series_of_category():
    """
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas series of category.
    """
    y = pd.Series(np.arange(10), dtype=float)
    exog = pd.Series(range(10), name='exog', dtype='category')
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )
    results = forecaster.create_train_X_y(y=y, exog=exog)
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['custom_predictor_0', 'custom_predictor_1',
                       'custom_predictor_2', 'custom_predictor_3',
                       'custom_predictor_4']
        ).assign(exog=pd.Categorical(range(5, 10), categories=range(10))),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_dataframe_of_category():
    """
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas dataframe with two columns of category.
    """
    y = pd.Series(np.arange(10), dtype=float)
    exog = pd.DataFrame({
               'exog_1': pd.Categorical(range(10)),
               'exog_2': pd.Categorical(range(100, 110)),
           })
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )
    results = forecaster.create_train_X_y(y=y, exog=exog)
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['custom_predictor_0', 'custom_predictor_1',
                       'custom_predictor_2', 'custom_predictor_3',
                       'custom_predictor_4']
        ).assign(
            exog_1=pd.Categorical(range(5, 10), categories=range(10)),
            exog_2=pd.Categorical(range(105, 110), categories=range(100, 110))
        ),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_dataframe_of_float_int_category():
    """
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas dataframe with 3 columns of float, int, category.
    """
    y = pd.Series(np.arange(10), dtype=float)
    exog = pd.DataFrame({'exog_1': pd.Series(np.arange(100, 110), dtype=float),
                         'exog_2': pd.Series(np.arange(1000, 1010), dtype=int),
                         'exog_3': pd.Categorical(range(100, 110))})
    
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5
                 )
    results = forecaster.create_train_X_y(y=y, exog=exog) 
           
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105., 1005.],
                             [5., 4., 3., 2., 1., 106., 1006.],
                             [6., 5., 4., 3., 2., 107., 1007.],
                             [7., 6., 5., 4., 3., 108., 1008.],
                             [8., 7., 6., 5., 4., 109., 1009.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['custom_predictor_0', 'custom_predictor_1',
                       'custom_predictor_2', 'custom_predictor_3',
                       'custom_predictor_4', 'exog_1', 'exog_2']
        ).astype({'exog_1': float, 
                  'exog_2': int}).assign(exog_3=pd.Categorical(range(105, 110), categories=range(100, 110))),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_create_train_X_y_output_when_y_is_series_10_and_transformer_y_is_StandardScaler():
    """
    Test the output of create_train_X_y when exog is None and transformer_y
    is StandardScaler.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = create_predictors,
                     window_size    = 5,
                     transformer_y  = StandardScaler()
                 )
    results = forecaster.create_train_X_y(y=pd.Series(np.arange(10), dtype=float))
    expected = (
        pd.DataFrame(
            data = np.array([[-0.17407766, -0.52223297, -0.87038828, -1.21854359, -1.5666989 ],
                             [0.17407766, -0.17407766, -0.52223297, -0.87038828, -1.21854359],
                             [0.52223297,  0.17407766, -0.17407766, -0.52223297, -0.87038828],
                             [0.87038828,  0.52223297,  0.17407766, -0.17407766, -0.52223297],
                             [1.21854359,  0.87038828,  0.52223297,  0.17407766, -0.17407766]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['custom_predictor_0', 'custom_predictor_1',
                       'custom_predictor_2', 'custom_predictor_3',
                       'custom_predictor_4']
        ),
        pd.Series(
            data  = np.array([0.17407766, 0.52223297, 0.87038828, 1.21854359, 1.5666989]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_create_train_X_y_output_when_exog_is_None_and_transformer_exog_is_not_None():
    """
    Test the output of create_train_X_y when exog is None and transformer_exog
    is not None.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor        = LinearRegression(),
                     fun_predictors   = create_predictors,
                     window_size      = 5,
                     transformer_exog = StandardScaler()
                 )
    results = forecaster.create_train_X_y(y=pd.Series(np.arange(10), dtype=float))
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['custom_predictor_0', 'custom_predictor_1',
                       'custom_predictor_2', 'custom_predictor_3',
                       'custom_predictor_4']
        ),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        )
    )
    
    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_create_train_X_y_output_when_transformer_y_and_transformer_exog():
    """
    Test the output of create_train_X_y when using transformer_y and transformer_exog.
    """
    y = pd.Series(np.arange(8), dtype=float)
    y.index = pd.date_range("1990-01-01", periods=8, freq='D')
    exog = pd.DataFrame({
               'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4],
               'col_2': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']},
               index = pd.date_range("1990-01-01", periods=8, freq='D')
           )

    transformer_y = StandardScaler()
    transformer_exog = ColumnTransformer(
                            [('scale', StandardScaler(), ['col_1']),
                             ('onehot', OneHotEncoder(), ['col_2'])],
                            remainder = 'passthrough',
                            verbose_feature_names_out = False
                        )

    forecaster = ForecasterAutoregCustom(
                     regressor        = LinearRegression(),
                     fun_predictors   = create_predictors,
                     window_size      = 5,
                     transformer_y    = transformer_y,
                     transformer_exog = transformer_exog
                 )
    results = forecaster.create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[0.21821789, -0.21821789, -0.65465367, -1.09108945,
                              -1.52752523, -0.25107995,  0.        ,  1.        ],
                             [0.65465367,  0.21821789, -0.21821789, -0.65465367,
                              -1.09108945, 1.79326881,  0.        ,  1.        ],
                             [1.09108945,  0.65465367,  0.21821789, -0.21821789,
                              -0.65465367, 0.01673866,  0.        ,  1.        ]]),
            index   = pd.date_range("1990-01-06", periods=3, freq='D'),
            columns = ['custom_predictor_0', 'custom_predictor_1',
                       'custom_predictor_2', 'custom_predictor_3',
                       'custom_predictor_4', 'col_1', 'col_2_a', 'col_2_b']
        ),
        pd.Series(
            data  = np.array([0.65465367, 1.09108945, 1.52752523]),
            index = pd.date_range("1990-01-06", periods=3, freq='D'),
            name  = 'y',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_create_train_X_y_output_when_y_is_series_exog_is_series_and_differentiation_is_1():
    """
    Test the output of create_train_X_y when using differentiation=1.
    """
    # Data differentiated
    diferenciator = TimeSeriesDifferentiator(order=1)
    data_diff = diferenciator.fit_transform(data)
    data_diff = pd.Series(data_diff, index=data.index).dropna()
    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff = exog.iloc[1:]
    end_train = '2003-03-01 23:59:00'

    forecaster_1 = ForecasterAutoregCustom(
                        regressor        = LinearRegression(),
                        fun_predictors   = create_predictors,
                        window_size      = 5
                   )
    forecaster_2 = ForecasterAutoregCustom(
                     regressor        = LinearRegression(),
                     fun_predictors   = create_predictors,
                     window_size      = 5,
                     differentiation  = 1
                  )
    X_train_1, y_train_1 = forecaster_1.create_train_X_y(
                                    data_diff.loc[:end_train],
                                    exog=exog_diff.loc[:end_train]
                            )
    X_train_2, y_train_2 = forecaster_2.create_train_X_y(
                            data.loc[:end_train],
                            exog=exog.loc[:end_train]
                        )
    pd.testing.assert_frame_equal(X_train_1, X_train_2, check_names=True)
    pd.testing.assert_series_equal(y_train_1, y_train_2, check_names=True)


def test_create_train_X_y_output_when_y_is_series_10_exog_is_series_and_differentiation_is_2():
    """
    Test the output of create_train_X_y when using differentiation=1.
    """

    # Data differentiated
    diferenciator = TimeSeriesDifferentiator(order=2)
    data_diff_2 = diferenciator.fit_transform(data)
    data_diff_2 = pd.Series(data_diff_2, index=data.index).dropna()
    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff_2 = exog.iloc[2:]
    end_train = '2003-03-01 23:59:00'

    forecaster_1 = ForecasterAutoregCustom(
                        regressor        = LinearRegression(),
                        fun_predictors   = create_predictors,
                        window_size      = 5
                   )
    forecaster_2 = ForecasterAutoregCustom(
                     regressor        = LinearRegression(),
                     fun_predictors   = create_predictors,
                     window_size      = 5,
                     differentiation  = 2
                  )
    X_train_1, y_train_1 = forecaster_1.create_train_X_y(
                                    data_diff_2.loc[:end_train],
                                    exog=exog_diff_2.loc[:end_train]
                            )
    X_train_2, y_train_2 = forecaster_2.create_train_X_y(
                            data.loc[:end_train],
                            exog=exog.loc[:end_train]
                        )
    pd.testing.assert_frame_equal(X_train_1, X_train_2, check_names=True)
    pd.testing.assert_series_equal(y_train_1, y_train_2, check_names=True)