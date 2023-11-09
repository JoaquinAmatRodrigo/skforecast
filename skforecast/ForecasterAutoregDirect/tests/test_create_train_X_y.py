# Unit test create_train_X_y ForecasterAutoregDirect
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.exceptions import MissingValuesExogWarning
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def test_create_train_X_y_TypeError_when_exog_is_categorical_of_no_int():
    """
    Test TypeError is raised when exog is categorical with no int values.
    """
    y = pd.Series(np.arange(6))
    exog = pd.Series(['A', 'B', 'C', 'A', 'B', 'C'], name='exog', dtype='category')
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=2, steps=2)

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
    y = pd.Series(np.arange(6))
    exog = pd.Series([1, 2, 3, np.nan, 5, 6], name='exog')
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=2, steps=2)

    warn_msg = re.escape(
                ("`exog` has missing values. Most machine learning models do "
                 "not allow missing values. Fitting the forecaster may fail.")  
              )
    with pytest.warns(MissingValuesExogWarning, match = warn_msg):
        forecaster.create_train_X_y(y=y, exog=exog)


def test_create_train_X_y_ValueError_when_len_y_is_lower_than_maximum_lag_plus_steps():
    """
    Test ValueError is raised when length of y is lower than maximum lag plus
    number of steps included in the forecaster.
    """
    y = pd.Series(np.arange(5), name='y')
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=3)

    err_msg = re.escape(
                (f"Minimum length of `y` for training this forecaster is "
                 f"{forecaster.max_lag + forecaster.steps}. Got {len(y)}. Reduce the "
                 f"number of predicted steps, {forecaster.steps}, or the maximum "
                 f"lag, {forecaster.max_lag}, if no more data is available.")
            )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.create_train_X_y(y=y)


@pytest.mark.parametrize("y                        , exog", 
                         [(pd.Series(np.arange(50)), pd.Series(np.arange(10))), 
                          (pd.Series(np.arange(10)), pd.Series(np.arange(50))), 
                          (pd.Series(np.arange(10)), pd.DataFrame(np.arange(50).reshape(25,2)))])
def test_create_train_X_y_ValueError_when_len_y_is_different_from_len_exog(y, exog):
    """
    Test ValueError is raised when length of y is not equal to length exog.
    """
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=3)

    err_msg = re.escape(
                (f"`exog` must have same number of samples as `y`. "
                 f"length `exog`: ({len(exog)}), length `y`: ({len(y)})")
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.create_train_X_y(y=y, exog=exog)


def test_create_train_X_y_ValueError_when_y_and_exog_have_different_index():
    """
    Test ValueError is raised when y and exog have different index.
    """
    y = pd.Series(np.arange(10), name='y')
    y.index = pd.date_range(start='2022-01-01', periods=10, freq='1D')
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=3)

    err_msg = re.escape(
                ("Different index for `y` and `exog`. They must be equal "
                 "to ensure the correct alignment of values.") 
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.fit(
            y    = y,
            exog = pd.Series(np.arange(10), index=pd.RangeIndex(start=0, stop=10, step=1))
        )


def test_create_train_X_y_output_when_lags_3_steps_1_and_exog_is_None():
    """
    Test output of create_train_X_y when regressor is LinearRegression, 
    lags is 3 and steps is 1.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = None

    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=1)
    results = forecaster.create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[2., 1., 0.],
                             [3., 2., 1.],
                             [4., 3., 2.],
                             [5., 4., 3.],
                             [6., 5., 4.],
                             [7., 6., 5.],
                             [8., 7., 6.]], dtype=float),
            index   = pd.RangeIndex(start=3, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3']
        ),
        {1: pd.Series(
                data  = np.array([3., 4., 5., 6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=3, stop=10, step=1),
                name  = "y_step_1"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


def test_create_train_X_y_output_when_interspersed_lags_steps_2_and_exog_is_None():
    """
    Test output of create_train_X_y when regressor is LinearRegression, 
    interspersed_lags and steps is 2.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = None

    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=[1, 3], steps=2)
    results = forecaster.create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[2., 0.],
                             [3., 1.],
                             [4., 2.],
                             [5., 3.],
                             [6., 4.],
                             [7., 5.]], dtype=float),
            index   = pd.RangeIndex(start=4, stop=10, step=1),
            columns = ['lag_1', 'lag_3']
        ),
        {1: pd.Series(
                data  = np.array([3., 4., 5., 6., 7., 8.], dtype=float), 
                index = pd.RangeIndex(start=3, stop=9, step=1),
                name  = "y_step_1"
            ),
         2: pd.Series(
                data  = np.array([4., 5., 6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=4, stop=10, step=1),
                name  = "y_step_2"
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
    y = pd.Series(np.arange(10), index=np.arange(6, 16), dtype=float)
    exog = pd.Series(np.arange(100, 110), index=np.arange(6, 16), 
                     name='exog', dtype=float)
    
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=3, steps=1)
    results = forecaster.create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[2., 1., 0., 103.],
                             [3., 2., 1., 104.],
                             [4., 3., 2., 105.],
                             [5., 4., 3., 106.],
                             [6., 5., 4., 107.],
                             [7., 6., 5., 108.],
                             [8., 7., 6., 109.]], dtype=float),
            index   = pd.RangeIndex(start=3, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'exog_step_1']
        ),
        {1: pd.Series(
                data  = np.array([3., 4., 5., 6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=3, stop=10, step=1),
                name  = "y_step_1"
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
def test_create_train_X_y_output_when_y_is_series_10_steps_1_and_exog_is_series_of_float_int(dtype):
    """
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)), steps=1 
    and exog is a pandas Series of floats or ints.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=dtype)

    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=5, steps=1)
    results = forecaster.create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105.],
                             [5., 4., 3., 2., 1., 106.],
                             [6., 5., 4., 3., 2., 107.],
                             [7., 6., 5., 4., 3., 108.],
                             [8., 7., 6., 5., 4., 109.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'exog_step_1']
        ).astype({'exog_step_1': dtype}),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "y_step_1"
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
def test_create_train_X_y_output_when_y_is_series_10_steps_2_and_exog_is_series_of_float_int(dtype):
    """
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)), steps=2 
    and exog is a pandas Series of floats or ints.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=dtype)
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=5, steps=2)
    results = forecaster.create_train_X_y(y=y, exog=exog)
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105., 106.],
                             [5., 4., 3., 2., 1., 106., 107.],
                             [6., 5., 4., 3., 2., 107., 108.],
                             [7., 6., 5., 4., 3., 108., 109.]], dtype=float),
            index   = pd.RangeIndex(start=6, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'exog_step_1', 'exog_step_2']
        ).astype({'exog_step_1': dtype, 'exog_step_2': dtype}),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=9, step=1),
                name  = "y_step_1"
            ),
         2: pd.Series(
                data  = np.array([6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=6, stop=10, step=1),
                name  = "y_step_2"
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
def test_create_train_X_y_output_when_y_is_series_10_steps_1_and_exog_is_dataframe_of_float_int(dtype):
    """
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)), steps=1 
    and exog is a pandas DataFrame of floats or ints.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.DataFrame({'exog_1': np.arange(100, 110, dtype=dtype),
                         'exog_2': np.arange(1000, 1010, dtype=dtype)})
    
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=5, steps=1)
    results = forecaster.create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105., 1005.],
                             [5., 4., 3., 2., 1., 106., 1006.],
                             [6., 5., 4., 3., 2., 107., 1007.],
                             [7., 6., 5., 4., 3., 108., 1008.],
                             [8., 7., 6., 5., 4., 109., 1009.]], dtype=float),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'exog_1_step_1', 'exog_2_step_1']
        ).astype({'exog_1_step_1': dtype, 'exog_2_step_1': dtype}),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "y_step_1"
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
def test_create_train_X_y_output_when_y_is_series_10_steps_3_and_exog_is_dataframe_of_float_int(dtype):
    """
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)), steps=3 
    and exog is a pandas DataFrame of floats or ints.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.DataFrame({'exog_1': np.arange(100, 110, dtype=dtype),
                         'exog_2': np.arange(1000, 1010, dtype=dtype)})
    
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=5, steps=3)
    results = forecaster.create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4, 3, 2, 1, 0, 105, 1005, 106, 1006, 107, 1007],
                             [5, 4, 3, 2, 1, 106, 1006, 107, 1007, 108, 1008],
                             [6, 5, 4, 3, 2, 107, 1007, 108, 1008, 109, 1009]],
                            dtype=float),
            index = pd.RangeIndex(start=7, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'exog_1_step_1', 'exog_2_step_1', 
                       'exog_1_step_2', 'exog_2_step_2', 
                       'exog_1_step_3', 'exog_2_step_3']
        ).astype({'exog_1_step_1': dtype, 'exog_2_step_1': dtype, 
                  'exog_1_step_2': dtype, 'exog_2_step_2': dtype, 
                  'exog_1_step_3': dtype, 'exog_2_step_3': dtype}),
        {1: pd.Series(
                data  = np.array([5., 6., 7.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=8, step=1),
                name  = "y_step_1"
            ),
         2: pd.Series(
                data  = np.array([6., 7., 8.], dtype=float), 
                index = pd.RangeIndex(start=6, stop=9, step=1),
                name  = "y_step_2"
            ),
         3: pd.Series(
                data  = np.array([7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=7, stop=10, step=1),
                name  = "y_step_3"
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
def test_create_train_X_y_output_when_y_is_series_10_steps_1_and_exog_is_series_of_bool_str(exog_values, dtype):
    """
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)), steps=1 
    and exog is a pandas Series of bool or str.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.Series(exog_values*10, name='exog', dtype=dtype)

    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=5, steps=1)
    results = forecaster.create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]], dtype=float),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(exog_step_1=exog_values*5).astype({'exog_step_1': dtype}),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "y_step_1"
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
def test_create_train_X_y_output_when_y_is_series_10_steps_2_and_exog_is_series_of_bool_str(exog_values, dtype):
    """
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)), steps=2
    and exog is a pandas Series of bool or str.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.Series(exog_values*10, name='exog', dtype=dtype)

    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=5, steps=2)
    results = forecaster.create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.]], dtype=float),
            index   = pd.RangeIndex(start=6, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(exog_step_1=exog_values*4,
                 exog_step_2=exog_values*4).astype({'exog_step_1': dtype, 
                                                    'exog_step_2': dtype}),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=9, step=1),
                name  = "y_step_1"
            ),
         2: pd.Series(
                data  = np.array([6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=6, stop=10, step=1),
                name  = "y_step_2"
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
def test_create_train_X_y_output_when_y_is_series_10_steps_1_and_exog_is_dataframe_of_bool_str(v_exog_1, v_exog_2, dtype):
    """
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)), steps=1 
    and exog is a pandas dataframe with two columns of bool or str.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.DataFrame({'exog_1': v_exog_1*10,
                         'exog_2': v_exog_2*10})
    
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=5, steps=1)
    results = forecaster.create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(exog_1_step_1=v_exog_1*5,
                 exog_2_step_1=v_exog_2*5).astype({'exog_1_step_1': dtype, 
                                                   'exog_2_step_1': dtype}),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "y_step_1"
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
def test_create_train_X_y_output_when_y_is_series_10_steps_3_and_exog_is_dataframe_of_bool_str(v_exog_1, v_exog_2, dtype):
    """
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)), steps=3 
    and exog is a pandas dataframe with two columns of bool or str.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.DataFrame({'exog_1': v_exog_1*10,
                         'exog_2': v_exog_2*10})
    
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=5, steps=3)
    results = forecaster.create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.]],
                            dtype=float),
            index = pd.RangeIndex(start=7, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(exog_1_step_1=v_exog_1*3,
                 exog_2_step_1=v_exog_2*3,
                 exog_1_step_2=v_exog_1*3,
                 exog_2_step_2=v_exog_2*3,
                 exog_1_step_3=v_exog_1*3,
                 exog_2_step_3=v_exog_2*3
        ).astype({'exog_1_step_1': dtype, 
                  'exog_2_step_1': dtype, 
                  'exog_1_step_2': dtype, 
                  'exog_2_step_2': dtype, 
                  'exog_1_step_3': dtype,
                  'exog_2_step_3': dtype}
        ),
        {1: pd.Series(
                data  = np.array([5., 6., 7.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=8, step=1),
                name  = "y_step_1"
            ),
         2: pd.Series(
                data  = np.array([6., 7., 8.], dtype=float), 
                index = pd.RangeIndex(start=6, stop=9, step=1),
                name  = "y_step_2"
            ),
         3: pd.Series(
                data  = np.array([7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=7, stop=10, step=1),
                name  = "y_step_3"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    

def test_create_train_X_y_output_when_y_is_series_10_steps_1_and_exog_is_series_of_category():
    """
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)), steps=1 
    and exog is a pandas Series of category.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.Series(range(10), name='exog', dtype='category')

    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=5, steps=1)
    results = forecaster.create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(exog_step_1=pd.Categorical(range(5, 10), categories=range(10))),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "y_step_1"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


def test_create_train_X_y_output_when_y_is_series_10_steps_2_and_exog_is_series_of_category():
    """
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)), steps=2 
    and exog is a pandas Series of category.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.Series(range(10), name='exog', dtype='category')

    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=5, steps=2)
    results = forecaster.create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.]]),
            index   = pd.RangeIndex(start=6, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(exog_step_1=pd.Categorical(range(5, 9), categories=range(10)),
                 exog_step_2=pd.Categorical(range(6, 10), categories=range(10))),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=9, step=1),
                name  = "y_step_1"
            ),
         2: pd.Series(
                data  = np.array([6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=6, stop=10, step=1),
                name  = "y_step_2"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


def test_create_train_X_y_output_when_y_is_series_10_steps_1_and_exog_is_dataframe_of_category():
    """
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)), steps=1 
    and exog is a pandas DataFrame of category.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.DataFrame({'exog_1': pd.Categorical(range(10)),
                         'exog_2': pd.Categorical(range(100, 110))})
    
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=5, steps=1)
    results = forecaster.create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(
            exog_1_step_1=pd.Categorical(range(5, 10), categories=range(10)),
            exog_2_step_1=pd.Categorical(range(105, 110), categories=range(100, 110))
        ),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "y_step_1"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


def test_create_train_X_y_output_when_y_is_series_10_steps_3_and_exog_is_dataframe_of_category():
    """
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)), steps=3 
    and exog is a pandas DataFrame of category.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.DataFrame({'exog_1': pd.Categorical(range(10)),
                         'exog_2': pd.Categorical(range(100, 110))})
    
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=5, steps=3)
    results = forecaster.create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.]],
                            dtype=float),
            index = pd.RangeIndex(start=7, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(
            exog_1_step_1=pd.Categorical(range(5, 8), categories=range(10)),
            exog_2_step_1=pd.Categorical(range(105, 108), categories=range(100, 110)),
            exog_1_step_2=pd.Categorical(range(6, 9), categories=range(10)),
            exog_2_step_2=pd.Categorical(range(106, 109), categories=range(100, 110)),
            exog_1_step_3=pd.Categorical(range(7, 10), categories=range(10)),
            exog_2_step_3=pd.Categorical(range(107, 110), categories=range(100, 110))
        ),
        {1: pd.Series(
                data  = np.array([5., 6., 7.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=8, step=1),
                name  = "y_step_1"
            ),
         2: pd.Series(
                data  = np.array([6., 7., 8.], dtype=float), 
                index = pd.RangeIndex(start=6, stop=9, step=1),
                name  = "y_step_2"
            ),
         3: pd.Series(
                data  = np.array([7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=7, stop=10, step=1),
                name  = "y_step_3"
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
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)), steps=1 
    and exog is a pandas DataFrame of float, int and category.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.DataFrame({'exog_1': pd.Series(np.arange(100, 110), dtype=float),
                         'exog_2': pd.Series(np.arange(1000, 1010), dtype=int),
                         'exog_3': pd.Categorical(range(100, 110))})
    
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=5, steps=1)
    results = forecaster.create_train_X_y(y=y, exog=exog)        
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105., 1005.],
                             [5., 4., 3., 2., 1., 106., 1006.],
                             [6., 5., 4., 3., 2., 107., 1007.],
                             [7., 6., 5., 4., 3., 108., 1008.],
                             [8., 7., 6., 5., 4., 109., 1009.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'exog_1_step_1', 'exog_2_step_1']
        ).astype({'exog_1_step_1': float, 
                  'exog_2_step_1': int}).assign(exog_3_step_1=pd.Categorical(range(105, 110), categories=range(100, 110))
        ),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "y_step_1"
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
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)), steps=3 
    and exog is a pandas DataFrame of float, int and category.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.DataFrame({'exog_1': pd.Series(np.arange(100, 110), dtype=float),
                         'exog_2': pd.Series(np.arange(1000, 1010), dtype=int),
                         'exog_3': pd.Categorical(range(100, 110))})
    
    forecaster = ForecasterAutoregDirect(LinearRegression(), lags=5, steps=3)
    results = forecaster.create_train_X_y(y=y, exog=exog)        
    expected = (
        pd.DataFrame(
            data = np.array([[4, 3, 2, 1, 0, 105, 1005, 105, 106, 1006, 106, 107, 1007, 107],
                             [5, 4, 3, 2, 1, 106, 1006, 106, 107, 1007, 107, 108, 1008, 108],
                             [6, 5, 4, 3, 2, 107, 1007, 107, 108, 1008, 108, 109, 1009, 109]],
                            dtype=float),
            index   = pd.RangeIndex(start=7, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
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
                name  = "y_step_1"
            ),
         2: pd.Series(
                data  = np.array([6., 7., 8.], dtype=float), 
                index = pd.RangeIndex(start=6, stop=9, step=1),
                name  = "y_step_2"
            ),
         3: pd.Series(
                data  = np.array([7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=7, stop=10, step=1),
                name  = "y_step_3"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


def test_create_train_X_y_output_when_y_is_series_10_and_transformer_y_is_StandardScaler():
    """
    Test the output of create_train_X_y when exog is None and transformer_y
    is StandardScaler with steps=1.
    """
    forecaster = ForecasterAutoregDirect(
                     regressor     = LinearRegression(),
                     lags          = 5,
                     steps         = 1,
                     transformer_y = StandardScaler()
                 )
    results = forecaster.create_train_X_y(y=pd.Series(np.arange(10), name='y', dtype=float))
    expected = (
        pd.DataFrame(
            data = np.array([[-0.17407766, -0.52223297, -0.87038828, -1.21854359, -1.5666989 ],
                             [0.17407766, -0.17407766, -0.52223297, -0.87038828, -1.21854359],
                             [0.52223297,  0.17407766, -0.17407766, -0.52223297, -0.87038828],
                             [0.87038828,  0.52223297,  0.17407766, -0.17407766, -0.52223297],
                             [1.21854359,  0.87038828,  0.52223297,  0.17407766, -0.17407766]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ),
        {1: pd.Series(
                data  = np.array([0.17407766, 0.52223297, 0.87038828, 
                                  1.21854359, 1.5666989]), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "y_step_1"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


def test_create_train_X_y_output_when_lags_3_steps_1_and_exog_is_None_and_transformer_exog_is_not_None():
    """
    Test output of create_train_X_y when regressor is LinearRegression, lags is 3
    and steps is 1 and transformer_exog is not None.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = None

    forecaster = ForecasterAutoregDirect(
                     regressor        = LinearRegression(),
                     lags             = 3,
                     steps            = 1,
                     transformer_exog = StandardScaler()
                 )
    results = forecaster.create_train_X_y(y=y, exog=exog)
    
    expected = (
        pd.DataFrame(
            data = np.array([[2., 1., 0.],
                             [3., 2., 1.],
                             [4., 3., 2.],
                             [5., 4., 3.],
                             [6., 5., 4.],
                             [7., 6., 5.],
                             [8., 7., 6.]], dtype=float),
            index   = pd.RangeIndex(start=3, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3']
        ),
        {1: pd.Series(
                data  = np.array([3., 4., 5., 6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=3, stop=10, step=1),
                name  = "y_step_1"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 


def test_create_train_X_y_output_when_transformer_y_and_transformer_exog_steps_2():
    """
    Test the output of create_train_X_y when using transformer_y and transformer_exog 
    with steps=2.
    """
    y = pd.Series(np.arange(10), dtype = float)
    y.index = pd.date_range("1990-01-01", periods=10, freq='D')
    exog = pd.DataFrame({
               'col_1': [7.5, 24.4, 54.2, 12.1, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4],
               'col_2': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b']},
               index = pd.date_range("1990-01-01", periods=10, freq='D')
           )

    transformer_y = StandardScaler()
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['col_1']),
                            ('onehot', OneHotEncoder(), ['col_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )

    forecaster = ForecasterAutoregDirect(
                     regressor        = LinearRegression(),
                     lags             = 5,
                     steps            = 2,
                     transformer_y    = transformer_y,
                     transformer_exog = transformer_exog
                 )
    results = forecaster.create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[-0.17407766, -0.52223297, -0.87038828, -1.21854359, -1.5666989,
                              0.57176024, 0. , 1., 0.28259414, 1. , 0.],
                             [0.17407766, -0.17407766, -0.52223297, -0.87038828, -1.21854359,
                              0.28259414,  1. ,  0. , -0.12486718,  0. , 1. ],
                             [0.52223297,  0.17407766, -0.17407766, -0.52223297, -0.87038828,
                              -0.12486718,  0. ,  1. ,  1.88177028,  1. , 0. ],
                             [0.87038828,  0.52223297,  0.17407766, -0.17407766, -0.52223297,
                              1.88177028,  1. ,  0. ,  0.13801109,  0. , 1. ]]),
            index   = pd.date_range("1990-01-07", periods=4, freq='D'),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'col_1_step_1',
                       'col_2_a_step_1', 'col_2_b_step_1', 'col_1_step_2', 'col_2_a_step_2',
                       'col_2_b_step_2']
        ),
        {1: pd.Series(
                data  = np.array([0.17407766, 0.52223297, 0.87038828, 1.21854359]), 
                index = pd.date_range("1990-01-06", periods=4, freq='D'),
                name  = "y_step_1"
            ),
         2: pd.Series(
                data  = np.array([0.52223297, 0.87038828, 1.21854359, 1.5666989]), 
                index = pd.date_range("1990-01-07", periods=4, freq='D'),
                name  = "y_step_2"
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 