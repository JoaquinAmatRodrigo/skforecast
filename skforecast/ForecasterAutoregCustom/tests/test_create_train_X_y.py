# Unit test create_train_X_y ForecasterAutoregCustom
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from sklearn.linear_model import LinearRegression


def create_predictors(y): # pragma: no cover
    """
    Create first 5 lags of a time series.
    """
    lags = y[-1:-6:-1]
    return lags


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
                 f'{forecaster.create_predictors.__name__}. For this Forecaster the '
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

    train_X = forecaster.create_train_X_y(
                y = pd.Series(np.arange(10), index=pd.date_range(start='2022-01-01', periods=10, freq='1D'))
            )[0]
    assert train_X.columns.to_list() == ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
    

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
    expected = (pd.DataFrame(
                    data = np.array([[4., 3., 2., 1., 0.],
                                     [5., 4., 3., 2., 1.],
                                     [6., 5., 4., 3., 2.],
                                     [7., 6., 5., 4., 3.],
                                     [8., 7., 6., 5., 4.]]),
                    index   = np.array([5, 6, 7, 8, 9]),
                    columns = ['custom_predictor_0', 'custom_predictor_1',
                               'custom_predictor_2', 'custom_predictor_3',
                               'custom_predictor_4']
                ),
                pd.Series(
                    np.array([5., 6., 7., 8., 9.]),
                    index = np.array([5, 6, 7, 8, 9]),
                    name = 'y'
                )
               )     

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            assert (results[i] == expected[i]).all()


def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_series():
    """
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas series
    """
    forecaster = ForecasterAutoregCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5
                )
    results = forecaster.create_train_X_y(
                y = pd.Series(np.arange(10, dtype=float)),
                exog =  pd.Series(np.arange(100, 110, dtype=float), name='exog')
              )
    expected = (pd.DataFrame(
                    data = np.array([[4., 3., 2., 1., 0., 105.],
                                     [5., 4., 3., 2., 1., 106.],
                                     [6., 5., 4., 3., 2., 107.],
                                     [7., 6., 5., 4., 3., 108.],
                                     [8., 7., 6., 5., 4., 109.]]),
                    index   = np.array([5, 6, 7, 8, 9]),
                    columns = ['custom_predictor_0', 'custom_predictor_1',
                               'custom_predictor_2', 'custom_predictor_3',
                               'custom_predictor_4', 'exog']
                ),
                pd.Series(
                    np.array([5., 6., 7., 8., 9.]),
                    index = np.array([5, 6, 7, 8, 9]),
                    name = 'y'
                )
               )       

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            assert (results[i] == expected[i]).all()


def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_dataframe():
    """
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas dataframe with two columns.
    """
    forecaster = ForecasterAutoregCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5
                )
    results = forecaster.create_train_X_y(
                y = pd.Series(np.arange(10, dtype=float)),
                exog = pd.DataFrame({
                            'exog_1' : np.arange(100, 110, dtype=float),
                            'exog_2' : np.arange(1000, 1010, dtype=float)
                })
              )
    expected = (pd.DataFrame(
                    data = np.array([[4., 3., 2., 1., 0., 105., 1005.],
                                     [5., 4., 3., 2., 1., 106., 1006.],
                                     [6., 5., 4., 3., 2., 107., 1007.],
                                     [7., 6., 5., 4., 3., 108., 1008.],
                                     [8., 7., 6., 5., 4., 109., 1009.]]),
                    index   = np.array([5, 6, 7, 8, 9]),
                    columns = ['custom_predictor_0', 'custom_predictor_1',
                               'custom_predictor_2', 'custom_predictor_3',
                               'custom_predictor_4', 'exog_1', 'exog_2']
                ),
                pd.Series(
                    np.array([5., 6., 7., 8., 9.]),
                    index = np.array([5, 6, 7, 8, 9]),
                    name = 'y'
                )
               )        

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            assert (results[i] == expected[i]).all()

    
def test_create_train_X_y_exception_when_y_and_exog_have_different_length():
    """
    Test exception is raised when length of y and length of exog are different.
    """
    forecaster = ForecasterAutoregCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5
                )

    y = pd.Series(np.arange(50))
    exog = pd.Series(np.arange(10))
    err_msg = re.escape((
                f'`exog` must have same number of samples as `y`. '
                f'length `exog`: ({len(exog)}), length `y`: ({len(y)})'
              ))
    with pytest.raises(ValueError, match = err_msg):
        forecaster.fit(y=y, exog=exog)

    y = pd.Series(np.arange(10))
    exog = pd.Series(np.arange(50))
    err_msg = re.escape((
                f'`exog` must have same number of samples as `y`. '
                f'length `exog`: ({len(exog)}), length `y`: ({len(y)})'
              ))
    with pytest.raises(ValueError, match = err_msg):
        forecaster.fit(y=y, exog=exog)

    y = pd.Series(np.arange(10))
    exog = pd.DataFrame(np.arange(50).reshape(25,2))
    err_msg = re.escape((
                f'`exog` must have same number of samples as `y`. '
                f'length `exog`: ({len(exog)}), length `y`: ({len(y)})'
              ))
    with pytest.raises(ValueError, match = err_msg):
        forecaster.fit(y=y, exog=exog)


def test_create_train_X_y_ValueError_fun_predictors_return_nan_values():
    """
    Test ValueError is raised when fun_predictors returns `NaN`values.
    """
    forecaster = ForecasterAutoregCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = lambda y: np.array([np.nan, np.nan]),
                    window_size    = 5
                )
    err_msg = re.escape("`fun_predictors()` is returning `NaN` values.")
    with pytest.raises(ValueError, match = err_msg):
        forecaster.fit(y=pd.Series(np.arange(50)))