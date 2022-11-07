# Unit test check_input_predict
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.utils import check_predict_input
from skforecast.utils import check_exog
from skforecast.utils import preprocess_exog
from skforecast.utils import preprocess_last_window


def test_check_input_predict_exception_when_fitted_is_False():
    """
    Test exception is raised when fitted is False.
    """
    err_msg = re.escape(
                ('This Forecaster instance is not fitted yet. Call `fit` with '
                 'appropriate arguments before using predict.')
              )
    with pytest.raises(NotFittedError, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoreg',
            steps           = 5,
            fitted          = False,
            included_exog   = False,
            index_type      = None,
            index_freq      = None,
            window_size     = None,
            last_window     = None,
            exog            = None,
            exog_type       = None,
            exog_col_names  = None,
            interval        = None,
            max_steps       = None,
            levels          = None,
            series_levels   = None
        )
        
        
def test_check_input_predict_exception_when_steps_int_lower_than_1():
    """
    Test exception is raised when steps is a value lower than 1.
    """
    steps = -5

    err_msg = re.escape(f'`steps` must be an integer greater than or equal to 1. Got {steps}.')
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoreg',
            steps           = steps,
            fitted          = True,
            included_exog   = False,
            index_type      = None,
            index_freq      = None,
            window_size     = None,
            last_window     = None,
            exog            = None,
            exog_type       = None,
            exog_col_names  = None,
            interval        = None,
            max_steps       = None,
            levels          = None,
            series_levels   = None
        )
        
        
def test_check_input_predict_exception_when_steps_list_lower_than_0():
    """
    Test exception is raised when steps is a list with a value lower than 0. 
    (`ForecasterAutoregDirect` and `ForecasterAutoregMultiVariate`).
    """
    steps = [-1, 0, 1]

    err_msg = re.escape(
                  (f"The minimum value of `steps` must be equal to or greater than 1. "
                   f"Got {min(steps) + 1}.")
              )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoregDirect',
            steps           = steps,
            fitted          = True,
            included_exog   = False,
            index_type      = None,
            index_freq      = None,
            window_size     = None,
            last_window     = None,
            exog            = None,
            exog_type       = None,
            exog_col_names  = None,
            interval        = None,
            max_steps       = None,
            levels          = None,
            series_levels   = None
        )


def test_check_input_predict_exception_when_max_steps_greater_than_max_steps():
    """
    Test exception is raised when max(steps) > max_steps. (`ForecasterAutoregDirect` 
    and `ForecasterAutoregMultiVariate`).
    """
    steps = list(range(20))
    max_steps = 10

    err_msg = re.escape(
                (f"The maximum value of `steps` must be less than or equal to "
                 f"the value of steps defined when initializing the forecaster. "
                 f"Got {max(steps)+1}, but the maximum is {max_steps}.")
              )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoregMultiVariate',
            steps           = steps,
            fitted          = True,
            included_exog   = False,
            index_type      = None,
            index_freq      = None,
            window_size     = None,
            last_window     = None,
            exog            = None,
            exog_type       = None,
            exog_col_names  = None,
            interval        = None,
            max_steps       = max_steps,
            levels          = None,
            series_levels   = None
        )


def test_check_input_predict_exception_when_ForecasterAutoregMultiSeries_and_level_not_str_list_or_None():
    """
    Test exception is raised when `levels` is not a str, a list or None.
    """
    levels = 5

    err_msg = re.escape(f'`levels` must be a `list` of column names, a `str` of a column name or `None`.')   
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoregMultiSeries',
            steps           = 5,
            fitted          = True,
            included_exog   = False,
            index_type      = None,
            index_freq      = None,
            window_size     = None,
            last_window     = None,
            exog            = None,
            exog_type       = None,
            exog_col_names  = None,
            interval        = None,
            max_steps       = None,
            levels          = levels,
            series_levels   = ['1', '2']
        )


@pytest.mark.parametrize("levels     , series_levels", 
                         [('1'       , ['2', '3']), 
                          (['1']     , ['2', '3']), 
                          (['1', '2'], ['2', '3'])])
def test_check_input_predict_exception_when_ForecasterAutoregMultiSeries_and_level_not_in_series_levels(levels, series_levels):
    """
    Test exception is raised when `levels` is not in `self.series_levels` in a 
    ForecasterAutoregMultiSeries.
    """
    err_msg = re.escape(f'`levels` must be in `series_levels` : {series_levels}.')
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoregMultiSeries',
            steps           = 5,
            fitted          = True,
            included_exog   = False,
            index_type      = None,
            index_freq      = None,
            window_size     = None,
            last_window     = None,
            exog            = None,
            exog_type       = None,
            exog_col_names  = None,
            interval        = None,
            max_steps       = None,
            levels          = levels,
            series_levels   = series_levels
        )


def test_check_input_predict_exception_when_exog_is_none_and_included_exog_is_true():
    """
    """
    err_msg = re.escape(
                ('Forecaster trained with exogenous variable/s. '
                 'Same variable/s must be provided in `predict()`.')
              )   
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoreg',
            steps           = 5,
            fitted          = True,
            included_exog   = True,
            index_type      = None,
            index_freq      = None,
            window_size     = None,
            last_window     = None,
            exog            = None,
            exog_type       = None,
            exog_col_names  = None,
            interval        = None,
            max_steps       = None,
            levels          = None,
            series_levels   = None
        )


def test_check_input_predict_exception_when_exog_is_not_none_and_included_exog_is_false():
    """
    """
    err_msg = re.escape(
                ('Forecaster trained without exogenous variable/s. '
                 '`exog` must be `None` in `predict()`.')
              )   
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoreg',
            steps           = 5,
            fitted          = True,
            included_exog   = False,
            index_type      = None,
            index_freq      = None,
            window_size     = None,
            last_window     = None,
            exog            = pd.Series(np.arange(10)),
            exog_type       = None,
            exog_col_names  = None,
            interval        = None,
            max_steps       = None,
            levels          = None,
            series_levels   = None
        )


@pytest.mark.parametrize("steps", [10, [1, 2, 3, 4, 5, 6], [2, 6]], 
                         ids=lambda steps: f'steps: {steps}')
def test_check_input_predict_exception_when_len_exog_is_less_than_steps(steps):
    """
    """
    max_step = max(steps)+1 if isinstance(steps, list) else steps
    err_msg = re.escape(
                f'`exog` must have at least as many values as the distance to '
                f'the maximum step predicted, {max_step}.'
            )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoreg',
            steps           = steps,
            fitted          = True,
            included_exog   = True,
            index_type      = None,
            index_freq      = None,
            window_size     = None,
            last_window     = None,
            exog            = pd.Series(np.arange(5)),
            exog_type       = None,
            exog_col_names  = None,
            interval        = None,
            max_steps       = None,
            levels          = None,
            series_levels   = None
        )


def test_check_input_predict_exception_when_exog_is_not_pandas_series_or_dataframe():
    """
    """
    err_msg = re.escape('`exog` must be a pandas Series or DataFrame.')
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoreg',
            steps           = 10,
            fitted          = True,
            included_exog   = True,
            index_type      = None,
            index_freq      = None,
            window_size     = 5,
            last_window     = None,
            exog            = np.arange(10),
            exog_type       = None,
            exog_col_names  = None,
            interval        = None,
            max_steps       = None,
            levels          = None,
            series_levels   = None
        )


def test_check_input_predict_exception_when_exog_has_missing_values():
    """
    """
    err_msg = re.escape('`exog` has missing values.')
    with pytest.raises(Exception, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoreg',
            steps           = 3,
            fitted          = True,
            included_exog   = True,
            index_type      = None,
            index_freq      = None,
            window_size     = 2,
            last_window     = None,
            exog            = pd.Series([1, 2, 3, np.nan]),
            exog_type       = None,
            exog_col_names  = None,
            interval        = None,
            max_steps       = None,
            levels          = None,
            series_levels   = None
        )


def test_check_input_predict_exception_when_exog_is_not_of_exog_type():
    """
    """
    exog = pd.Series(np.arange(10))
    exog_type = pd.DataFrame

    err_msg = re.escape(f'Expected type for `exog`: {exog_type}. Got {type(exog)}.')
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoreg',
            steps           = 5,
            fitted          = True,
            included_exog   = True,
            index_type      = None,
            index_freq      = None,
            window_size     = None,
            last_window     = None,
            exog            = exog,
            exog_type       = exog_type,
            exog_col_names  = None,
            interval        = None,
            max_steps       = None,
            levels          = None,
            series_levels   = None
        )


def test_check_input_predict_exception_when_exog_is_dataframe_without_columns_in_exog_col_names():
    """
    """
    exog = pd.DataFrame(np.arange(10).reshape(5, 2), columns=['col1', 'col2'])
    exog_col_names = ['col1', 'col3']

    err_msg = re.escape(
                (f'Missing columns in `exog`. Expected {exog_col_names}. '
                 f'Got {exog.columns.to_list()}.')
              )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoreg',
            steps           = 2,
            fitted          = True,
            included_exog   = True,
            index_type      = None,
            index_freq      = None,
            window_size     = None,
            last_window     = None,
            exog            = exog,
            exog_type       = pd.DataFrame,
            exog_col_names  = exog_col_names,
            interval        = None,
            max_steps       = None,
            levels          = None,
            series_levels   = None
        )


def test_check_input_predict_exception_when_exog_index_is_not_of_index_type():
    """
    """
    exog = pd.Series(np.arange(10))
    index_type = pd.DatetimeIndex
    check_exog(exog = exog)
    _, exog_index = preprocess_exog(exog=exog.iloc[:0, ])

    err_msg = re.escape(
                (f'Expected index of type {index_type} for `exog`. '
                 f'Got {type(exog_index)}.')
              )
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoreg',
            steps           = 10,
            fitted          = True,
            included_exog   = True,
            index_type      = index_type,
            index_freq      = None,
            window_size     = None,
            last_window     = None,
            exog            = exog,
            exog_type       = pd.Series,
            exog_col_names  = None,
            interval        = None,
            max_steps       = None,
            levels          = None,
            series_levels   = None
        )


def test_check_input_predict_exception_when_exog_index_frequency_is_not_index_freq():
    """
    """
    exog = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10))
    index_freq = 'Y'
    check_exog(exog = exog)
    _, exog_index = preprocess_exog(exog=exog.iloc[:0, ])

    err_msg = re.escape(
                (f'Expected frequency of type {index_freq} for `exog`. '
                 f'Got {exog_index.freqstr}.')
              )
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoreg',
            steps           = 10,
            fitted          = True,
            included_exog   = True,
            index_type      = pd.DatetimeIndex,
            index_freq      = index_freq,
            window_size     = None,
            last_window     = None,
            exog            = exog,
            exog_type       = pd.Series,
            exog_col_names  = None,
            interval        = None,
            max_steps       = None,
            levels          = None,
            series_levels   = None
        )


def test_check_input_predict_exception_when_length_last_window_is_lower_than_window_size():
    """
    """
    window_size = 10

    err_msg = re.escape(
                (f'`last_window` must have as many values as as needed to '
                 f'calculate the predictors. For this forecaster it is {window_size}.')
              )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoreg',
            steps           = 10,
            fitted          = True,
            included_exog   = True,
            index_type      = pd.RangeIndex,
            index_freq      = None,
            window_size     = window_size,
            last_window     = pd.Series(np.arange(5)),
            exog            = pd.Series(np.arange(10)),
            exog_type       = pd.Series,
            exog_col_names  = None,
            interval        = None,
            max_steps       = None,
            levels          = None,
            series_levels   = None
        )


@pytest.mark.parametrize("forecaster_type", 
                         ['class.ForecasterAutoregMultiSeries',
                          'class.ForecasterAutoregMultiVariate'], 
                         ids=lambda ft: f'forecaster_type: {ft}')
def test_check_input_predict_exception_when_last_window_is_not_pandas_dataframe(forecaster_type):
    """
    `ForecasterAutoregMultiSeries` and `ForecasterAutoregMultiVariate`.
    """
    last_window = np.arange(5)
    err_msg = re.escape(f'`last_window` must be a pandas DataFrame. Got {type(last_window)}.')
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_type = forecaster_type,
            steps           = 10,
            fitted          = True,
            included_exog   = True,
            index_type      = pd.RangeIndex,
            index_freq      = None,
            window_size     = 5,
            last_window     = last_window,
            exog            = pd.Series(np.arange(10)),
            exog_type       = pd.Series,
            exog_col_names  = None,
            interval        = None,
            max_steps       = None,
            levels          = '1',
            series_levels   = ['1', '2']
        )


@pytest.mark.parametrize("levels     , last_window", 
                         [('1'       , pd.DataFrame({'3': [1, 2, 3], '4': [1, 2, 3]})), 
                          (['1']     , pd.DataFrame({'3': [1, 2, 3], '4': [1, 2, 3]})), 
                          (['1', '2'], pd.DataFrame({'3': [1, 2, 3], '4': [1, 2, 3]}))], 
                         ids = lambda values : f'levels: {values}'
                        )
def test_check_input_predict_exception_when_levels_not_in_last_window_ForecasterAutoregMultiSeries(levels, last_window):
    """
    """
    err_msg = re.escape(
                    (f'`last_window` must contain a column(s) named as the level(s) to be predicted.\n'
                     f'    `levels` : {levels}.\n'
                     f'    `last_window` columns : {list(last_window.columns)}.')
                )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoregMultiSeries',
            steps           = 10,
            fitted          = True,
            included_exog   = True,
            index_type      = pd.RangeIndex,
            index_freq      = None,
            window_size     = 2,
            last_window     = last_window,
            exog            = pd.Series(np.arange(10)),
            exog_type       = pd.Series,
            exog_col_names  = None,
            interval        = None,
            max_steps       = None,
            levels          = levels,
            series_levels   = ['1', '2']
        )


def test_check_input_predict_exception_when_series_levels_not_last_window_ForecasterAutoregMultiVariate():
    """
    Check exception is raised when column names of series using during fit do not
    match with last_window column names.
    """
    last_window = pd.DataFrame({'l1': [1, 2, 3], '4': [1, 2, 3]})
    series_levels = ['l1', 'l2']
    err_msg = re.escape(
                    (f'`last_window` columns must be the same as `series` column names.\n'
                     f'    `last_window` columns : {list(last_window.columns)}.\n'
                     f'    `series` columns      : {series_levels}.')
                )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoregMultiVariate',
            steps           = 10,
            fitted          = True,
            included_exog   = True,
            index_type      = pd.RangeIndex,
            index_freq      = None,
            window_size     = 2,
            last_window     = last_window,
            exog            = pd.Series(np.arange(10)),
            exog_type       = pd.Series,
            exog_col_names  = None,
            interval        = None,
            max_steps       = None,
            levels          = None,
            series_levels   = series_levels
        )


def test_check_input_predict_exception_when_last_window_is_not_pandas_series():
    """
    """
    err_msg = re.escape('`last_window` must be a pandas Series.')
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoreg',
            steps           = 10,
            fitted          = True,
            included_exog   = True,
            index_type      = pd.RangeIndex,
            index_freq      = None,
            window_size     = 5,
            last_window     = np.arange(5),
            exog            = pd.Series(np.arange(10)),
            exog_type       = pd.Series,
            exog_col_names  = None,
            interval        = None,
            max_steps       = None,
            levels          = None,
            series_levels   = None
        )


def test_check_input_predict_exception_when_last_window_has_missing_values():
    """
    """
    err_msg = re.escape('`last_window` has missing values.')
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoreg',
            steps           = 10,
            fitted          = True,
            included_exog   = True,
            index_type      = pd.RangeIndex,
            index_freq      = None,
            window_size     = 5,
            last_window     = pd.Series([1, 2, 3, 4, 5, np.nan]),
            exog            = pd.Series(np.arange(10)),
            exog_type       = pd.Series,
            exog_col_names  = None,
            interval        = None,
            max_steps       = None,
            levels          = None,
            series_levels   = None
        )


def test_check_input_predict_exception_when_last_window_index_is_not_of_index_type():
    """
    """
    last_window = pd.Series(np.arange(10))
    index_type = pd.DatetimeIndex
    _, last_window_index = preprocess_last_window(
                                last_window = last_window.iloc[:0]
                           ) 

    err_msg = re.escape(
                (f'Expected index of type {index_type} for `last_window`. '
                 f'Got {type(last_window_index)}.')
              )
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoreg',
            steps           = 10,
            fitted          = True,
            included_exog   = False,
            index_type      = index_type,
            index_freq      = None,
            window_size     = 5,
            last_window     = last_window,
            exog            = None,
            exog_type       = None,
            exog_col_names  = None,
            interval        = None,
            max_steps       = None,
            levels          = None,
            series_levels   = None
        )


def test_check_input_predict_exception_when_last_window_index_frequency_is_not_index_freq():
    """
    """
    last_window = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10))
    index_freq = 'Y'
    _, last_window_index = preprocess_last_window(
                                last_window = last_window.iloc[:0]
                           )

    err_msg = re.escape(
                (f'Expected frequency of type {index_freq} for `last_window`. '
                 f'Got {last_window_index.freqstr}.')
              )
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoreg',
            steps           = 10,
            fitted          = True,
            included_exog   = False,
            index_type      = pd.DatetimeIndex,
            index_freq      = index_freq,
            window_size     = 5,
            last_window     = last_window,
            exog            = None,
            exog_type       = None,
            exog_col_names  = None,
            interval        = None,
            max_steps       = None,
            levels          = None,
            series_levels   = None
        )