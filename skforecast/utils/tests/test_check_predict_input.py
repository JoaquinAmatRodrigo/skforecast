# Unit test check_input_predict
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
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
    with pytest.raises(Exception, match = err_msg):
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
            level           = None,
            series_levels   = None
        )
        
        
def test_check_input_predict_exception_when_steps_is_lower_than_1():
    """
    Test exception is raised when steps is a value lower than 1.
    """
    steps = -5

    err_msg = re.escape(f'`steps` must be integer greater than 0. Got {steps}.')
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
            level           = None,
            series_levels   = None
        )


def test_check_input_predict_exception_when_steps_is_greater_than_max_steps():
    """
    Test exception is raised when steps > max_steps.
    """
    steps = 20
    max_steps = 10

    err_msg = re.escape(
                (f'`steps` must be lower or equal to the value of steps defined '
                 f'when initializing the forecaster. Got {steps} but the maximum '
                 f'is {max_steps}.')
              )
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
            max_steps       = max_steps,
            level           = None,
            series_levels   = None
        )


def test_check_input_predict_exception_when_ForecasterAutoregMultiSeries_and_level_not_in_series_levels():
    """
    Test exception is raised when `level` is not in `self.series_levels` in a 
    ForecasterAutoregMultiSeries.
    """
    level = '1',
    series_levels = ['2', '3']

    err_msg = re.escape(f'`level` must be one of the `series_levels` : {series_levels}')
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
            max_steps       = 10,
            level           = level,
            series_levels   = series_levels
        )


def test_check_input_predict_exception_when_exog_is_none_and_included_exog_is_true():
    """
    """
    err_msg = re.escape(
                ('Forecaster trained with exogenous variable/s. '
                 'Same variable/s must be provided in `predict()`.')
              )   
    with pytest.raises(Exception, match = err_msg):
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
            level           = None,
            series_levels   = None
        )


def test_check_input_predict_exception_when_exog_is_not_none_and_included_exog_is_false():
    """
    """
    err_msg = re.escape(
                ('Forecaster trained without exogenous variable/s. '
                 '`exog` must be `None` in `predict()`.')
              )   
    with pytest.raises(Exception, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoreg',
            steps           = 5,
            fitted          = True,
            included_exog   = False,
            index_type      = None,
            index_freq      = None,
            window_size     = None,
            last_window     = None,
            exog            = np.arange(10),
            exog_type       = None,
            exog_col_names  = None,
            interval        = None,
            max_steps       = None,
        )


def test_check_input_predict_exception_when_len_exog_is_less_than_steps():
    """
    """
    err_msg = re.escape('`exog` must have at least as many values as `steps` predicted.')
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoreg',
            steps           = 10,
            fitted          = True,
            included_exog   = True,
            index_type      = None,
            index_freq      = None,
            window_size     = None,
            last_window     = None,
            exog            = np.arange(5),
            exog_type       = None,
            exog_col_names  = None,
            interval        = None,
            max_steps       = None,
            level           = None,
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
            level           = None,
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
            level           = None,
            series_levels   = None
        )


def test_check_input_predict_exception_when_exog_is_not_of_exog_type():
    """
    """
    exog = pd.Series(np.arange(10))
    exog_type = pd.DataFrame

    err_msg = re.escape(f'Expected type for `exog`: {exog_type}. Got {type(exog)}')
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
            level           = None,
            series_levels   = None
        )


def test_check_input_predict_exception_when_exog_is_dataframe_without_columns_in_exog_col_names():
    """
    """
    exog = pd.DataFrame(np.arange(10).reshape(5,2), columns=['col1', 'col2'])
    exog_col_names = ['col1', 'col3']

    err_msg = re.escape(
                (f'Missing columns in `exog`. Expected {exog_col_names}. '
                 f'Got {exog.columns.to_list()}')
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
            level           = None,
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
                 f'Got {type(exog_index)}')
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
            level           = None,
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
                 f'Got {exog_index.freqstr}')
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
            level           = None,
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
            level           = None,
            series_levels   = None
        )


def test_check_input_predict_exception_when_last_window_is_not_pandas_dataframe_ForecasterAutoregMultiSeries():
    """
    """
    err_msg = re.escape('`last_window` must be a pandas DataFrame.')
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_type = 'class.ForecasterAutoregMultiSeries',
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
            level           = '1',
            series_levels   = ['1', '2']
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
            level           = None,
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
            level           = None,
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
                 f'Got {type(last_window_index)}')
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
            level           = None,
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
                 f'Got {last_window_index.freqstr}')
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
            level           = None,
            series_levels   = None
        )