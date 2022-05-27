# Unit test check_input_predict
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import check_predict_input


def test_check_input_predict_exception_when_fitted_is_False():
    '''
    Test exception is raised when fitted is False.
    '''
    with pytest.raises(Exception):
        check_predict_input(
            forecaster_type = 'ForecasterAutoreg',
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
            max_steps       = None,
        )
        
        
def test_check_input_predict_exception_when_steps_is_lower_than_1():
    '''
    Test exception is steps is a value lower than 1.
    '''
    with pytest.raises(Exception):
        check_predict_input(
            forecaster_type = 'ForecasterAutoreg',
            steps           = -5,
            fitted          = True,
            included_exog   = False,
            index_type      = None,
            index_freq      = None,
            window_size     = None,
            last_window     = None,
            exog            = None,
            exog_type       = None,
            exog_col_names  = None,
            max_steps       = None,
        )


def test_check_input_predict_exception_when_steps_is_greater_than_max_steps():
    '''
    '''
    with pytest.raises(Exception):
        check_predict_input(
            forecaster_type = 'ForecasterAutoreg',
            steps           = 20,
            fitted          = True,
            included_exog   = False,
            index_type      = None,
            index_freq      = None,
            window_size     = None,
            last_window     = None,
            exog            = None,
            exog_type       = None,
            exog_col_names  = None,
            max_steps       = 10,
        )


def test_check_input_predict_exception_when_exog_is_not_none_and_included_exog_is_false():
    '''
    '''
    with pytest.raises(Exception):
        check_predict_input(
            forecaster_type = 'ForecasterAutoreg',
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
            max_steps       = None,
        )


def test_check_input_predict_exception_when_exog_is_none_and_included_exog_is_true():
    '''
    '''
    with pytest.raises(Exception):
        check_predict_input(
            forecaster_type = 'ForecasterAutoreg',
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
            max_steps       = None,
        )


def test_check_input_predict_exception_when_len_exog_is_less_than_steps():
    '''
    '''
    with pytest.raises(Exception):
        check_predict_input(
            forecaster_type = 'ForecasterAutoreg',
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
            max_steps       = None,
        )


def test_check_input_predict_exception_when_exog_is_not_pandas_series_or_dataframe():
    '''
    '''
    with pytest.raises(Exception):
        check_predict_input(
            forecaster_type = 'ForecasterAutoreg',
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
            max_steps       = None,
        )


def test_check_input_predict_exception_when_exog_has_missing_values():
    '''
    '''
    with pytest.raises(Exception):
        check_predict_input(
            forecaster_type = 'ForecasterAutoreg',
            steps           = 10,
            fitted          = True,
            included_exog   = True,
            index_type      = None,
            index_freq      = None,
            window_size     = 5,
            last_window     = None,
            exog            = pd.Series([1, 2, 3, np.nan]),
            exog_type       = None,
            exog_col_names  = None,
            max_steps       = None,
        )


def test_check_input_predict_exception_when_exog_is_not_of_exog_type():
    '''
    '''
    with pytest.raises(Exception):
        check_predict_input(
            forecaster_type = 'ForecasterAutoreg',
            steps           = 10,
            fitted          = True,
            included_exog   = True,
            index_type      = None,
            index_freq      = None,
            window_size     = None,
            last_window     = None,
            exog            = np.arange(10),
            exog_type       = pd.Series,
            exog_col_names  = None,
            max_steps       = None,
        )


def test_check_input_predict_exception_when_exog_is_dataframe_without_columns_in_exog_col_names():
    '''
    '''
    with pytest.raises(Exception):
        check_predict_input(
            forecaster_type = 'ForecasterAutoreg',
            steps           = 2,
            fitted          = True,
            included_exog   = True,
            index_type      = None,
            index_freq      = None,
            window_size     = None,
            last_window     = None,
            exog            = pd.DataFrame(np.arange(10).reshape(5,2), columns=['col1', 'col2']),
            exog_type       = pd.DataFrame,
            exog_col_names  = ['col1', 'col3'],
            max_steps       = None,
        )


def test_check_input_predict_exception_when_exog_index_is_not_of_index_type():
    '''
    '''
    with pytest.raises(Exception):
        check_predict_input(
            forecaster_type = 'ForecasterAutoreg',
            steps           = 10,
            fitted          = True,
            included_exog   = True,
            index_type      = pd.DatetimeIndex,
            index_freq      = None,
            window_size     = None,
            last_window     = None,
            exog            = pd.Series(np.arange(10)),
            exog_type       = pd.Series,
            exog_col_names  = None,
            max_steps       = None,
        )


def test_check_input_predict_exception_when_exog_index_frequency_is_not_index_freq():
    '''
    '''
    with pytest.raises(Exception):
        check_predict_input(
            forecaster_type = 'ForecasterAutoreg',
            steps           = 10,
            fitted          = True,
            included_exog   = True,
            index_type      = pd.DatetimeIndex,
            index_freq      = 'Y',
            window_size     = None,
            last_window     = None,
            exog            = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10)),
            exog_type       = pd.Series,
            exog_col_names  = None,
            max_steps       = None,
        )


def test_check_input_predict_exception_when_length_last_window_is_lower_than_window_size():
    '''
    '''
    with pytest.raises(Exception):
        check_predict_input(
            forecaster_type = 'ForecasterAutoreg',
            steps           = 10,
            fitted          = True,
            included_exog   = True,
            index_type      = pd.RangeIndex,
            index_freq      = None,
            window_size     = 10,
            last_window     = pd.Series(np.arange(5)),
            exog            = pd.Series(np.arange(10)),
            exog_type       = pd.Series,
            exog_col_names  = None,
            max_steps       = None,
        )


def test_check_input_predict_exception_when_last_window_is_not_pandas_series():
    '''
    '''
    with pytest.raises(Exception):
        check_predict_input(
            forecaster_type = 'ForecasterAutoreg',
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
            max_steps       = None,
        )


def test_check_input_predict_exception_when_last_window_is_not_pandas_dataframe_ForecasterAutoregMultiSeries():
    '''
    '''
    with pytest.raises(Exception):
        check_predict_input(
            forecaster_type = 'ForecasterAutoregMultiSeries',
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
            max_steps       = None,
        )


def test_check_input_predict_exception_when_last_window_has_missing_values():
    '''
    '''
    with pytest.raises(Exception):
        check_predict_input(
            forecaster_type = 'ForecasterAutoreg',
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
            max_steps       = None,
        )


def test_check_input_predict_exception_when_last_window_index_is_not_of_index_type():
    '''
    '''
    with pytest.raises(Exception):
        check_predict_input(
            forecaster_type = 'ForecasterAutoreg',
            steps           = 10,
            fitted          = True,
            included_exog   = True,
            index_type      = pd.DatetimeIndex,
            index_freq      = None,
            window_size     = None,
            last_window     = pd.Series(np.arange(10)),
            exog            = None,
            exog_type       = None,
            exog_col_names  = None,
            max_steps       = None,
        )


def test_check_input_predict_exception_when_last_window_index_frequency_is_not_index_freq():
    '''
    '''
    with pytest.raises(Exception):
        check_predict_input(
            forecaster_type = 'ForecasterAutoreg',
            steps           = 10,
            fitted          = True,
            included_exog   = True,
            index_type      = pd.DatetimeIndex,
            index_freq      = 'Y',
            window_size     = None,
            last_window     = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10)),
            exog            = None,
            exog_type       = None,
            exog_col_names  = None,
            max_steps       = None,
        )