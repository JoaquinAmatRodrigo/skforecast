# Unit test check_predict_input
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
from skforecast.exceptions import MissingValuesExogWarning


def test_check_predict_input_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    err_msg = re.escape(
                ("This Forecaster instance is not fitted yet. Call `fit` with "
                 "appropriate arguments before using predict.")
              )
    with pytest.raises(NotFittedError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterAutoreg',
            steps            = 5,
            fitted           = False,
            included_exog    = False,
            index_type       = None,
            index_freq       = None,
            window_size      = None,
            last_window      = None,
            last_window_exog = None,
            exog             = None,
            exog_type        = None,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )
        
        
def test_check_predict_input_ValueError_when_steps_int_lower_than_1():
    """
    Test ValueError is raised when steps is a value lower than 1.
    """
    steps = -5

    err_msg = re.escape(f'`steps` must be an integer greater than or equal to 1. Got {steps}.')
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterAutoreg',
            steps            = steps,
            fitted           = True,
            included_exog    = False,
            index_type       = None,
            index_freq       = None,
            window_size      = None,
            last_window      = None,
            last_window_exog = None,
            exog             = None,
            exog_type        = None,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )
        
        
def test_check_predict_input_ValueError_when_steps_list_lower_than_1():
    """
    Test ValueError is raised when steps is a list with a value lower than 1. 
    (`ForecasterAutoregDirect` and `ForecasterAutoregMultiVariate`).
    """
    steps = [0, 1, 2]

    err_msg = re.escape(
                  (f"The minimum value of `steps` must be equal to or greater than 1. "
                   f"Got {min(steps)}.")
              )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterAutoregDirect',
            steps            = steps,
            fitted           = True,
            included_exog    = False,
            index_type       = None,
            index_freq       = None,
            window_size      = None,
            last_window      = None,
            last_window_exog = None,
            exog             = None,
            exog_type        = None,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )


def test_check_predict_input_ValueError_when_last_step_greater_than_max_steps():
    """
    Test ValueError is raised when max(steps) > max_steps. (`ForecasterAutoregDirect` 
    and `ForecasterAutoregMultiVariate`).
    """
    steps = list(np.arange(20) + 1)
    max_steps = 10

    err_msg = re.escape(
                (f"The maximum value of `steps` must be less than or equal to "
                 f"the value of steps defined when initializing the forecaster. "
                 f"Got {max(steps)}, but the maximum is {max_steps}.")
              )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterAutoregMultiVariate',
            steps            = steps,
            fitted           = True,
            included_exog    = False,
            index_type       = None,
            index_freq       = None,
            window_size      = None,
            last_window      = None,
            last_window_exog = None,
            exog             = None,
            exog_type        = None,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = max_steps,
            levels           = None,
            series_col_names = None
        )


def test_check_predict_input_TypeError_when_ForecasterAutoregMultiSeries_and_level_not_str_list_or_None():
    """
    Test TypeError is raised when `levels` is not a str, a list or None.
    """
    levels = 5

    err_msg = re.escape(f'`levels` must be a `list` of column names, a `str` of a column name or `None`.')   
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterAutoregMultiSeries',
            steps            = 5,
            fitted           = True,
            included_exog    = False,
            index_type       = None,
            index_freq       = None,
            window_size      = None,
            last_window      = None,
            last_window_exog = None,
            exog             = None,
            exog_type        = None,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = levels,
            series_col_names = ['1', '2']
        )


@pytest.mark.parametrize("levels     , series_col_names", 
                         [('1'       , ['2', '3']), 
                          (['1']     , ['2', '3']), 
                          (['1', '2'], ['2', '3'])])
def test_check_predict_input_ValueError_when_ForecasterAutoregMultiSeries_and_level_not_in_series_col_names(levels, series_col_names):
    """
    Test ValueError is raised when `levels` is not in `self.series_col_names` in a 
    ForecasterAutoregMultiSeries.
    """
    err_msg = re.escape(f'`levels` must be in `series_col_names` : {series_col_names}.')
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterAutoregMultiSeries',
            steps            = 5,
            fitted           = True,
            included_exog    = False,
            index_type       = None,
            index_freq       = None,
            window_size      = None,
            last_window      = None,
            last_window_exog = None,
            exog             = None,
            exog_type        = None,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = levels,
            series_col_names = series_col_names
        )


def test_check_predict_input_ValueError_when_exog_is_none_and_included_exog_is_true():
    """
    """
    err_msg = re.escape(
                ('Forecaster trained with exogenous variable/s. '
                 'Same variable/s must be provided when predicting.')
              )   
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterAutoreg',
            steps            = 5,
            fitted           = True,
            included_exog    = True,
            index_type       = None,
            index_freq       = None,
            window_size      = None,
            last_window      = None,
            last_window_exog = None,
            exog             = None,
            exog_type        = None,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )


def test_check_predict_input_ValueError_when_exog_is_not_none_and_included_exog_is_false():
    """
    """
    err_msg = re.escape(
                ('Forecaster trained without exogenous variable/s. '
                 '`exog` must be `None` when predicting.')
              )   
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterAutoreg',
            steps            = 5,
            fitted           = True,
            included_exog    = False,
            index_type       = None,
            index_freq       = None,
            window_size      = None,
            last_window      = None,
            last_window_exog = None,
            exog             = pd.Series(np.arange(10)),
            exog_type        = None,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )


@pytest.mark.parametrize("forecaster_name", 
                         ['ForecasterAutoregMultiSeries',
                          'ForecasterAutoregMultiVariate'], 
                         ids=lambda ft: f'forecaster_name: {ft}')
def test_check_predict_input_TypeError_when_last_window_is_not_pandas_DataFrame(forecaster_name):
    """
    `ForecasterAutoregMultiSeries` and `ForecasterAutoregMultiVariate`.
    """
    last_window = np.arange(5)
    err_msg = re.escape(f'`last_window` must be a pandas DataFrame. Got {type(last_window)}.')
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = forecaster_name,
            steps            = 10,
            fitted           = True,
            included_exog    = True,
            index_type       = pd.RangeIndex,
            index_freq       = None,
            window_size      = 5,
            last_window      = last_window,
            last_window_exog = None,
            exog             = pd.Series(np.arange(10)),
            exog_type        = pd.Series,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = '1',
            series_col_names = ['1', '2']
        )


@pytest.mark.parametrize("levels     , last_window", 
                         [('1'       , pd.DataFrame({'3': [1, 2, 3], '4': [1, 2, 3]})), 
                          (['1']     , pd.DataFrame({'3': [1, 2, 3], '4': [1, 2, 3]})), 
                          (['1', '2'], pd.DataFrame({'3': [1, 2, 3], '4': [1, 2, 3]}))], 
                         ids = lambda values : f'levels: {values}'
                        )
def test_check_predict_input_ValueError_when_levels_not_in_last_window_ForecasterAutoregMultiSeries(levels, last_window):
    """
    Check ValueError is raised when levels are no the same as last_window column names.
    """
    err_msg = re.escape(
                    (f'`last_window` must contain a column(s) named as the level(s) to be predicted.\n'
                     f'    `levels` : {levels}.\n'
                     f'    `last_window` columns : {list(last_window.columns)}.')
                )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterAutoregMultiSeries',
            steps            = 10,
            fitted           = True,
            included_exog    = True,
            index_type       = pd.RangeIndex,
            index_freq       = None,
            window_size      = 2,
            last_window      = last_window,
            last_window_exog = None,
            exog             = pd.Series(np.arange(10)),
            exog_type        = pd.Series,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = levels,
            series_col_names = ['1', '2']
        )


def test_check_predict_input_ValueError_when_series_col_names_not_last_window_ForecasterAutoregMultiVariate():
    """
    Check ValueError is raised when column names of series using during fit do not
    match with last_window column names.
    """
    last_window = pd.DataFrame({'l1': [1, 2, 3], '4': [1, 2, 3]})
    series_col_names = ['l1', 'l2']
    err_msg = re.escape(
                    (f'`last_window` columns must be the same as `series` column names.\n'
                     f'    `last_window` columns : {list(last_window.columns)}.\n'
                     f'    `series` columns      : {series_col_names}.')
                )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterAutoregMultiVariate',
            steps            = 10,
            fitted           = True,
            included_exog    = True,
            index_type       = pd.RangeIndex,
            index_freq       = None,
            window_size      = 2,
            last_window      = last_window,
            last_window_exog = None,
            exog             = pd.Series(np.arange(10)),
            exog_type        = pd.Series,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = series_col_names
        )


def test_check_predict_input_TypeError_when_last_window_is_not_pandas_series():
    """
    """
    last_window = np.arange(5)
    err_msg = re.escape(f'`last_window` must be a pandas Series. Got {type(last_window)}.')
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterAutoreg',
            steps            = 10,
            fitted           = True,
            included_exog    = True,
            index_type       = pd.RangeIndex,
            index_freq       = None,
            window_size      = 5,
            last_window      = last_window,
            last_window_exog = None,
            exog             = pd.Series(np.arange(10)),
            exog_type        = pd.Series,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )


def test_check_predict_input_ValueError_when_length_last_window_is_lower_than_window_size():
    """
    """
    window_size = 10

    err_msg = re.escape(
                (f'`last_window` must have as many values as needed to '
                 f'generate the predictors. For this forecaster it is {window_size}.')
              )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterAutoreg',
            steps            = 10,
            fitted           = True,
            included_exog    = True,
            index_type       = pd.RangeIndex,
            index_freq       = None,
            window_size      = window_size,
            last_window      = pd.Series(np.arange(5)),
            last_window_exog = None,
            exog             = pd.Series(np.arange(10)),
            exog_type        = pd.Series,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )


def test_check_predict_input_ValueError_when_last_window_has_missing_values():
    """
    """
    err_msg = re.escape('`last_window` has missing values.')
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterAutoreg',
            steps            = 10,
            fitted           = True,
            included_exog    = True,
            index_type       = pd.RangeIndex,
            index_freq       = None,
            window_size      = 5,
            last_window      = pd.Series([1, 2, 3, 4, 5, np.nan]),
            last_window_exog = None,
            exog             = pd.Series(np.arange(10)),
            exog_type        = pd.Series,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )


def test_check_predict_input_TypeError_when_last_window_index_is_not_of_index_type():
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
            forecaster_name  = 'ForecasterAutoreg',
            steps            = 10,
            fitted           = True,
            included_exog    = False,
            index_type       = index_type,
            index_freq       = None,
            window_size      = 5,
            last_window      = last_window,
            last_window_exog = None,
            exog             = None,
            exog_type        = None,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )


def test_check_predict_input_TypeError_when_last_window_index_frequency_is_not_index_freq():
    """
    """
    last_window = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10, freq='D'))
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
            forecaster_name  = 'ForecasterAutoreg',
            steps            = 10,
            fitted           = True,
            included_exog    = False,
            index_type       = pd.DatetimeIndex,
            index_freq       = index_freq,
            window_size      = 5,
            last_window      = last_window,
            last_window_exog = None,
            exog             = None,
            exog_type        = None,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )


def test_check_predict_input_TypeError_when_exog_is_not_pandas_series_or_dataframe():
    """
    """
    err_msg = re.escape('`exog` must be a pandas Series or DataFrame.')
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterAutoreg',
            steps            = 10,
            fitted           = True,
            included_exog    = True,
            index_type       = pd.DatetimeIndex,
            index_freq       = 'M',
            window_size      = 5,
            last_window      = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10, freq='M')),
            last_window_exog = None,
            exog             = np.arange(10),
            exog_type        = None,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )


def test_check_predict_input_MissingValuesExogWarning_when_exog_has_missing_values():
    """
    """
    warn_msg = re.escape(
                ("`exog` has missing values. Most of machine learning models do "
                 "not allow missing values. `predict` method may fail.")
               )
    with pytest.warns(MissingValuesExogWarning, match = warn_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterAutoreg',
            steps            = 3,
            fitted           = True,
            included_exog    = True,
            index_type       = pd.DatetimeIndex,
            index_freq       = 'M',
            window_size      = 2,
            last_window      = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10, freq='M')),
            last_window_exog = None,
            exog             = pd.Series([1, 2, 3, np.nan], index=pd.date_range(start='11/1/2018', periods=4, freq='M')),
            exog_type        = pd.Series,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )


def test_check_predict_input_TypeError_when_exog_is_not_of_exog_type():
    """
    """
    exog = pd.Series(np.arange(10))
    exog_type = pd.DataFrame

    err_msg = re.escape(f"Expected type for `exog`: {exog_type}. Got {type(exog)}.")
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterAutoreg',
            steps            = 5,
            fitted           = True,
            included_exog    = True,
            index_type       = pd.DatetimeIndex,
            index_freq       = 'M',
            window_size      = 5,
            last_window      = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10, freq='M')),
            last_window_exog = None,
            exog             = exog,
            exog_type        = exog_type,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )


@pytest.mark.parametrize("steps", [10, [1, 2, 3, 4, 5, 6], [2, 6]], 
                         ids=lambda steps: f'steps: {steps}')
def test_check_predict_input_ValueError_when_len_exog_is_less_than_steps(steps):
    """
    """
    last_step = max(steps) if isinstance(steps, list) else steps
    err_msg = re.escape(
                f'`exog` must have at least as many values as the distance to '
                f'the maximum step predicted, {last_step}.'
            )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterAutoreg',
            steps            = steps,
            fitted           = True,
            included_exog    = True,
            index_type       = pd.DatetimeIndex,
            index_freq       = 'M',
            window_size      = 5,
            last_window      = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10, freq='M')),
            last_window_exog = None,
            exog             = pd.Series(np.arange(5)),
            exog_type        = pd.Series,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )


def test_check_predict_input_ValueError_when_exog_is_dataframe_without_columns_in_exog_col_names():
    """
    Raise ValueError when there are missing columns in `exog`.
    """
    exog = pd.DataFrame(np.arange(10).reshape(5, 2), columns=['col1', 'col2'])
    exog_col_names = ['col1', 'col3']

    err_msg = re.escape(
                (f'Missing columns in `exog`. Expected {exog_col_names}. '
                 f'Got {exog.columns.to_list()}.')
              )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterAutoreg',
            steps            = 2,
            fitted           = True,
            included_exog    = True,
            index_type       = pd.DatetimeIndex,
            index_freq       = 'M',
            window_size      = 5,
            last_window      = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10, freq='M')),
            last_window_exog = None,
            exog             = exog,
            exog_type        = pd.DataFrame,
            exog_col_names   = exog_col_names,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )


def test_check_predict_input_TypeError_when_exog_index_is_not_of_index_type():
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
            forecaster_name  = 'ForecasterAutoreg',
            steps            = 10,
            fitted           = True,
            included_exog    = True,
            index_type       = index_type,
            index_freq       = 'M',
            window_size      = 5,
            last_window      = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10, freq='M')),
            last_window_exog = None,
            exog             = exog,
            exog_type        = pd.Series,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )


def test_check_predict_input_TypeError_when_exog_index_frequency_is_not_index_freq():
    """
    """
    exog = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10))
    index_freq = 'M'
    check_exog(exog = exog)
    _, exog_index = preprocess_exog(exog=exog.iloc[:0, ])

    err_msg = re.escape(
                (f'Expected frequency of type {index_freq} for `exog`. '
                 f'Got {exog_index.freqstr}.')
              )
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterAutoreg',
            steps            = 10,
            fitted           = True,
            included_exog    = True,
            index_type       = pd.DatetimeIndex,
            index_freq       = 'M',
            window_size      = 5,
            last_window      = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10, freq='M')),
            last_window_exog = None,
            exog             = exog,
            exog_type        = pd.Series,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )


def test_check_predict_input_ValueError_when_exog_index_does_not_follow_last_window_index_DatetimeIndex():
    """
    Raise exception if `exog` index does not start at the end of `last_window` index when DatetimeIndex.
    """
    exog_datetime = pd.Series(data=np.random.rand(10))
    exog_datetime.index = pd.date_range(start='2022-03-01', periods=10, freq='M')
    lw_datetime = pd.Series(data=np.random.rand(10))
    lw_datetime.index = pd.date_range(start='2022-01-01', periods=10, freq='M')

    expected_index = '2022-11-30 00:00:00'

    err_msg = re.escape(
                (f'To make predictions `exog` must start one step ahead of `last_window`.\n'
                 f'    `last_window` ends at : {lw_datetime.index[-1]}.\n'
                 f'    `exog` starts at      : {exog_datetime.index[0]}.\n'
                 f'     Expected index       : {expected_index}.')
        )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterAutoreg',
            steps            = 10,
            fitted           = True,
            included_exog    = True,
            index_type       = pd.DatetimeIndex,
            index_freq       = 'M',
            window_size      = 5,
            last_window      = lw_datetime,
            last_window_exog = None,
            exog             = exog_datetime,
            exog_type        = pd.Series,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )


def test_check_predict_input_ValueError_when_exog_index_does_not_follow_last_window_index_RangeIndex():
    """
    Raise ValueError if `exog` index does not start at the end of `last_window` index when RangeIndex.
    """
    exog_datetime = pd.Series(data=np.random.rand(10))
    exog_datetime.index = pd.RangeIndex(start=11, stop=21)
    lw_datetime = pd.Series(data=np.random.rand(10))
    lw_datetime.index = pd.RangeIndex(start=0, stop=10)

    expected_index = 10

    err_msg = re.escape(
                (f'To make predictions `exog` must start one step ahead of `last_window`.\n'
                 f'    `last_window` ends at : {lw_datetime.index[-1]}.\n'
                 f'    `exog` starts at      : {exog_datetime.index[0]}.\n'
                 f'     Expected index       : {expected_index}.')
              )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterAutoreg',
            steps            = 10,
            fitted           = True,
            included_exog    = True,
            index_type       = pd.RangeIndex,
            index_freq       = 'M',
            window_size      = 5,
            last_window      = lw_datetime,
            last_window_exog = None,
            exog             = exog_datetime,
            exog_type        = pd.Series,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )


def test_check_predict_input_ValueError_when_ForecasterSarimax_last_window_exog_is_not_None_and_included_exog_is_false():
    """
    Check ValueError is raised when last_window_exog is not None, but included_exog
    is False.
    """
    err_msg = re.escape(
                ('Forecaster trained without exogenous variable/s. '
                 '`last_window_exog` must be `None` when predicting.')
              )   
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterSarimax',
            steps            = 3,
            fitted           = True,
            included_exog    = False,
            index_type       = pd.RangeIndex,
            index_freq       = None,
            window_size      = 3,
            last_window      = pd.Series(np.arange(5)),
            last_window_exog = pd.Series(np.arange(5)),
            exog             = None,
            exog_type        = None,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )


def test_check_predict_input_TypeError_when_ForecasterSarimax_last_window_exog_is_not_pandas_Series_or_DataFrame():
    """
    """
    last_window_exog = np.arange(5)
    err_msg = re.escape(
                (f'`last_window_exog` must be a pandas Series or a '
                 f'pandas DataFrame. Got {type(last_window_exog)}.')
            )
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterSarimax',
            steps            = 10,
            fitted           = True,
            included_exog    = True,
            index_type       = pd.RangeIndex,
            index_freq       = None,
            window_size      = 3,
            last_window      = pd.Series(np.arange(5)),
            last_window_exog = last_window_exog,
            exog             = pd.Series(np.arange(10), index=pd.RangeIndex(start=5, stop=15)),
            exog_type        = pd.Series,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )


def test_check_predict_input_ValueError_when_ForecasterSarimax_length_last_window_exog_is_lower_than_window_size():
    """
    """
    window_size = 10

    err_msg = re.escape(
                (f'`last_window_exog` must have as many values as needed to '
                 f'generate the predictors. For this forecaster it is {window_size}.')
              )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterSarimax',
            steps            = 10,
            fitted           = True,
            included_exog    = True,
            index_type       = pd.RangeIndex,
            index_freq       = None,
            window_size      = window_size,
            last_window      = pd.Series(np.arange(10)),
            last_window_exog = pd.Series(np.arange(5)),
            exog             = pd.Series(np.arange(10), index=pd.RangeIndex(start=10, stop=20)),
            exog_type        = pd.Series,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )


def test_check_predict_input_ValueError_when_ForecasterSarimax_last_window_exog_has_missing_values():
    """
    """
    warn_msg = re.escape(
                ("`last_window_exog` has missing values. Most of machine learning models "
                 "do not allow missing values. `predict` method may fail.")
            )
    with pytest.warns(MissingValuesExogWarning, match = warn_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterSarimax',
            steps            = 10,
            fitted           = True,
            included_exog    = True,
            index_type       = pd.RangeIndex,
            index_freq       = None,
            window_size      = 5,
            last_window      = pd.Series(np.arange(10)),
            last_window_exog = pd.Series([1, 2, 3, 4, 5, np.nan]),
            exog             = pd.Series(np.arange(10), index=pd.RangeIndex(start=10, stop=20)),
            exog_type        = pd.Series,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )


def test_check_predict_input_TypeError_when_ForecasterSarimax_last_window_exog_index_is_not_of_index_type():
    """
    """
    last_window_exog = pd.Series(np.arange(10))
    index_type = pd.DatetimeIndex
    _, last_window_exog_index = preprocess_last_window(
                                last_window = last_window_exog.iloc[:0]
                           ) 

    err_msg = re.escape(
                (f'Expected index of type {index_type} for `last_window_exog`. '
                 f'Got {type(last_window_exog_index)}.')
              )
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterSarimax',
            steps            = 10,
            fitted           = True,
            included_exog    = True,
            index_type       = index_type,
            index_freq       = 'M',
            window_size      = 5,
            last_window      = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10, freq='M')),
            last_window_exog = last_window_exog,
            exog             = pd.Series(np.arange(10), index=pd.date_range(start='30/11/2018', periods=10, freq='M')),
            exog_type        = pd.Series,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )


def test_check_predict_input_TypeError_when_ForecasterSarimax_last_window_exog_index_frequency_is_not_index_freq():
    """
    """
    last_window_exog = pd.Series(np.arange(10), index=pd.date_range(start='2018', periods=10, freq='Y'))
    index_freq = 'M'
    _, last_window_exog_index = preprocess_last_window(
                                last_window = last_window_exog.iloc[:0]
                           )

    err_msg = re.escape(
                (f'Expected frequency of type {index_freq} for `last_window_exog`. '
                 f'Got {last_window_exog_index.freqstr}.')
              )
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterSarimax',
            steps            = 10,
            fitted           = True,
            included_exog    = True,
            index_type       = pd.DatetimeIndex,
            index_freq       = index_freq,
            window_size      = 5,
            last_window      = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10, freq='M')),
            last_window_exog = last_window_exog,
            exog             = pd.Series(np.arange(10), index=pd.date_range(start='30/11/2018', periods=10, freq='M')),
            exog_type        = pd.Series,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )


def test_check_predict_input_ValueError_when_last_window_exog_is_dataframe_without_columns_in_exog_col_names():
    """
    Raise ValueError when there are missing columns in `last_window_exog`, ForecasterSarimax.
    """
    exog = pd.DataFrame(np.arange(10).reshape(5, 2), columns=['col1', 'col3'])
    exog.index=pd.date_range(start='6/1/2018', periods=5, freq='M')
    last_window_exog = pd.DataFrame(np.arange(10).reshape(5, 2), columns=['col1', 'col2'])
    last_window_exog.index=pd.date_range(start='1/1/2018', periods=5, freq='M')
    exog_col_names = ['col1', 'col3']

    err_msg = re.escape(
                (f'Missing columns in `exog`. Expected {exog_col_names}. '
                 f'Got {last_window_exog.columns.to_list()}.') 
              )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterSarimax',
            steps            = 2,
            fitted           = True,
            included_exog    = True,
            index_type       = pd.DatetimeIndex,
            index_freq       = 'M',
            window_size      = 5,
            last_window      = pd.Series(np.arange(5), index=pd.date_range(start='1/1/2018', periods=5, freq='M')),
            last_window_exog = last_window_exog,
            exog             = exog,
            exog_type        = pd.DataFrame,
            exog_col_names   = exog_col_names,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )