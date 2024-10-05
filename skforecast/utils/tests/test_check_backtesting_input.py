# Unit test check_backtesting_input
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.exceptions import NotFittedError
from skforecast.Sarimax import Sarimax
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from skforecast.model_selection._split import TimeSeriesFold
from skforecast.utils import check_backtesting_input

# Fixtures
from skforecast.model_selection.tests.fixtures_model_selection import y
from skforecast.model_selection_multiseries.tests.fixtures_model_selection_multiseries import series


def test_check_backtesting_input_TypeError_when_cv_not_TimeSeries_Fold():
    """
    Test TypeError is raised in check_backtesting_input if `cv` is not a
    TimeSeriesFold object.
    """
    forecaster = ForecasterAutoreg(regressor=Ridge(), lags=2)
    y = pd.Series(np.arange(50))
    y.index = pd.date_range(start='2000-01-01', periods=len(y), freq='D')
    
    class BadCv():
        pass

    err_msg = re.escape("`cv` must be a TimeSeriesFold object. Got BadCv.")
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = BadCv(),
            metric                  = 'mean_absolute_error',
            y                       = y,
            series                  = None,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            show_progress           = False,
            suppress_warnings       = False
        )


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoreg(regressor=Ridge(), lags=2),
                          ForecasterAutoregDirect(regressor=Ridge(), lags=2, steps=3),
                          ForecasterSarimax(regressor=Sarimax(order=(1, 1, 1)))], 
                         ids = lambda fr: f'forecaster: {type(fr).__name__}')
def test_check_backtesting_input_TypeError_when_y_is_not_pandas_Series_uniseries(forecaster):
    """
    Test TypeError is raised in check_backtesting_input if `y` is not a 
    pandas Series in forecasters uni-series.
    """
    bad_y = np.arange(50)
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(bad_y) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True,
             verbose               = False
         )

    err_msg = re.escape("`y` must be a pandas Series.")
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = 'mean_absolute_error',
            y                       = bad_y,
            series                  = None,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            show_progress           = False,
            suppress_warnings       = False
        )


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoregMultiVariate(regressor=Ridge(), lags=2, 
                                                        steps=3, level='l1')], 
                         ids = lambda fr: f'forecaster: {type(fr).__name__}')
def test_check_backtesting_input_TypeError_when_series_is_not_pandas_DataFrame_multiseries(forecaster):
    """
    Test TypeError is raised in check_backtesting_input if `series` is not a 
    pandas DataFrame in forecasters multiseries.
    """
    bad_series = pd.Series(np.arange(50))

    err_msg = re.escape("`series` must be a pandas DataFrame.")
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            steps                   = 3,
            metric                  = 'mean_absolute_error',
            y                       = None,
            series                  = bad_series,
            initial_train_size      = len(bad_series[:-12]),
            fixed_train_size        = False,
            gap                     = 0,
            allow_incomplete_fold   = False,
            refit                   = False,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            verbose                 = False,
            show_progress           = False,
            suppress_warnings       = False
        )


def test_check_backtesting_input_TypeError_when_series_is_not_pandas_DataFrame_multiseries_dict():
    """
    Test TypeError is raised in check_backtesting_input if `series` is not a 
    pandas DataFrame in forecasters multiseries with dict.
    """
    forecaster = ForecasterAutoregMultiSeries(regressor=Ridge(), lags=2)
    bad_series = pd.Series(np.arange(50))

    err_msg = re.escape(
        (f"`series` must be a pandas DataFrame or a dict of DataFrames or Series. "
         f"Got {type(bad_series)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            steps                   = 3,
            metric                  = 'mean_absolute_error',
            y                       = None,
            series                  = bad_series,
            initial_train_size      = len(bad_series[:-12]),
            fixed_train_size        = False,
            gap                     = 0,
            allow_incomplete_fold   = False,
            refit                   = False,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            verbose                 = False,
            show_progress           = False,
            suppress_warnings       = False
        )


def test_check_backtesting_input_TypeError_when_series_is_dict_of_pandas_Series_multiseries_dict():
    """
    Test TypeError is raised in check_backtesting_input if `series` is not a 
    dict of pandas Series in forecasters multiseries with dict.
    """
    forecaster = ForecasterAutoregMultiSeries(regressor=Ridge(), lags=2)
    bad_series = {'l1': np.arange(50)}

    err_msg = re.escape(
        ("If `series` is a dictionary, all series must be a named "
         "pandas Series or a pandas DataFrame with a single column. "
         "Review series: ['l1']")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            steps                   = 3,
            metric                  = 'mean_absolute_error',
            y                       = None,
            series                  = bad_series,
            initial_train_size      = len(bad_series['l1'][:-12]),
            fixed_train_size        = False,
            gap                     = 0,
            allow_incomplete_fold   = False,
            refit                   = False,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            verbose                 = False,
            show_progress           = False,
            suppress_warnings       = False
        )


def test_check_backtesting_input_ValueError_when_series_is_dict_no_DatetimeIndex_multiseries_dict():
    """
    Test ValueError is raised in check_backtesting_input if `series` is a 
    dict with pandas Series with no DatetimeIndex in forecasters 
    multiseries with dict.
    """
    forecaster = ForecasterAutoregMultiSeries(regressor=Ridge(), lags=2)
    series_dict = {
        'l1': pd.Series(np.arange(50)),
        'l2': pd.Series(np.arange(50))
    }

    err_msg = re.escape(
        ("If `series` is a dictionary, all series must have a Pandas DatetimeIndex "
         "as index with the same frequency. Review series: ['l1', 'l2']")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            steps                   = 3,
            metric                  = 'mean_absolute_error',
            y                       = None,
            series                  = series_dict,
            initial_train_size      = len(series_dict['l1'][:-12]),
            fixed_train_size        = False,
            gap                     = 0,
            allow_incomplete_fold   = False,
            refit                   = False,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            verbose                 = False,
            show_progress           = False,
            suppress_warnings       = False
        )


def test_check_backtesting_input_ValueError_when_series_is_dict_diff_freq_multiseries_dict():
    """
    Test ValueError is raised in check_backtesting_input if `series` is a 
    dict with pandas Series of difference frequency in forecasters 
    multiseries with dict.
    """
    forecaster = ForecasterAutoregMultiSeries(regressor=Ridge(), lags=2)
    series_dict = {
        'l1': pd.Series(np.arange(50)),
        'l2': pd.Series(np.arange(50))
    }
    series_dict['l1'].index = pd.date_range(
        start='2000-01-01', periods=len(series_dict['l1']), freq='D'
    )
    series_dict['l2'].index = pd.date_range(
        start='2000-01-01', periods=len(series_dict['l2']), freq='MS'
    )

    err_msg = re.escape(
        ("If `series` is a dictionary, all series must have a Pandas DatetimeIndex "
         "as index with the same frequency. Found frequencies: ['<Day>', '<MonthBegin>']")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            steps                   = 3,
            metric                  = 'mean_absolute_error',
            y                       = None,
            series                  = series_dict,
            initial_train_size      = len(series_dict['l1'][:-12]),
            fixed_train_size        = False,
            gap                     = 0,
            allow_incomplete_fold   = False,
            refit                   = False,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            verbose                 = False,
            show_progress           = False,
            suppress_warnings       = False
        )


def test_check_backtesting_input_TypeError_when_not_valid_exog_type_multiseries_dict():
    """
    Test TypeError is raised in check_backtesting_input if `exog` is not a
    pandas Series, DataFrame, dictionary of pandas Series/DataFrames or None.
    """
    forecaster = ForecasterAutoregMultiSeries(regressor=Ridge(), lags=2)
    series_dict = {
        'l1': pd.Series(np.arange(50)),
        'l2': pd.Series(np.arange(50))
    }
    series_dict['l1'].index = pd.date_range(
        start='2000-01-01', periods=len(series_dict['l1']), freq='D'
    )
    series_dict['l2'].index = pd.date_range(
        start='2000-01-01', periods=len(series_dict['l2']), freq='D'
    )

    bad_exog = np.arange(50)

    err_msg = re.escape(
        (f"`exog` must be a pandas Series, DataFrame, dictionary of pandas "
         f"Series/DataFrames or None. Got {type(bad_exog)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            steps                   = 3,
            metric                  = 'mean_absolute_error',
            y                       = None,
            series                  = series_dict,
            exog                    = bad_exog,
            initial_train_size      = len(series_dict['l1'][:-12]),
            fixed_train_size        = False,
            gap                     = 0,
            allow_incomplete_fold   = False,
            refit                   = False,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            verbose                 = False,
            show_progress           = False,
            suppress_warnings       = False
        )


def test_check_backtesting_input_TypeError_when_not_valid_exog_dict_type_multiseries_dict():
    """
    Test TypeError is raised in check_backtesting_input if `exog` is not a
    dictionary of pandas Series/DataFrames.
    """
    forecaster = ForecasterAutoregMultiSeries(regressor=Ridge(), lags=2)
    series_dict = {
        'l1': pd.Series(np.arange(50)),
        'l2': pd.Series(np.arange(50))
    }
    series_dict['l1'].index = pd.date_range(
        start='2000-01-01', periods=len(series_dict['l1']), freq='D'
    )
    series_dict['l2'].index = pd.date_range(
        start='2000-01-01', periods=len(series_dict['l2']), freq='D'
    )

    bad_exog = {'l1': np.arange(50)}

    err_msg = re.escape(
        ("If `exog` is a dictionary, All exog must be a named pandas "
         "Series, a pandas DataFrame or None. Review exog: ['l1']")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            steps                   = 3,
            metric                  = 'mean_absolute_error',
            y                       = None,
            series                  = series_dict,
            exog                    = bad_exog,
            initial_train_size      = len(series_dict['l1'][:-12]),
            fixed_train_size        = False,
            gap                     = 0,
            allow_incomplete_fold   = False,
            refit                   = False,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            verbose                 = False,
            show_progress           = False,
            suppress_warnings       = False
        )


def test_check_backtesting_input_TypeError_when_not_valid_exog_type():
    """
    Test TypeError is raised in check_backtesting_input if `exog` is not a
    pandas Series, DataFrame or None.
    """
    y = pd.Series(np.arange(50))
    y.index = pd.date_range(start='2000-01-01', periods=len(y), freq='D')

    forecaster = ForecasterAutoreg(regressor=Ridge(), lags=2)

    bad_exog = np.arange(50)

    err_msg = re.escape(
        (f"`exog` must be a pandas Series, DataFrame or None. Got {type(bad_exog)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            steps                   = 3,
            metric                  = 'mean_absolute_error',
            y                       = y,
            series                  = None,
            exog                    = bad_exog,
            initial_train_size      = len(y[:-12]),
            fixed_train_size        = False,
            gap                     = 0,
            allow_incomplete_fold   = False,
            refit                   = False,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            verbose                 = False,
            show_progress           = False,
            suppress_warnings       = False
        )


@pytest.mark.parametrize("steps", 
                         ['not_int', 2.3, 0], 
                         ids = lambda value: f'steps: {value}')
def test_check_backtesting_input_TypeError_when_steps_not_int_greater_or_equal_1(steps):
    """
    Test TypeError is raised in check_backtesting_input if `steps` is not an 
    integer greater than or equal to 1.
    """
    forecaster = ForecasterAutoreg(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )
    
    err_msg = re.escape(
        f"`steps` must be an integer greater than or equal to 1. Got {steps}."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            steps                   = steps,
            metric                  = 'mean_absolute_error',
            y                       = y,
            series                  = None,
            initial_train_size      = len(y[:-12]),
            fixed_train_size        = False,
            gap                     = 0,
            allow_incomplete_fold   = False,
            refit                   = False,
            interval                = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            verbose                 = False,
            show_progress           = False,
            suppress_warnings       = False
        )


@pytest.mark.parametrize("gap", 
                         ['not_int', 2.3, -1], 
                         ids = lambda value: f'gap: {value}')
def test_check_backtesting_input_TypeError_when_gap_not_int_greater_or_equal_0(gap):
    """
    Test TypeError is raised in check_backtesting_input if `gap` is not an 
    integer greater than or equal to 0.
    """
    forecaster = ForecasterAutoreg(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )
    
    err_msg = re.escape(
        f"`gap` must be an integer greater than or equal to 0. Got {gap}."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            steps                   = 3,
            metric                  = 'mean_absolute_error',
            y                       = y,
            series                  = None,
            initial_train_size      = len(y[:-12]),
            fixed_train_size        = False,
            gap                     = gap,
            allow_incomplete_fold   = False,
            refit                   = False,
            interval                = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            verbose                 = False,
            show_progress           = False,
            suppress_warnings       = False
        )


@pytest.mark.parametrize("skip_folds", 
                         ['not_int', 2.3], 
                         ids = lambda value: f'skip_folds: {value}')
def test_check_backtesting_input_TypeError_when_skip_folds_not_int_list_or_None(skip_folds):
    """
    Test TypeError is raised in check_backtesting_input if `skip_folds` is not an 
    integer, a list of integers or None.
    """
    forecaster = ForecasterAutoreg(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )
    
    err_msg = re.escape(
        (f"`skip_folds` must be an integer greater than 0, a list of "
         f"integers or `None`. Got {type(skip_folds)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            steps                   = 3,
            metric                  = 'mean_absolute_error',
            y                       = y,
            series                  = None,
            initial_train_size      = len(y[:-12]),
            fixed_train_size        = False,
            gap                     = 0,
            skip_folds              = skip_folds,
            allow_incomplete_fold   = False,
            refit                   = False,
            interval                = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            verbose                 = False,
            show_progress           = False,
            suppress_warnings       = False
        )


@pytest.mark.parametrize("skip_folds", 
                         [0, -1], 
                         ids = lambda value: f'skip_folds: {value}')
def test_check_backtesting_input_ValueError_when_skip_folds_int_less_than_1(skip_folds):
    """
    Test ValueError is raised in check_backtesting_input if `skip_folds` is 
    an integer less than 1.
    """
    forecaster = ForecasterAutoreg(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )
    
    err_msg = re.escape(
        (f"`skip_folds` must be an integer greater than 0, a list of "
         f"integers or `None`. Got {skip_folds}.")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            steps                   = 3,
            metric                  = 'mean_absolute_error',
            y                       = y,
            series                  = None,
            initial_train_size      = len(y[:-12]),
            fixed_train_size        = False,
            gap                     = 0,
            skip_folds              = skip_folds,
            allow_incomplete_fold   = False,
            refit                   = False,
            interval                = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            verbose                 = False,
            show_progress           = False,
            suppress_warnings       = False
        )


def test_check_backtesting_input_ValueError_when_skip_folds_list_with_0():
    """
    Test ValueError is raised in check_backtesting_input if `skip_folds` is 
    a list containing the value 0.
    """
    forecaster = ForecasterAutoreg(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )
    
    err_msg = re.escape(
        ("`skip_folds` cannot contain the value 0, the first fold is "
         "needed to train the forecaster.")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            steps                   = 3,
            metric                  = 'mean_absolute_error',
            y                       = y,
            series                  = None,
            initial_train_size      = len(y[:-12]),
            fixed_train_size        = False,
            gap                     = 0,
            skip_folds              = [0, 2],
            allow_incomplete_fold   = False,
            refit                   = False,
            interval                = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            verbose                 = False,
            show_progress           = False,
            suppress_warnings       = False
        )


def test_check_backtesting_input_TypeError_when_metric_not_correct_type():
    """
    Test TypeError is raised in check_backtesting_input if `metric` is not string, 
    a callable function, or a list containing multiple strings and/or callables.
    """
    forecaster = ForecasterAutoreg(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )
    metric = 5
    
    err_msg = re.escape(
        (f"`metric` must be a string, a callable function, or a list containing "
         f"multiple strings and/or callables. Got {type(metric)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            steps                   = 5,
            metric                  = metric,
            y                       = y,
            series                  = None,
            initial_train_size      = len(y[:-12]),
            fixed_train_size        = False,
            gap                     = 0,
            allow_incomplete_fold   = False,
            refit                   = False,
            interval                = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            verbose                 = False,
            show_progress           = False,
            suppress_warnings       = False
        )


@pytest.mark.parametrize("initial_train_size", 
                         [20., 21.2, 'not_int'], 
                         ids = lambda value: f'initial_train_size: {value}')
def test_check_backtesting_input_TypeError_when_initial_train_size_is_not_an_int_or_None(initial_train_size):
    """
    Test TypeError is raised in check_backtesting_input when 
    initial_train_size is not an integer.
    """
    forecaster = ForecasterAutoreg(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )
    
    err_msg = re.escape(
        (f"If used, `initial_train_size` must be an integer greater than the "
         f"window_size of the forecaster. Got type {type(initial_train_size)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            steps                   = 3,
            metric                  = 'mean_absolute_error',
            y                       = y,
            series                  = None,
            initial_train_size      = initial_train_size,
            fixed_train_size        = False,
            gap                     = 0,
            allow_incomplete_fold   = False,
            refit                   = False,
            interval                = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            verbose                 = False,
            show_progress           = False,
            suppress_warnings       = False
        )


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoreg(regressor=Ridge(), lags=2),
                          ForecasterAutoregMultiSeries(regressor=Ridge(), lags=2)], 
                         ids = lambda fr: f'forecaster: {type(fr).__name__}')
def test_check_backtesting_input_ValueError_when_initial_train_size_more_than_or_equal_to_data_length(forecaster):
    """
    Test ValueError is raised in check_backtesting_input when 
    initial_train_size >= length `y` or `series` depending on the forecaster.
    """
    if type(forecaster).__name__ == 'ForecasterAutoreg':
        data_length = len(y)
        data_name = 'y'
    else:
        data_length = len(series)
        data_name = 'series'

    initial_train_size = data_length
    
    err_msg = re.escape(
        (f"If used, `initial_train_size` must be an integer smaller "
         f"than the length of `{data_name}` ({data_length}).")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            steps                   = 3,
            metric                  = 'mean_absolute_error',
            y                       = y,
            series                  = series,
            initial_train_size      = initial_train_size,
            fixed_train_size        = False,
            gap                     = 0,
            allow_incomplete_fold   = False,
            refit                   = False,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            verbose                 = False,
            show_progress           = False,
            suppress_warnings       = False
        )


def test_check_backtesting_input_ValueError_when_initial_train_size_less_than_forecaster_window_size():
    """
    Test ValueError is raised in check_backtesting_input when 
    initial_train_size < forecaster.window_size.
    """
    forecaster = ForecasterAutoreg(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )

    initial_train_size = forecaster.window_size - 1
    
    err_msg = re.escape(
        (f"If used, `initial_train_size` must be an integer greater than "
         f"the window_size of the forecaster ({forecaster.window_size}).")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            steps                   = 3,
            metric                  = 'mean_absolute_error',
            y                       = y,
            series                  = series,
            initial_train_size      = initial_train_size,
            fixed_train_size        = False,
            gap                     = 0,
            allow_incomplete_fold   = False,
            refit                   = False,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            verbose                 = False,
            show_progress           = False,
            suppress_warnings       = False
        )


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoreg(regressor=Ridge(), lags=2),
                          ForecasterAutoregMultiSeries(regressor=Ridge(), lags=2)], 
                         ids = lambda fr: f'forecaster: {type(fr).__name__}')
def test_check_backtesting_input_ValueError_when_initial_train_size_plus_gap_less_than_data_length(forecaster):
    """
    Test ValueError is raised in check_backtesting_input when 
    initial_train_size + gap >= length `y` or `series` depending on the forecaster.
    """
    if type(forecaster).__name__ == 'ForecasterAutoreg':
        data_length = len(y)
        data_name = 'y'
    else:
        data_length = len(series)
        data_name = 'series'

    initial_train_size = len(y) - 1
    gap = 2
    
    err_msg = re.escape(
        (f"The combination of initial_train_size {initial_train_size} and "
         f"gap {gap} cannot be greater than the length of `{data_name}` "
         f"({data_length}).")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            steps                   = 3,
            metric                  = 'mean_absolute_error',
            y                       = y,
            series                  = series,
            initial_train_size      = initial_train_size,
            fixed_train_size        = False,
            gap                     = gap,
            allow_incomplete_fold   = False,
            refit                   = False,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            verbose                 = False,
            show_progress           = False,
            suppress_warnings       = False
        )


def test_check_backtesting_input_ValueError_ForecasterSarimax_when_initial_train_size_is_None():
    """
    Test ValueError is raised in check_backtesting_input when initial_train_size 
    is None with a ForecasterSarimax.
    """
    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 1, 1)))

    initial_train_size = None
    
    err_msg = re.escape(
        (f"`initial_train_size` must be an integer smaller than the "
         f"length of `y` ({len(y)}).")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            steps                   = 3,
            metric                  = 'mean_absolute_error',
            y                       = y,
            series                  = series,
            initial_train_size      = initial_train_size,
            fixed_train_size        = False,
            gap                     = 0,
            allow_incomplete_fold   = False,
            refit                   = False,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            verbose                 = False,
            show_progress           = False,
            suppress_warnings       = False
        )


def test_check_backtesting_input_NotFittedError_when_initial_train_size_None_and_forecaster_not_fitted():
    """
    Test NotFittedError is raised in check_backtesting_input when 
    initial_train_size is None and forecaster is not fitted.
    """
    forecaster = ForecasterAutoreg(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )

    initial_train_size = None
    
    err_msg = re.escape(
        ("`forecaster` must be already trained if no `initial_train_size` "
         "is provided.")
    )
    with pytest.raises(NotFittedError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            steps                   = 3,
            metric                  = 'mean_absolute_error',
            y                       = y,
            series                  = series,
            initial_train_size      = initial_train_size,
            fixed_train_size        = False,
            gap                     = 0,
            allow_incomplete_fold   = False,
            refit                   = False,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            verbose                 = False,
            show_progress           = False,
            suppress_warnings       = False
        )


def test_check_backtesting_input_ValueError_when_initial_train_size_None_and_refit_True():
    """
    Test ValueError is raised in check_backtesting_input when initial_train_size 
    is None and refit is True.
    """
    forecaster = ForecasterAutoreg(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )
    forecaster.is_fitted = True

    initial_train_size = None
    refit = True
    
    err_msg = re.escape(
        "`refit` is only allowed when `initial_train_size` is not `None`."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            steps                   = 3,
            metric                  = 'mean_absolute_error',
            y                       = y,
            series                  = series,
            initial_train_size      = initial_train_size,
            fixed_train_size        = False,
            gap                     = 0,
            allow_incomplete_fold   = False,
            refit                   = refit,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            verbose                 = False,
            show_progress           = False,
            suppress_warnings       = False
        )
        

@pytest.mark.parametrize("refit", 
                         ['not_bool_int', -1, 1.5], 
                         ids = lambda value: f'refit: {value}')
def test_check_backtesting_input_TypeError_when_refit_not_bool_or_int(refit):
    """
    Test TypeError is raised in check_backtesting_input when `refit` is not a 
    boolean or a integer greater than 0.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )
    
    err_msg = re.escape(f"`refit` must be a boolean or an integer greater than 0. Got {refit}.")
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster         = forecaster,
            steps              = 3,
            metric             = 'mean_absolute_error',
            y                  = y,
            series             = series,
            initial_train_size = len(y[:-12]),
            refit              = refit,
            gap                = 0,
            interval           = None,
            alpha              = None,
            n_boot             = 500,
            random_state       = 123,
        )


@pytest.mark.parametrize("boolean_argument", 
                         ['add_aggregated_metric', 'use_in_sample_residuals', 
                          'use_binned_residuals', 'show_progress', 
                          'suppress_warnings', 'suppress_warnings_fit'], 
                         ids = lambda argument: f'{argument}')
def test_check_backtesting_input_TypeError_when_boolean_arguments_not_bool(boolean_argument):
    """
    Test TypeError is raised in check_backtesting_input when boolean arguments 
    are not boolean.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )
    
    boolean_arguments = {
        'add_aggregated_metric': False,
        'use_in_sample_residuals': False,
        'use_binned_residuals': False,
        'show_progress': False,
        'suppress_warnings': False,
        'suppress_warnings_fit': False
    }
    boolean_arguments[boolean_argument] = 'not_bool'
    
    err_msg = re.escape(f"`{boolean_argument}` must be a boolean: `True`, `False`.")
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster         = forecaster,
            steps              = 3,
            metric             = 'mean_absolute_error',
            y                  = y,
            series             = series,
            initial_train_size = len(y[:-12]),
            refit              = False,
            gap                = 0,
            interval           = None,
            alpha              = None,
            n_boot             = 500,
            random_state       = 123,
            **boolean_arguments
        )


@pytest.mark.parametrize("int_argument, value",
                         [('n_boot', 2.2), 
                          ('n_boot', -2),
                          ('random_state', 'not_int'),  
                          ('random_state', -3)], 
                         ids = lambda argument: f'{argument}')
def test_check_backtesting_input_TypeError_when_integer_arguments_not_int_or_greater_than_0(int_argument, value):
    """
    Test TypeError is raised in check_backtesting_input when integer arguments 
    are not int or are greater than 0.
    """
    forecaster = ForecasterAutoreg(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )
    
    integer_arguments = {'n_boot': 500,
                         'random_state': 123}
    integer_arguments[int_argument] = value
    
    err_msg = re.escape(f"`{int_argument}` must be an integer greater than 0. Got {value}.")
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            steps                   = 3,
            metric                  = 'mean_absolute_error',
            y                       = y,
            series                  = series,
            initial_train_size      = len(y[:-12]),
            fixed_train_size        = False,
            gap                     = 0,
            allow_incomplete_fold   = False,
            refit                   = False,
            interval                = None,
            alpha                   = None,
            use_in_sample_residuals = True,
            verbose                 = False,
            show_progress           = False,
            suppress_warnings       = False,
            **integer_arguments
        )


@pytest.mark.parametrize("n_jobs", 
                         [1.0, 'not_int_auto'], 
                         ids = lambda value: f'n_jobs: {value}')
def test_check_backtesting_input_TypeError_when_n_jobs_not_int_or_auto(n_jobs):
    """
    Test TypeError is raised in check_backtesting_input when n_jobs  
    is not an integer or 'auto'.
    """
    forecaster = ForecasterAutoreg(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )
    
    err_msg = re.escape(
        (f"`n_jobs` must be an integer or `'auto'`. Got {n_jobs}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            steps                   = 3,
            metric                  = 'mean_absolute_error',
            y                       = y,
            series                  = None,
            initial_train_size      = len(y[:-12]),
            fixed_train_size        = False,
            gap                     = 0,
            allow_incomplete_fold   = False,
            refit                   = False,
            interval                = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            n_jobs                  = n_jobs,
            verbose                 = False,
            show_progress           = False,
            suppress_warnings       = False
        )


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoreg(regressor=Ridge(), lags=2),
                          ForecasterAutoregMultiSeries(regressor=Ridge(), lags=2)], 
                         ids = lambda fr: f'forecaster: {type(fr).__name__}')
def test_check_backtesting_input_ValueError_when_not_enough_data_to_create_a_fold(forecaster):
    """
    Test ValueError is raised in check_backtesting_input when there is not enough 
    data to evaluate even single fold because `allow_incomplete_fold` = `False`.
    """
    if type(forecaster).__name__ == 'ForecasterAutoreg':
        data_length = len(y)
    else:
        data_length = len(series)
    
    initial_train_size = data_length - 12
    gap = 10
    steps = 5
    
    err_msg = re.escape(
        (f"There is not enough data to evaluate {steps} steps in a single "
         f"fold. Set `allow_incomplete_fold` to `True` to allow incomplete folds.\n"
         f"    Data available for test : {data_length - (initial_train_size + gap)}\n"
         f"    Steps                   : {steps}")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            steps                   = steps,
            metric                  = 'mean_absolute_error',
            y                       = y,
            series                  = series,
            initial_train_size      = initial_train_size,
            fixed_train_size        = False,
            gap                     = gap,
            allow_incomplete_fold   = False,
            refit                   = False,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            verbose                 = False,
            show_progress           = False,
            suppress_warnings       = False
        )
