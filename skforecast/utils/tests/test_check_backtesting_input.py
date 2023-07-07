# Unit test check_backtesting_input
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from pmdarima.arima import ARIMA
from sklearn.linear_model import Ridge
from sklearn.exceptions import NotFittedError
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.ForecasterAutoregMultiSeriesCustom import ForecasterAutoregMultiSeriesCustom
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from skforecast.utils import check_backtesting_input

# Fixtures
from skforecast.model_selection.tests.fixtures_model_selection import y
from skforecast.model_selection_multiseries.tests.fixtures_model_selection_multiseries import series

def create_predictors(y): # pragma: no cover
    """
    Create first 3 lags of a time series.
    """
    
    lags = y[-1:-4:-1]
    
    return lags


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoreg(regressor=Ridge(), lags=2),
                          ForecasterAutoregCustom(regressor=Ridge(), window_size=3,
                                                  fun_predictors=create_predictors),
                          ForecasterAutoregDirect(regressor=Ridge(), lags=2, steps=3),
                          ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))], 
                         ids = lambda fr : f'forecaster: {type(fr).__name__}' )
def test_check_backtesting_input_TypeError_when_y_is_not_pandas_Series_uniseries(forecaster):
    """
    Test TypeError is raised in check_backtesting_input if `y` is not a 
    pandas Series in forecasters uni-series.
    """
    bad_y = np.arange(50)

    err_msg = re.escape("`y` must be a pandas Series.")
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster            = forecaster,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            y                     = bad_y,
            series                = None,
            initial_train_size    = len(bad_y[:-12]),
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            refit                 = False,
            interval              = None,
            alpha                 = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
        )


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoregMultiSeries(regressor=Ridge(), lags=2),
                          ForecasterAutoregMultiSeriesCustom(regressor=Ridge(), 
                                                             window_size=3,
                                                             fun_predictors=create_predictors),
                          ForecasterAutoregMultiVariate(regressor=Ridge(), lags=2, 
                                                        steps=3, level='l1')], 
                         ids = lambda fr : f'forecaster: {type(fr).__name__}' )
def test_check_backtesting_input_TypeError_when_series_is_not_pandas_DataFrame_multiseries(forecaster):
    """
    Test TypeError is raised in check_backtesting_input if `series` is not a 
    pandas DataFrame in forecasters multiseries.
    """
    bad_series = pd.Series(np.arange(50))

    err_msg = re.escape("`series` must be a pandas DataFrame.")
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster            = forecaster,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            y                     = None,
            series                = bad_series,
            initial_train_size    = len(bad_series[:-12]),
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            refit                 = False,
            interval              = None,
            alpha                 = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
        )


@pytest.mark.parametrize("steps", 
                         ['not_int', 2.3, 0], 
                         ids = lambda value : f'steps: {value}' )
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
            forecaster            = forecaster,
            steps                 = steps,
            metric                = 'mean_absolute_error',
            y                     = y,
            series                = None,
            initial_train_size    = len(y[:-12]),
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            refit                 = False,
            interval              = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
        )


@pytest.mark.parametrize("gap", 
                         ['not_int', 2.3, -1], 
                         ids = lambda value : f'gap: {value}' )
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
            forecaster            = forecaster,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            y                     = y,
            series                = None,
            initial_train_size    = len(y[:-12]),
            fixed_train_size      = False,
            gap                   = gap,
            allow_incomplete_fold = False,
            refit                 = False,
            interval              = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
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
            forecaster            = forecaster,
            steps                 = 5,
            metric                = metric,
            y                     = y,
            series                = None,
            initial_train_size    = len(y[:-12]),
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            refit                 = False,
            interval              = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
        )


@pytest.mark.parametrize("initial_train_size", 
                         [20., 21.2, 'not_int'], 
                         ids = lambda value : f'initial_train_size: {value}' )
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
            forecaster            = forecaster,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            y                     = y,
            series                = None,
            initial_train_size    = initial_train_size,
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            refit                 = False,
            interval              = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
        )


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoreg(regressor=Ridge(), lags=2),
                          ForecasterAutoregMultiSeries(regressor=Ridge(), lags=2)], 
                         ids = lambda fr : f'forecaster: {type(fr).__name__}' )
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
            forecaster            = forecaster,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            y                     = y,
            series                = series,
            initial_train_size    = initial_train_size,
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            refit                 = False,
            interval              = None,
            alpha                 = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
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
            forecaster            = forecaster,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            y                     = y,
            series                = series,
            initial_train_size    = initial_train_size,
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            refit                 = False,
            interval              = None,
            alpha                 = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
        )


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoreg(regressor=Ridge(), lags=2),
                          ForecasterAutoregMultiSeries(regressor=Ridge(), lags=2)], 
                         ids = lambda fr : f'forecaster: {type(fr).__name__}' )
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
            forecaster            = forecaster,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            y                     = y,
            series                = series,
            initial_train_size    = initial_train_size,
            fixed_train_size      = False,
            gap                   = gap,
            allow_incomplete_fold = False,
            refit                 = False,
            interval              = None,
            alpha                 = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
        )


def test_check_backtesting_input_ValueError_when_series_different_length_initial_train_size():
    """
    Test ValueError is raised in check_backtesting_input when series have different 
    length and initial_train_size is not enough to reach the first non-null value.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )
    series_nan = series.copy()
    series_nan['l2'].iloc[:20] = np.nan
    
    err_msg = re.escape(
                    ("All values of series 'l2' are NaN. When working "
                     "with series of different lengths, make sure that "
                     "`initial_train_size` has an appropriate value so that "
                     "all series reach the first non-null value.")
                )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster            = forecaster,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            y                     = None,
            series                = series_nan,
            initial_train_size    = 15,
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            refit                 = False,
            interval              = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
        )


def test_check_backtesting_input_ValueError_ForecasterSarimax_when_initial_train_size_is_None():
    """
    Test ValueError is raised in check_backtesting_input when initial_train_size 
    is None with a ForecasterSarimax.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(order=(1,1,1)))

    initial_train_size = None
    
    err_msg = re.escape(
                (f"`initial_train_size` must be an integer smaller than the "
                 f"length of `y` ({len(y)}).")
              )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster            = forecaster,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            y                     = y,
            series                = series,
            initial_train_size    = initial_train_size,
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            refit                 = False,
            interval              = None,
            alpha                 = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
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
            forecaster            = forecaster,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            y                     = y,
            series                = series,
            initial_train_size    = initial_train_size,
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            refit                 = False,
            interval              = None,
            alpha                 = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
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
    forecaster.fitted = True

    initial_train_size = None
    refit = True
    
    err_msg = re.escape(
                "`refit` is only allowed when `initial_train_size` is not `None`."
            )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster            = forecaster,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            y                     = y,
            series                = series,
            initial_train_size    = initial_train_size,
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            refit                 = refit,
            interval              = None,
            alpha                 = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
        )
        

@pytest.mark.parametrize("refit", 
                         ['not_bool_int', -1, 1.5], 
                         ids = lambda value : f'refit: {value}' )
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
            forecaster            = forecaster,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            y                     = y,
            series                = series,
            initial_train_size    = len(y[:-12]),
            refit                 = refit,
            gap                   = 0,
            interval              = None,
            alpha                 = None,
            n_boot                = 500,
            random_state          = 123,
        )


@pytest.mark.parametrize("boolean_argument", 
                         ['fixed_train_size', 'allow_incomplete_fold', 
                          'in_sample_residuals', 'verbose', 'show_progress'], 
                         ids = lambda argument : f'{argument}' )
def test_check_backtesting_input_TypeError_when_boolean_arguments_not_bool(boolean_argument):
    """
    Test TypeError is raised in check_backtesting_input when boolean arguments 
    are not boolean.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )
    
    boolean_arguments = {'fixed_train_size': False,
                         'allow_incomplete_fold': False,
                         'in_sample_residuals': False,
                         'verbose': False,
                         'show_progress': False}
    boolean_arguments[boolean_argument] = 'not_bool'
    
    err_msg = re.escape(f"`{boolean_argument}` must be a boolean: `True`, `False`.")
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster            = forecaster,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            y                     = y,
            series                = series,
            initial_train_size    = len(y[:-12]),
            refit                 = False,
            gap                   = 0,
            interval              = None,
            alpha                 = None,
            n_boot                = 500,
            random_state          = 123,
            **boolean_arguments
        )


@pytest.mark.parametrize("int_argument, value",
                         [('n_boot', 2.2), 
                          ('n_boot', -2),
                          ('random_state', 'not_int'),  
                          ('random_state', -3)], 
                         ids = lambda argument : f'{argument}' )
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
            forecaster            = forecaster,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            y                     = y,
            series                = series,
            initial_train_size    = len(y[:-12]),
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            refit                 = False,
            interval              = None,
            alpha                 = None,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False,
            **integer_arguments
        )


@pytest.mark.parametrize("n_jobs", 
                         [1.0, 'not_int_auto'], 
                         ids = lambda value : f'n_jobs: {value}')
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
            forecaster            = forecaster,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            y                     = y,
            series                = None,
            initial_train_size    = len(y[:-12]),
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            refit                 = False,
            interval              = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            n_jobs                = n_jobs,
            verbose               = False,
            show_progress         = False
        )


@pytest.mark.parametrize("forecaster", 
                         [ForecasterAutoreg(regressor=Ridge(), lags=2),
                          ForecasterAutoregMultiSeries(regressor=Ridge(), lags=2)], 
                         ids = lambda fr : f'forecaster: {type(fr).__name__}' )
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
            forecaster            = forecaster,
            steps                 = steps,
            metric                = 'mean_absolute_error',
            y                     = y,
            series                = series,
            initial_train_size    = initial_train_size,
            fixed_train_size      = False,
            gap                   = gap,
            allow_incomplete_fold = False,
            refit                 = False,
            interval              = None,
            alpha                 = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
        )