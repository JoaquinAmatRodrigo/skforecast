# Unit test backtesting_forecaster
# ==============================================================================
import re
from typing import Type
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.exceptions import NotFittedError
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.model_selection import backtesting_forecaster

# Fixtures _backtesting_forecaster_refit Series (skforecast==0.4.2)
# np.random.seed(123)
# y = np.random.rand(50)

y = pd.Series(
        np.array([0.69646919, 0.28613933, 0.22685145, 0.55131477, 0.71946897,
                  0.42310646, 0.9807642 , 0.68482974, 0.4809319 , 0.39211752,
                  0.34317802, 0.72904971, 0.43857224, 0.0596779 , 0.39804426,
                  0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759,
                  0.63440096, 0.84943179, 0.72445532, 0.61102351, 0.72244338,
                  0.32295891, 0.36178866, 0.22826323, 0.29371405, 0.63097612,
                  0.09210494, 0.43370117, 0.43086276, 0.4936851 , 0.42583029,
                  0.31226122, 0.42635131, 0.89338916, 0.94416002, 0.50183668,
                  0.62395295, 0.1156184 , 0.31728548, 0.41482621, 0.86630916,
                  0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453]
        )
    )


def test_backtesting_forecaster_TypeError_when_forecaster_not_supported_types():
    """
    Test TypeError is raised in backtesting_forecaster if Forecaster is not one 
    of the types supported by the function.
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    err_msg = re.escape(
            ("`forecaster` must be of type `ForecasterAutoreg`, `ForecasterAutoregCustom` "
             "or `ForecasterAutoregDirect`, for all other types of forecasters "
             "use the functions available in the other `model_selection` modules.")
        )
    with pytest.raises(TypeError, match = err_msg):
        backtesting_forecaster(
            forecaster            = forecaster,
            y                     = y,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            initial_train_size    = len(y[:-12]),
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            exog                  = None,
            refit                 = False,
            interval              = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
        )


def test_backtesting_forecaster_TypeError_when_y_is_not_pandas_Series():
    """
    Test TypeError is raised in backtesting_forecaster if `y` is not a 
    pandas Series.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )
    bad_y = np.arange(50)

    err_msg = re.escape("`y` must be a pandas Series.")
    with pytest.raises(TypeError, match = err_msg):
        backtesting_forecaster(
            forecaster            = forecaster,
            y                     = bad_y,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            initial_train_size    = len(y[:-12]),
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            exog                  = None,
            refit                 = False,
            interval              = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
        )


@pytest.mark.parametrize("steps", 
                         ['not_int', 2.3, 0], 
                         ids = lambda value : f'steps: {value}' )
def test_backtesting_forecaster_TypeError_when_steps_not_int_greater_or_equal_1(steps):
    """
    Test TypeError is raised in backtesting_forecaster if `steps` is not an 
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
        backtesting_forecaster(
            forecaster            = forecaster,
            y                     = y,
            steps                 = steps,
            metric                = 'mean_absolute_error',
            initial_train_size    = len(y[:-12]),
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            exog                  = None,
            refit                 = False,
            interval              = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
        )


def test_backtesting_forecaster_TypeError_when_metric_not_correct_type():
    """
    Test TypeError is raised in backtesting_forecaster if `metric` is not string, 
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
        backtesting_forecaster(
            forecaster            = forecaster,
            y                     = y,
            steps                 = 5,
            metric                = metric,
            initial_train_size    = len(y[:-12]),
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            exog                  = None,
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
def test_backtesting_forecaster_TypeError_when_initial_train_size_is_not_an_int_or_None(initial_train_size):
    """
    Test TypeError is raised in backtesting_forecaster when 
    initial_train_size is not an integer.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )
    
    err_msg = re.escape(
            (f'If used, `initial_train_size` must be an integer greater than '
             f'the window_size of the forecaster. Got {type(initial_train_size)}.')
        )
    with pytest.raises(TypeError, match = err_msg):
        backtesting_forecaster(
            forecaster            = forecaster,
            y                     = y,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            initial_train_size    = initial_train_size,
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            exog                  = None,
            refit                 = False,
            interval              = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
        )


@pytest.mark.parametrize("initial_train_size", 
                         [(len(y)), (len(y) + 1)], 
                         ids = lambda value : f'len: {value}' )
def test_backtesting_forecaster_ValueError_when_initial_train_size_more_than_or_equal_to_len_y(initial_train_size):
    """
    Test ValueError is raised in backtesting_forecaster when 
    initial_train_size >= len(y).
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )
    
    err_msg = re.escape(
            (f'If used, `initial_train_size` must be an integer '
             f'smaller than the length of `y` ({len(y)}).')
        )
    with pytest.raises(ValueError, match = err_msg):
        backtesting_forecaster(
            forecaster            = forecaster,
            y                     = y,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            initial_train_size    = initial_train_size,
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            exog                  = None,
            refit                 = False,
            interval              = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
        )


def test_backtesting_forecaster_ValueError_when_initial_train_size_less_than_forecaster_window_size():
    """
    Test ValueError is raised in backtesting_forecaster when 
    initial_train_size < forecaster.window_size.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    initial_train_size = forecaster.window_size - 1
    
    err_msg = re.escape(
            (f'If used, `initial_train_size` must be an integer greater than '
             f'the window_size of the forecaster ({forecaster.window_size}).')
        )
    with pytest.raises(ValueError, match = err_msg):
        backtesting_forecaster(
            forecaster            = forecaster,
            y                     = y,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            initial_train_size    = initial_train_size,
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            exog                  = None,
            refit                 = False,
            interval              = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
        )


def test_backtesting_forecaster_ValueError_when_initial_train_size_plus_gap_less_than_y_size():
    """
    Test ValueError is raised in backtesting_forecaster when 
    initial_train_size < forecaster.window_size.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    initial_train_size = len(y) - 1
    gap = 2
    
    err_msg = re.escape(
                (f"The combination of initial_train_size {initial_train_size} and gap "
                 f"{gap} cannot be greater than the length of y ({len(y)}).")
        )
    with pytest.raises(ValueError, match = err_msg):
        backtesting_forecaster(
            forecaster            = forecaster,
            y                     = y,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            initial_train_size    = initial_train_size,
            fixed_train_size      = False,
            gap                   = gap,
            allow_incomplete_fold = False,
            exog                  = None,
            refit                 = False,
            interval              = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
        )


def test_backtesting_forecaster_NotFittedError_when_initial_train_size_None_and_forecaster_not_fitted():
    """
    Test NotFittedError is raised in backtesting_forecaster when initial_train_size 
    is None and forecaster is not fitted.
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
        backtesting_forecaster(
            forecaster            = forecaster,
            y                     = y,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            initial_train_size    = initial_train_size,
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            exog                  = None,
            refit                 = False,
            interval              = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
        )


def test_backtesting_forecaster_ValueError_when_initial_train_size_None_and_refit_True():
    """
    Test ValueError is raised in backtesting_forecaster when initial_train_size is None
    and refit is True.
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
        backtesting_forecaster(
            forecaster            = forecaster,
            y                     = y,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            initial_train_size    = initial_train_size,
            fixed_train_size      = False,
            gap                   = 0,
            allow_incomplete_fold = False,
            exog                  = None,
            refit                 = refit,
            interval              = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
        )


@pytest.mark.parametrize("boolean_argument", 
                         ['fixed_train_size', 'allow_incomplete_fold', 'refit',
                          'in_sample_residuals', 'verbose', 'show_progress'], 
                         ids = lambda argument : f'{argument}' )
def test_backtesting_forecaster_TypeError_when_boolean_arguments_not_bool(boolean_argument):
    """
    Test TypeError is raised in backtesting_forecaster when boolean arguments 
    are not boolean.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )
    
    boolean_arguments = {'fixed_train_size': False,
                         'allow_incomplete_fold': False,
                         'refit': False,
                         'in_sample_residuals': False,
                         'verbose': False,
                         'show_progress': False,}
    boolean_arguments[boolean_argument] = 'not_bool'
    
    err_msg = re.escape(f"`{boolean_argument}` must be a boolean: `True`, `False`.")
    with pytest.raises(TypeError, match = err_msg):
        backtesting_forecaster(
            forecaster            = forecaster,
            y                     = y,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            initial_train_size    = len(y[:-12]),
            gap                   = 0,
            exog                  = None,
            interval              = None,
            n_boot                = 500,
            random_state          = 123,
            **boolean_arguments
        )


@pytest.mark.parametrize("int_argument, value",
                         [('gap', 'not_int'),
                          ('n_boot', 2.2), 
                          ('n_boot', -2),
                          ('random_state', 'not_int'),  
                          ('random_state', -3)], 
                         ids = lambda argument : f'{argument}' )
def test_backtesting_forecaster_TypeError_when_integer_arguments_not_int_or_greater_than_0(int_argument, value):
    """
    Test TypeError is raised in backtesting_forecaster when integer arguments 
    are not int or are greater than 0.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )
    
    integer_arguments = {'gap': 0,
                         'n_boot': 500,
                         'random_state': 123}
    integer_arguments[int_argument] = value
    
    err_msg = re.escape(f"`{int_argument}` must be an integer greater than 0. Got {value}.")
    with pytest.raises(TypeError, match = err_msg):
        backtesting_forecaster(
            forecaster            = forecaster,
            y                     = y,
            steps                 = 3,
            metric                = 'mean_absolute_error',
            initial_train_size    = len(y[:-12]),
            fixed_train_size      = False,
            allow_incomplete_fold = False,
            exog                  = None,
            refit                 = False,
            interval              = None,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False,
            **integer_arguments
        )


def test_backtesting_forecaster_ValueError_when_not_enough_data_to_create_a_fold():
    """
    Test ValueError is raised in backtesting_forecaster when there is not enough 
    data to evaluate even single fold because `allow_incomplete_fold` = `False`.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )
    
    initial_train_size = len(y[:-12])
    gap = 10
    steps = 5
    
    err_msg = re.escape(
            (f"There is not enough data to evaluate {steps} steps in a single "
             f"fold. Set `allow_incomplete_fold` to `True` to allow incomplete folds.\n"
             f"    Data available for test : {len(y) - (initial_train_size + gap)} \n"
             f"    Steps                   : {steps}\n")
        )
    
    with pytest.raises(ValueError, match = err_msg):
        backtesting_forecaster(
            forecaster            = forecaster,
            y                     = y,
            steps                 = steps,
            metric                = 'mean_absolute_error',
            initial_train_size    = initial_train_size,
            fixed_train_size      = False,
            gap                   = gap,
            allow_incomplete_fold = False,
            exog                  = None,
            refit                 = False,
            interval              = None,
            n_boot                = 500,
            random_state          = 123,
            in_sample_residuals   = True,
            verbose               = False,
            show_progress         = False
        )