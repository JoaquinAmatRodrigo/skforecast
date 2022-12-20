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


@pytest.mark.parametrize("initial_train_size", 
                         [20., 21.2, 'not_int'], 
                         ids = lambda value : f'initial_train_size: {value}' )
def test_backtesting_forecaster_exception_when_initial_train_size_is_not_an_int_or_None(initial_train_size):
    """
    Test Exception is raised in backtesting_forecaster when 
    initial_train_size is not an integer or None.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )
    
    err_msg = re.escape(
            (f'If used, `initial_train_size` must be an integer greater than '
             f'the window_size of the forecaster ({forecaster.window_size}).')
        )
    with pytest.raises(ValueError, match = err_msg):
        backtesting_forecaster(
            forecaster          = forecaster,
            y                   = y,
            steps               = 3,
            metric              = 'mean_absolute_error',
            initial_train_size  = initial_train_size,
            refit               = False,
            fixed_train_size    = False,
            exog                = None,
            interval            = None,
            n_boot              = 500,
            random_state        = 123,
            in_sample_residuals = True,
            verbose             = False
        )


@pytest.mark.parametrize("initial_train_size", 
                         [(len(y)), (len(y) + 1)], 
                         ids = lambda value : f'len: {value}' )
def test_backtesting_forecaster_exception_when_initial_train_size_more_than_or_equal_to_len_y(initial_train_size):
    """
    Test Exception is raised in backtesting_forecaster when 
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
            forecaster          = forecaster,
            y                   = y,
            steps               = 3,
            metric              = 'mean_absolute_error',
            initial_train_size  = initial_train_size,
            refit               = False,
            fixed_train_size    = False,
            exog                = None,
            interval            = None,
            n_boot              = 500,
            random_state        = 123,
            in_sample_residuals = True,
            verbose             = False
        )


def test_backtesting_forecaster_exception_when_initial_train_size_less_than_forecaster_window_size():
    """
    Test Exception is raised in backtesting_forecaster when 
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
            forecaster          = forecaster,
            y                   = y,
            steps               = 3,
            metric              = 'mean_absolute_error',
            initial_train_size  = initial_train_size,
            refit               = False,
            fixed_train_size    = False,
            exog                = None,
            interval            = None,
            n_boot              = 500,
            random_state        = 123,
            in_sample_residuals = True,
            verbose             = False
        )


def test_backtesting_forecaster_exception_when_initial_train_size_None_and_forecaster_not_fitted():
    """
    Test Exception is raised in backtesting_forecaster when initial_train_size is None and
    forecaster is not fitted.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    initial_train_size = None
    
    err_msg = re.escape('`forecaster` must be already trained if no `initial_train_size` is provided.')
    with pytest.raises(NotFittedError, match = err_msg):
        backtesting_forecaster(
            forecaster          = forecaster,
            y                   = y,
            steps               = 3,
            metric              = 'mean_absolute_error',
            initial_train_size  = initial_train_size,
            refit               = False,
            fixed_train_size    = False,
            exog                = None,
            interval            = None,
            n_boot              = 500,
            random_state        = 123,
            in_sample_residuals = True,
            verbose             = False
        )


def test_backtesting_forecaster_exception_when_refit_not_bool():
    """
    Test Exception is raised in backtesting_forecaster when refit is not bool.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    refit = 'not_bool'
    
    err_msg = re.escape( f'`refit` must be boolean: `True`, `False`.')
    with pytest.raises(TypeError, match = err_msg):
        backtesting_forecaster(
            forecaster          = forecaster,
            y                   = y,
            steps               = 3,
            metric              = 'mean_absolute_error',
            initial_train_size  = len(y[:-12]),
            refit               = refit,
            fixed_train_size    = False,
            exog                = None,
            interval            = None,
            n_boot              = 500,
            random_state        = 123,
            in_sample_residuals = True,
            verbose             = False
        )


def test_backtesting_forecaster_exception_when_initial_train_size_None_and_refit_True():
    """
    Test Exception is raised in backtesting_forecaster when initial_train_size is None
    and refit is True.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )
    forecaster.fitted = True

    initial_train_size = None
    refit = True
    
    err_msg = re.escape(f'`refit` is only allowed when `initial_train_size` is not `None`.')
    with pytest.raises(ValueError, match = err_msg):
        backtesting_forecaster(
            forecaster          = forecaster,
            y                   = y,
            steps               = 3,
            metric              = 'mean_absolute_error',
            initial_train_size  = initial_train_size,
            refit               = refit,
            fixed_train_size    = False,
            exog                = None,
            interval            = None,
            n_boot              = 500,
            random_state        = 123,
            in_sample_residuals = True,
            verbose             = False
        )


def test_backtesting_forecaster_exception_when_interval_not_None_and_ForecasterAutoregDirect():
    """
    Test Exception is raised in backtesting_forecaster when interval is not None 
    and forecaster is a ForecasterAutoregDirect.
    """
    forecaster = ForecasterAutoregDirect(
                    regressor = Ridge(random_state=123),
                    steps     = 3,
                    lags      = 2
                 )

    interval = [10, 90]
    
    err_msg = re.escape(
            ('Interval prediction is only available when forecaster is of type '
             'ForecasterAutoreg or ForecasterAutoregCustom.')
        )
    with pytest.raises(TypeError, match = err_msg):
        backtesting_forecaster(
            forecaster          = forecaster,
            y                   = y,
            steps               = 3,
            metric              = 'mean_absolute_error',
            initial_train_size  = len(y[:-12]),
            refit               = False,
            fixed_train_size    = False,
            exog                = None,
            interval            = interval,
            n_boot              = 500,
            random_state        = 123,
            in_sample_residuals = True,
            verbose             = False
        )


def test_backtesting_forecaster_exception_when_forecaster_ForecasterAutoregMultiSeries():
    """
    Test Exception is raised in backtesting_forecaster when forecaster is of type
    ForecasterAutoregMultiSeries.
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    err_msg = re.escape(
            ('`forecaster` must be of type `ForecasterAutoreg`, `ForecasterAutoregCustom` '
             'or `ForecasterAutoregDirect`, for all other types of forecasters '
             'use the functions available in the `model_selection` module.')
        )
    with pytest.raises(TypeError, match = err_msg):
        backtesting_forecaster(
            forecaster          = forecaster,
            y                   = y,
            steps               = 3,
            metric              = 'mean_absolute_error',
            initial_train_size  = len(y[:-12]),
            refit               = False,
            fixed_train_size    = False,
            exog                = None,
            interval            = None,
            n_boot              = 500,
            random_state        = 123,
            in_sample_residuals = True,
            verbose             = False
        )