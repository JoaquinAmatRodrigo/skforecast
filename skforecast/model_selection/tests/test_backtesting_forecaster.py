# Unit test backtesting_forecaster
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
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
              0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453]))


def test_backtesting_forecaster_exception_when_initial_train_size_more_than_len_y():
    '''
    Test Exception is raised in backtesting_forecaster when initial_train_size > len(y).
    '''
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    initial_train_size = len(y) + 1
    
    with pytest.raises(Exception):
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
    '''
    Test Exception is raised in backtesting_forecaster when initial_train_size < forecaster.window_size.
    '''
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    initial_train_size = forecaster.window_size - 1
    
    with pytest.raises(Exception):
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
    '''
    Test Exception is raised in backtesting_forecaster when initial_train_size is None and
    forecaster is not fitted.
    '''
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    initial_train_size = None
    
    with pytest.raises(Exception):
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
    '''
    Test Exception is raised in backtesting_forecaster when refit is not bool.
    '''
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    refit = 'not_bool'
    
    with pytest.raises(Exception):
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
    '''
    Test Exception is raised in backtesting_forecaster when initial_train_size is None
    and refit is True.
    '''
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    initial_train_size = None
    refit = True
    
    with pytest.raises(Exception):
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


def test_backtesting_forecaster_exception_when_interval_not_None_and_ForecasterAutoregMultiOutput():
    '''
    Test Exception is raised in backtesting_forecaster when interval is not None and
    forecaster is a ForecasterAutoregMultiOutput.
    '''
    forecaster = ForecasterAutoregMultiOutput(
                    regressor = Ridge(random_state=123),
                    steps     = 3,
                    lags      = 2
                 )

    interval = [10, 90]
    
    with pytest.raises(Exception):
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