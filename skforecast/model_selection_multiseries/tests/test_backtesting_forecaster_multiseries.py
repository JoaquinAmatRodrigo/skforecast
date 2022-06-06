# Unit test backtesting_forecaster_multiseries
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.model_selection_multiseries import backtesting_forecaster_multiseries

# Fixtures
series = pd.DataFrame({'1': pd.Series(np.arange(20)), 
                       '2': pd.Series(np.arange(20))
                      })


def test_backtesting_forecaster_multiseries_exception_when_initial_train_size_more_than_len_y():
    '''
    Test Exception is raised in backtesting_forecaster_multiseries when initial_train_size > len(series).
    '''
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 3
                 )

    initial_train_size = len(series) + 1
    
    with pytest.raises(Exception):
        backtesting_forecaster_multiseries(
            forecaster          = forecaster,
            series              = series,
            steps               = 4,
            level               = '1',
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


def test_backtesting_forecaster_multiseries_exception_when_initial_train_size_less_than_forecaster_window_size():
    '''
    Test Exception is raised in backtesting_forecaster_multiseries when initial_train_size < forecaster.window_size.
    '''
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 3
                 )

    initial_train_size = forecaster.window_size - 1
    
    with pytest.raises(Exception):
        backtesting_forecaster_multiseries(
            forecaster          = forecaster,
            series              = series,
            steps               = 4,
            level               = '1',
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


def test_backtesting_forecaster_multiseries_exception_when_initial_train_size_None_and_forecaster_not_fitted():
    '''
    Test Exception is raised in backtesting_forecaster_multiseries when initial_train_size is None and
    forecaster is not fitted.
    '''
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    initial_train_size = None
    
    with pytest.raises(Exception):
        backtesting_forecaster_multiseries(
            forecaster          = forecaster,
            series              = series,
            steps               = 4,
            level               = '1',
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


def test_backtesting_forecaster_multiseries_exception_when_refit_not_bool():
    '''
    Test Exception is raised in backtesting_forecaster_multiseries when refit is not bool.
    '''
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    refit = 'not_bool'
    
    with pytest.raises(Exception):
        backtesting_forecaster_multiseries(
            forecaster          = forecaster,
            series              = series,
            steps               = 4,
            level               = '1',
            metric              = 'mean_absolute_error',
            initial_train_size  = 12,
            refit               = refit,
            fixed_train_size    = False,
            exog                = None,
            interval            = None,
            n_boot              = 500,
            random_state        = 123,
            in_sample_residuals = True,
            verbose             = False
        )


def test_backtesting_forecaster_multiseries_exception_when_initial_train_size_None_and_refit_True():
    '''
    Test Exception is raised in backtesting_forecaster_multiseries when initial_train_size is None
    and refit is True.
    '''
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )
    forecaster.fitted = True

    initial_train_size = None
    refit = True
    
    with pytest.raises(Exception):
        backtesting_forecaster_multiseries(
            forecaster          = forecaster,
            series              = series,
            steps               = 4,
            level               = '1',
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