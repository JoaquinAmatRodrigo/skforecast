# Unit test _evaluate_grid_hyperparameters_sarimax
# ==============================================================================
import re
import pytest
import platform
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from pmdarima.arima import ARIMA
from skforecast.Sarimax import Sarimax
from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.model_selection_sarimax.model_selection_sarimax import _evaluate_grid_hyperparameters_sarimax

from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True) # hide progress bar

# Fixtures
from ...ForecasterSarimax.tests.fixtures_ForecasterSarimax import y_datetime
from ...ForecasterSarimax.tests.fixtures_ForecasterSarimax import exog_datetime


def test_ValueError_evaluate_grid_hyperparameters_sarimax_when_return_best_and_len_y_exog_different():
    """
    Test ValueError is raised in _evaluate_grid_hyperparameters_sarimax when return_best 
    and length of `y` and `exog` do not match.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    exog_test = exog_datetime[:30].copy()

    err_msg = re.escape(
            (f'`exog` must have same number of samples as `y`. '
             f'length `exog`: ({len(exog_test)}), length `y`: ({len(y_datetime)})')
        )
    with pytest.raises(ValueError, match = err_msg):
        _evaluate_grid_hyperparameters_sarimax(
            forecaster         = forecaster,
            y                  = y_datetime,
            exog               = exog_test,
            param_grid         = [{'order': (1,1,1)}, {'order': (1,2,2)}, {'order': (1,2,3)}],
            steps              = 3,
            metric             = 'mean_absolute_error',
            initial_train_size = len(y_datetime)-12,
            fixed_train_size   = False,
            refit              = False,
            return_best        = True,
            verbose            = False
        )


def test_exception_evaluate_grid_hyperparameters_sarimax_metric_list_duplicate_names():
    """
    Test exception is raised in _evaluate_grid_hyperparameters_sarimax when a `list` of 
    metrics is used with duplicate names.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    err_msg = re.escape('When `metric` is a `list`, each metric name must be unique.')
    with pytest.raises(ValueError, match = err_msg):
        _evaluate_grid_hyperparameters_sarimax(
            forecaster         = forecaster,
            y                  = y_datetime,
            exog               = exog_datetime,
            param_grid         = [{'order': (1,1,1)}, {'order': (1,2,2)}, {'order': (1,2,3)}],
            steps              = 3,
            metric             = ['mean_absolute_error', mean_absolute_error],
            initial_train_size = len(y_datetime)-12,
            fixed_train_size   = False,
            refit              = False,
            return_best        = True,
            verbose            = False
        )


def test_output_evaluate_grid_hyperparameters_sarimax_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_sarimax in ForecasterSarimax with mocked
    (mocked done in Skforecast v0.7.0).
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    param_grid = [{'order': (3, 2 ,0), 'trend': None}, 
                  {'order': (3, 2, 0), 'trend': 'c'}]

    results = _evaluate_grid_hyperparameters_sarimax(
                  forecaster         = forecaster,
                  y                  = y_datetime,
                  param_grid         = param_grid,
                  steps              = 3,
                  refit              = False,
                  metric             = 'mean_squared_error',
                  initial_train_size = len(y_datetime)-12,
                  fixed_train_size   = False,
                  return_best        = False,
                  verbose            = False
              )
    
    expected_results = pd.DataFrame(
        data  = {'params'            : [{'order': (3, 2 ,0), 'trend': None}, 
                                        {'order': (3, 2, 0), 'trend': 'c'}],
                 'mean_squared_error': np.array([0.03683793, 0.03740798]),
                 'order'             : [(3, 2, 0), (3, 2, 0)],
                 'trend'             : [None, 'c']},
        index = pd.Index(np.array([0, 1]), dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results, atol=0.0001)


def test_output_evaluate_grid_hyperparameters_sarimax_exog_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_sarimax in ForecasterSarimax 
    with exog with mocked (mocked done in Skforecast v0.7.0).
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    param_grid = [{'order': (3, 2, 0), 'trend': None}, 
                  {'order': (3, 2, 0), 'trend': 'c'}]

    results = _evaluate_grid_hyperparameters_sarimax(
                  forecaster         = forecaster,
                  y                  = y_datetime,
                  exog               = exog_datetime,
                  param_grid         = param_grid,
                  steps              = 3,
                  refit              = False,
                  metric             = 'mean_squared_error',
                  initial_train_size = len(y_datetime)-12,
                  fixed_train_size   = False,
                  return_best        = False,
                  verbose            = False
              )
    
    expected_results = pd.DataFrame(
        data  = {'params'            : [{'order': (3, 2 ,0), 'trend': None}, 
                                        {'order': (3, 2, 0), 'trend': 'c'}],
                 'mean_squared_error': np.array([0.18551857, 0.19151678]),
                 'order'             : [(3, 2, 0), (3, 2, 0)],
                 'trend'             : [None, 'c']},
        index = pd.Index(np.array([0, 1]), dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results, atol=0.0001)


def test_output_evaluate_grid_hyperparameters_sarimax_metric_list_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_sarimax in ForecasterSarimax 
    with multiple metrics with mocked (mocked done in Skforecast v0.7.0).
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    param_grid = [{'order': (3, 2, 0), 'trend': None}, 
                  {'order': (3, 2, 0), 'trend': 'c'}]

    results = _evaluate_grid_hyperparameters_sarimax(
                  forecaster         = forecaster,
                  y                  = y_datetime,
                  param_grid         = param_grid,
                  steps              = 3,
                  refit              = True,
                  metric             = [mean_absolute_error, 'mean_squared_error'],
                  initial_train_size = len(y_datetime)-12,
                  fixed_train_size   = False,
                  return_best        = False,
                  verbose            = False
              )
    
    expected_results = pd.DataFrame(
        data  = {'params'             : [{'order': (3, 2 ,0), 'trend': None}, 
                                         {'order': (3, 2, 0), 'trend': 'c'}],
                 'mean_absolute_error': np.array([0.15724498, 0.16638452]),
                 'mean_squared_error' : np.array([0.0387042 , 0.04325543]),
                 'order'              : [(3, 2, 0), (3, 2, 0)],
                 'trend'              : [None, 'c']},
        index = pd.Index(np.array([0, 1]), dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results, atol=0.0001)
    

def test_evaluate_grid_hyperparameters_sarimax_when_return_best():
    """
    Test forecaster is refitted when return_best=True in 
    _evaluate_grid_hyperparameters_sarimax.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    param_grid = [{'order': (3, 2, 0), 'trend': None}, 
                  {'order': (3, 2, 0), 'trend': 'c'}]

    _evaluate_grid_hyperparameters_sarimax(
        forecaster            = forecaster,
        y                     = y_datetime,
        param_grid            = param_grid,
        steps                 = 3,
        refit                 = True,
        metric                = mean_absolute_error,
        initial_train_size    = len(y_datetime)-12,
        fixed_train_size      = True,
        return_best           = True,
        suppress_warnings_fit = False,
        verbose               = False
    )
    
    expected_params = {
        'concentrate_scale': False,
        'dates': None,
        'disp': False,
        'enforce_invertibility': True,
        'enforce_stationarity': True,
        'freq': None,
        'hamilton_representation': False,
        'maxiter': 1000,
        'measurement_error': False,
        'method': 'cg',
        'missing': 'none',
        'mle_regression': True,
        'order': (3, 2, 0),
        'seasonal_order': (0, 0, 0, 0),
        'simple_differencing': False,
        'sm_fit_kwargs': {},
        'sm_init_kwargs': {},
        'sm_predict_kwargs': {},
        'start_params': None,
        'time_varying_regression': False,
        'trend': None,
        'trend_offset': 1,
        'use_exact_diffuse': False,
        'validate_specification': True
    }
    
    assert expected_params == forecaster.params