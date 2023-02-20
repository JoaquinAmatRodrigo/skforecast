# Unit test _evaluate_grid_hyperparameters_sarimax
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.model_selection_sarimax.model_selection_sarimax import _evaluate_grid_hyperparameters_sarimax
from pmdarima.arima import ARIMA

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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(1,1,1)))

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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(1,1,1)))

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
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(1,1,1)))

    param_grid = [{'order': (1,1,1), 'seasonal_order': (0,0,0,0)}, 
                  {'order': (1,2,3), 'seasonal_order': (2,2,2,4)}]

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
        data  = {'params'            : [{'order': (1,1,1), 'seasonal_order': (0,0,0,0)}, 
                                        {'order': (1,2,3), 'seasonal_order': (2,2,2,4)}],
                    'mean_squared_error': np.array([0.07440938870701645, 0.33300231290569804]),
                    'order'             : [(1, 1, 1), (1, 2, 3)],
                    'seasonal_order'    : [(0, 0, 0, 0), (2, 2, 2, 4)]},
        index = np.array([0, 1])
    )

    pd.testing.assert_frame_equal(results, expected_results, atol=0.0015)


def test_output_evaluate_grid_hyperparameters_sarimax_exog_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_sarimax in ForecasterSarimax 
    with exog with mocked (mocked done in Skforecast v0.7.0).
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(1,1,1)))

    param_grid = [{'order': (1,1,1), 'with_intercept': False}, 
                  {'order': (1,1,1), 'with_intercept': True}, 
                  {'order': (2,2,2), 'with_intercept': False}, 
                  {'order': (2,2,2), 'with_intercept': True}]

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
        data  = {'params'            : [{'order': (1,1,1), 'with_intercept': False}, 
                                        {'order': (1,1,1), 'with_intercept': True}, 
                                        {'order': (2,2,2), 'with_intercept': False}, 
                                        {'order': (2,2,2), 'with_intercept': True}],
                 'mean_squared_error': np.array([0.06404248, 0.06874972, 0.08803988, 0.08971294]),
                 'order'             : [(1, 1, 1), (1, 1, 1), (2, 2, 2), (2, 2, 2)],
                 'with_intercept'    : [False, True, False, True]},
        index = np.array([0, 1, 2, 3])
    )

    pd.testing.assert_frame_equal(results, expected_results, atol=0.001)


def test_output_evaluate_grid_hyperparameters_sarimax_metric_list_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_sarimax in ForecasterSarimax 
    with multiple metrics with mocked (mocked done in Skforecast v0.7.0).
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(1,1,1)))

    param_grid = [{'order': (1,1,1), 'method': 'lbfgs'}, 
                  {'order': (2,2,2), 'method': 'powell'}]

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
        data  = {'params'             : [{'order': (1,1,1), 'method': 'lbfgs'}, 
                                        {'order': (2,2,2), 'method': 'powell'}],
                    'mean_absolute_error': np.array([0.2249986568362591, 0.27299724595171604]),
                    'mean_squared_error' : np.array([0.07785902055180988, 0.1219896314815332]),
                    'order'              : [(1, 1, 1), (2, 2, 2)],
                    'method'             : ['lbfgs', 'powell']},
        index = np.array([0, 1])
    )

    pd.testing.assert_frame_equal(results, expected_results, atol=0.015)
    

def test_evaluate_grid_hyperparameters_sarimax_when_return_best():
    """
    Test forecaster is refitted when return_best=True in 
    _evaluate_grid_hyperparameters_sarimax.
    """
    forecaster = ForecasterSarimax(regressor=ARIMA(maxiter=1000, order=(3,3,3)))

    param_grid = [{'order': (1,1,1), 'method': 'lbfgs'}, 
                  {'order': (1,1,1), 'method': 'powell'}, 
                  {'order': (2,2,2), 'method': 'lbfgs'}, 
                  {'order': (2,2,2), 'method': 'powell'}]

    _evaluate_grid_hyperparameters_sarimax(
        forecaster         = forecaster,
        y                  = y_datetime,
        param_grid         = param_grid,
        steps              = 3,
        refit              = True,
        metric             = mean_absolute_error,
        initial_train_size = len(y_datetime)-12,
        fixed_train_size   = True,
        return_best        = True,
        verbose            = False
    )
    
    expected_params = {'maxiter'           : 1000,
                       'method'            : 'powell',
                       'order'             : (1, 1, 1),
                       'out_of_sample_size': 0,
                       'scoring'           : 'mse',
                       'scoring_args'      : None,
                       'seasonal_order'    : (0, 0, 0, 0),
                       'start_params'      : None,
                       'suppress_warnings' : False,
                       'trend'             : None,
                       'with_intercept'    : True}
    
    assert expected_params == forecaster.params
    assert expected_params['method'] == forecaster.regressor.method
    assert expected_params['order'] == forecaster.regressor.order