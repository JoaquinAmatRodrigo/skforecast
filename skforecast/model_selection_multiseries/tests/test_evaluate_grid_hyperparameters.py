# Unit test _evaluate_grid_hyperparameters_multiseries
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.model_selection_multiseries.model_selection_multiseries import _evaluate_grid_hyperparameters_multiseries

from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True) # hide progress bar

# Fixtures
series = pd.DataFrame({'1': pd.Series(np.arange(20)), 
                       '2': pd.Series(np.arange(20))
                      })


def test_evaluate_grid_hyperparameters_multiseries_exception_when_levels_list_not_list():
    '''
    Test Exception is raised in _evaluate_grid_hyperparameters_multiseries when 
    levels_list is not a list.
    '''
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 3
                 )

    levels_list = 'not_a_list'
    
    with pytest.raises(Exception):
        _evaluate_grid_hyperparameters_multiseries(
            forecaster          = forecaster,
            series              = series,
            param_grid          = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            steps               = 4,
            metric              = 'mean_absolute_error',
            initial_train_size  = 12,
            fixed_train_size    = False,
            levels_list         = levels_list,
            levels_weights      = None,
            exog                = None,
            lags_grid           = [2, 4],
            refit               = False,
            return_best         = False,
            verbose             = False
        )


def test_evaluate_grid_hyperparameters_multiseries_exception_when_levels_weights_not_dict():
    '''
    Test Exception is raised in _evaluate_grid_hyperparameters_multiseries when 
    levels_weights is not a dict.
    '''
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 3
                 )

    levels_weights = 'not_a_dict'
    
    with pytest.raises(Exception):
        _evaluate_grid_hyperparameters_multiseries(
            forecaster          = forecaster,
            series              = series,
            param_grid          = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            steps               = 4,
            metric              = 'mean_absolute_error',
            initial_train_size  = 12,
            fixed_train_size    = False,
            levels_list         = None,
            levels_weights      = levels_weights,
            exog                = None,
            lags_grid           = [2, 4],
            refit               = False,
            return_best         = False,
            verbose             = False
        )


def test_evaluate_grid_hyperparameters_multiseries_exception_when_levels_list_and_levels_weights_not_match():
    '''
    Test Exception is raised in _evaluate_grid_hyperparameters_multiseries when 
    levels_weights are different from column names of series, `levels_list`.
    '''
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 3
                 )

    levels_list = ['1', '2']
    levels_weights = {'1': 0.5, 'not_2': 0.5}
    
    with pytest.raises(Exception):
        _evaluate_grid_hyperparameters_multiseries(
            forecaster          = forecaster,
            series              = series,
            param_grid          = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            steps               = 4,
            metric              = 'mean_absolute_error',
            initial_train_size  = 12,
            fixed_train_size    = False,
            levels_list         = levels_list,
            levels_weights      = levels_weights,
            exog                = None,
            lags_grid           = [2, 4],
            refit               = False,
            return_best         = False,
            verbose             = False
        )


def test_evaluate_grid_hyperparameters_multiseries_exception_when_levels_weights_not_sum_1():
    '''
    Test Exception is raised in _evaluate_grid_hyperparameters_multiseries when 
    levels_weights do not add up to 1.0.
    '''
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 3
                 )

    levels_weights = {'1': 0.5, '2': 0.6}
    
    with pytest.raises(Exception):
        _evaluate_grid_hyperparameters_multiseries(
            forecaster          = forecaster,
            series              = series,
            param_grid          = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            steps               = 4,
            metric              = 'mean_absolute_error',
            initial_train_size  = 12,
            fixed_train_size    = False,
            levels_list         = None,
            levels_weights      = levels_weights,
            exog                = None,
            lags_grid           = [2, 4],
            refit               = False,
            return_best         = False,
            verbose             = False
        )