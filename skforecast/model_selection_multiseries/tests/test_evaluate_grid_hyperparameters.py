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
    Test Exception is raised in _evaluate_grid_hyperparameters_multiseries when levels_list is not a list.
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
    Test Exception is raised in _evaluate_grid_hyperparameters_multiseries when levels_weights is not a dict.
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


def test_output_evaluate_grid_hyperparameters_ForecasterAutoreg_with_mocked():
    '''
    Test output of _evaluate_grid_hyperparameters in ForecasterAutoreg with mocked
    (mocked done in Skforecast v0.4.3)
    '''
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                    )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]
    idx = len(lags_grid)*len(param_grid)

    results = _evaluate_grid_hyperparameters(
                            forecaster  = forecaster,
                            y           = y,
                            lags_grid   = lags_grid,
                            param_grid  = param_grid,
                            steps       = steps,
                            refit       = False,
                            metric      = 'mean_squared_error',
                            initial_train_size = len(y_train),
                            fixed_train_size   = False,
                            return_best = False,
                            verbose     = False
                             )
    
    expected_results = pd.DataFrame({
            'lags'  :[[1, 2], [1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            'params':[{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, {'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            'metric':np.array([0.06464646, 0.06502362, 0.06745534, 0.06779272, 0.06802481, 0.06948609]),                                                               
            'alpha' :np.array([0.01, 0.1 , 1.  , 0.01, 0.1 , 1.  ])
                                     },
            index=np.arange(idx)
                                   )

    pd.testing.assert_frame_equal(results, expected_results)
    

def test_output_evaluate_grid_hyperparameters_ForecasterAutoreg_lags_grid_is_None_with_mocked():
    '''
    Test output of _evaluate_grid_hyperparameters in ForecasterAutoreg when lags_grid is None with mocked
    (mocked done in Skforecast v0.4.3), should use forecaster.lags as lags_grid.
    '''
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                    )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = None
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters(
                            forecaster  = forecaster,
                            y           = y,
                            lags_grid   = lags_grid,
                            param_grid  = param_grid,
                            steps       = steps,
                            refit       = False,
                            metric      = 'mean_squared_error',
                            initial_train_size = len(y_train),
                            fixed_train_size   = False,
                            return_best = False,
                            verbose     = False
                             )
    
    expected_results = pd.DataFrame({
            'lags'  :[[1, 2], [1, 2], [1, 2]],
            'params':[{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            'metric':np.array([0.06464646, 0.06502362, 0.06745534]),                                                               
            'alpha' :np.array([0.01, 0.1 , 1.])
                                     },
            index=[0, 1, 2]
                                   )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_ForecasterAutoregCustom_with_mocked():
    '''
    Test output of _evaluate_grid_hyperparameters in ForecasterAutoregCustom with mocked
    (mocked done in Skforecast v0.4.3)
    '''
    def create_predictors(y):
        '''
        Create first 4 lags of a time series, used in ForecasterAutoregCustom.
        '''

        lags = y[-1:-5:-1]

        return lags
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = Ridge(random_state=123),
                        fun_predictors = create_predictors,
                        window_size    = 4
                        )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]
    idx = len(param_grid)

    results = _evaluate_grid_hyperparameters(
                            forecaster  = forecaster,
                            y           = y,
                            param_grid  = param_grid,
                            steps       = steps,
                            refit       = False,
                            metric      = 'mean_squared_error',
                            initial_train_size = len(y_train),
                            fixed_train_size   = False,
                            return_best = False,
                            verbose     = False
                            )
    
    expected_results = pd.DataFrame({
            'lags'  :['custom predictors', 'custom predictors', 'custom predictors'],
            'params':[{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            'metric':np.array([0.06779272, 0.06802481, 0.06948609]),                                                               
            'alpha' :np.array([0.01, 0.1 , 1.])
                                     },
            index=np.arange(idx)
                                   )
    
    pd.testing.assert_frame_equal(results, expected_results)
    

def test_evaluate_grid_hyperparameters_when_return_best():
    '''
    Test forecaster is refited when return_best=True in _evaluate_grid_hyperparameters
    '''
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                    )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    _evaluate_grid_hyperparameters(
            forecaster  = forecaster,
            y           = y,
            lags_grid   = lags_grid,
            param_grid  = param_grid,
            steps       = steps,
            refit       = False,
            metric      = 'mean_squared_error',
            initial_train_size = len(y_train),
            fixed_train_size   = False,
            return_best = True,
            verbose     = False
            )
    
    expected_lags = np.array([1, 2])
    expected_alpha = 0.01
    
    assert (expected_lags == forecaster.lags).all()
    assert expected_alpha == forecaster.regressor.alpha