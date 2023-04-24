# Unit test random_search_forecaster_multiseries
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.ForecasterAutoregMultiSeriesCustom import ForecasterAutoregMultiSeriesCustom
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from skforecast.model_selection_multiseries import random_search_forecaster_multiseries
from skforecast.model_selection_multiseries import random_search_forecaster_multivariate

from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True) # hide progress bar

# Fixtures
from .fixtures_model_selection_multiseries import series

def create_predictors(y): # pragma: no cover
    """
    Create first 4 lags of a time series.
    """
    lags = y[-1:-5:-1]

    return lags


def test_output_random_search_forecaster_multiseries_ForecasterAutoregMultiSeries_with_mocked():
    """
    Test output of random_search_forecaster_multiseries in ForecasterAutoregMultiSeries 
    with mocked (mocked done in Skforecast v0.5.0)
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    lags_grid = [2, 4]
    param_distributions = {'alpha':np.logspace(-5, 3, 10)}
    n_iter = 3

    results = random_search_forecaster_multiseries(
                    forecaster          = forecaster,
                    series              = series,
                    param_distributions = param_distributions,
                    steps               = steps,
                    metric              = 'mean_absolute_error',
                    initial_train_size  = len(series) - n_validation,
                    fixed_train_size    = False,
                    levels              = None,
                    exog                = None,
                    lags_grid           = lags_grid,
                    refit               = False,
                    n_iter              = n_iter,
                    return_best         = False,
                    verbose             = False
              )
    
    expected_results = pd.DataFrame({
            'levels':[['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2']],
            'lags'  :[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2], [1, 2], [1, 2]],
            'params':[{'alpha': 1e-05}, {'alpha': 0.03593813663804626}, {'alpha': 16.681005372000556}, 
                      {'alpha': 16.681005372000556}, {'alpha': 0.03593813663804626}, {'alpha': 1e-05}],
            'mean_absolute_error':np.array([0.20967967565103562, 0.20968441516920436, 0.20988932397621246,
                                            0.2104645379335131, 0.2107874324738886, 0.2107879393001434]),                                                               
            'alpha' :np.array([1e-05, 0.03593813663804626, 16.681005372000556, 
                               16.681005372000556, 0.03593813663804626, 1e-05])
                                     },
            index=pd.Index([4, 3, 5, 2, 0, 1], dtype='int64')
                                   )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_random_search_forecaster_multiseries_ForecasterAutoregMultiSeriesCustom_with_mocked():
    """
    Test output of random_search_forecaster_multiseries in ForecasterAutoregMultiSeriesCustom 
    with mocked (mocked done in Skforecast v0.5.0)
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = Ridge(random_state=123),
                     fun_predictors = create_predictors,
                     window_size    = 4
                 )

    steps = 3
    n_validation = 12
    param_distributions = {'alpha':np.logspace(-5, 3, 10)}
    n_iter = 3

    results = random_search_forecaster_multiseries(
                    forecaster          = forecaster,
                    series              = series,
                    param_distributions = param_distributions,
                    steps               = steps,
                    metric              = 'mean_absolute_error',
                    initial_train_size  = len(series) - n_validation,
                    fixed_train_size    = False,
                    levels              = None,
                    exog                = None,
                    lags_grid           = None,
                    refit               = False,
                    n_iter              = n_iter,
                    return_best         = False,
                    verbose             = False
              )
    
    expected_results = pd.DataFrame({
            'levels':[['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2']],
            'lags'  :['custom predictors', 'custom predictors', 'custom predictors'],
            'params':[{'alpha': 1e-05}, {'alpha': 0.03593813663804626}, {'alpha': 16.681005372000556}],
            'mean_absolute_error':np.array([0.20967967565103562, 0.20968441516920436, 0.20988932397621246]),                                                               
            'alpha' :np.array([1e-05, 0.03593813663804626, 16.681005372000556])
                                     },
            index=pd.Index([1, 0, 2], dtype='int64')
                                   )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_random_search_forecaster_multiseries_ForecasterAutoregMultiVariate_with_mocked():
    """
    Test output of random_search_forecaster_multiseries in ForecasterAutoregMultiVariate 
    with mocked (mocked done in Skforecast v0.6.0)
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor = Ridge(random_state=123),
                     level     = 'l1',
                     lags      = 2,
                     steps     = 3
                 )

    steps = 3
    n_validation = 12
    lags_grid = [2, 4]
    param_distributions = {'alpha':np.logspace(-5, 3, 10)}
    n_iter = 3

    results = random_search_forecaster_multivariate(
                    forecaster          = forecaster,
                    series              = series,
                    param_distributions = param_distributions,
                    steps               = steps,
                    metric              = 'mean_absolute_error',
                    initial_train_size  = len(series) - n_validation,
                    fixed_train_size    = False,
                    levels              = None,
                    exog                = None,
                    lags_grid           = lags_grid,
                    refit               = False,
                    n_iter              = n_iter,
                    return_best         = False,
                    verbose             = True
              )
    
    expected_results = pd.DataFrame({
            'levels': [['l1'], ['l1'], ['l1'], ['l1'], ['l1'], ['l1']],
            'lags'  : [[1, 2], [1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            'params': [{'alpha': 1e-05}, {'alpha': 0.03593813663804626}, {'alpha': 16.681005372000556},
                       {'alpha': 16.681005372000556}, {'alpha': 0.03593813663804626}, {'alpha': 1e-05}],
            'mean_absolute_error': np.array([0.20107097, 0.20135665, 0.20991177, 0.21121154, 0.22640335, 0.22645387]),                                                               
            'alpha' : np.array([1.00000000e-05, 3.59381366e-02, 1.66810054e+01, 1.66810054e+01, 3.59381366e-02, 1.00000000e-05])
                                     },
            index=pd.Index([1, 0, 2, 5, 3, 4], dtype='int64')
                                   )

    pd.testing.assert_frame_equal(results, expected_results)