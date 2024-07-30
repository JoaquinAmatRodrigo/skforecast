# Unit test grid_search_forecaster_multiseries
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.ForecasterAutoregMultiSeriesCustom import ForecasterAutoregMultiSeriesCustom
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from skforecast.model_selection_multiseries import grid_search_forecaster_multiseries
from skforecast.model_selection_multiseries import grid_search_forecaster_multivariate

from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # hide progress bar

# Fixtures
from .fixtures_model_selection_multiseries import series

def create_predictors(y):  # pragma: no cover
    """
    Create first 2 lags of a time series.
    """
    lags = y[-1:-3:-1]

    return lags


def test_output_grid_search_forecaster_multiseries_ForecasterAutoregMultiSeries_with_mocked():
    """
    Test output of grid_search_forecaster_multiseries in ForecasterAutoregMultiSeries 
    with mocked (mocked done in Skforecast v0.5.0)
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = Ridge(random_state=123),
                     lags               = 2, 
                     encoding           = 'onehot',
                     transformer_series = None
                 )

    steps = 3
    n_validation = 12
    lags_grid = [2, 4]
    param_grid = {'alpha': [0.01, 0.1, 1]}

    results = grid_search_forecaster_multiseries(
                  forecaster          = forecaster,
                  series              = series,
                  param_grid          = param_grid,
                  steps               = steps,
                  metric              = 'mean_absolute_error',
                  aggregate_metric    = 'weighted_average',
                  initial_train_size  = len(series) - n_validation,
                  fixed_train_size    = False,
                  levels              = None,
                  exog                = None,
                  lags_grid           = lags_grid,
                  refit               = False,
                  return_best         = False,
                  verbose             = True
              )
    
    expected_results = pd.DataFrame({
        'levels': [['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2']],
        'lags'  : [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2], [1, 2], [1, 2]],
        'lags_label': [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2], [1, 2], [1, 2]],
        'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, {'alpha': 1}, {'alpha': 0.1}, {'alpha': 0.01}],
        'mean_absolute_error__weighted_average':  np.array(
            [0.20968100463227382, 0.20969259779858337, 0.20977945312386406, 
             0.21077344827205086, 0.21078653113227208, 0.21078779824759553]
        ),                                                               
        'alpha' : np.array([0.01, 0.1, 1., 1., 0.1, 0.01])
        },
        index = pd.Index([0, 1, 2, 3, 4, 5], dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_grid_search_forecaster_multiseries_ForecasterAutoregMultiSeries_multiple_aggregated_metrics_with_mocked():
    """
    Test output of grid_search_forecaster_multiseries in ForecasterAutoregMultiSeries 
    with mocked (mocked done in Skforecast v0.5.0)
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = Ridge(random_state=123),
                     lags               = 2, 
                     encoding           = 'onehot',
                     transformer_series = None
                 )

    steps = 3
    n_validation = 12
    lags_grid = [2, 4]
    param_grid = {'alpha': [0.01, 0.1, 1]}

    results = grid_search_forecaster_multiseries(
                  forecaster          = forecaster,
                  series              = series,
                  param_grid          = param_grid,
                  steps               = steps,
                  metric              = ['mean_absolute_error', 'mean_absolute_scaled_error'],
                  aggregate_metric   = ['weighted_average', 'average', 'pooling'],
                  initial_train_size  = len(series) - n_validation,
                  fixed_train_size    = False,
                  levels              = None,
                  exog                = None,
                  lags_grid           = lags_grid,
                  refit               = False,
                  return_best         = False,
                  verbose             = False,
                  show_progress       = False,
              )
    
    expected_results = pd.DataFrame({
        'levels': {0: ['l1', 'l2'],
        1: ['l1', 'l2'],
        2: ['l1', 'l2'],
        3: ['l1', 'l2'],
        4: ['l1', 'l2'],
        5: ['l1', 'l2']},
        'lags': {0: np.array([1, 2, 3, 4]),
        1: np.array([1, 2, 3, 4]),
        2: np.array([1, 2, 3, 4]),
        3: np.array([1, 2]),
        4: np.array([1, 2]),
        5: np.array([1, 2])},
        'lags_label': {0: np.array([1, 2, 3, 4]),
        1: np.array([1, 2, 3, 4]),
        2: np.array([1, 2, 3, 4]),
        3: np.array([1, 2]),
        4: np.array([1, 2]),
        5: np.array([1, 2])},
        'params': {0: {'alpha': 0.01},
        1: {'alpha': 0.1},
        2: {'alpha': 1},
        3: {'alpha': 1},
        4: {'alpha': 0.1},
        5: {'alpha': 0.01}},
        'mean_absolute_error__weighted_average': {0: 0.20968100547390048,
        1: 0.20969259864077977,
        2: 0.20977945397058564,
        3: 0.21077344921320568,
        4: 0.21078653208835063,
        5: 0.21078779920557153},
        'mean_absolute_error__average': {0: 0.20968100547390048,
        1: 0.20969259864077974,
        2: 0.20977945397058564,
        3: 0.21077344921320565,
        4: 0.21078653208835063,
        5: 0.21078779920557153},
        'mean_absolute_error__pooling': {0: 0.20968100547390045,
        1: 0.2096925986407798,
        2: 0.20977945397058564,
        3: 0.21077344921320565,
        4: 0.21078653208835063,
        5: 0.21078779920557153},
        'mean_absolute_scaled_error__weighted_average': {0: 0.7969369551529275,
        1: 0.7969838748911608,
        2: 0.7973389652448446,
        3: 0.8009631048212882,
        4: 0.8009302953795885,
        5: 0.8009249124659391},
        'mean_absolute_scaled_error__average': {0: 0.7969369551529275,
        1: 0.7969838748911608,
        2: 0.7973389652448445,
        3: 0.8009631048212883,
        4: 0.8009302953795885,
        5: 0.8009249124659391},
        'mean_absolute_scaled_error__pooling': {0: 0.7809734688246502,
        1: 0.7810166484905049,
        2: 0.7813401480275807,
        3: 0.7850423618302551,
        4: 0.785091090032226,
        5: 0.7850958095104122},
        'alpha': {0: 0.01, 1: 0.1, 2: 1.0, 3: 1.0, 4: 0.1, 5: 0.01}
    })

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_grid_search_forecaster_multiseries_ForecasterAutoregMultiSeriesCustom_with_mocked():
    """
    Test output of grid_search_forecaster_multiseries in ForecasterAutoregMultiSeriesCustom
    with mocked (mocked done in Skforecast v0.5.0)
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = Ridge(random_state=123),
                     fun_predictors     = create_predictors,
                     window_size        = 2, 
                     encoding           = 'onehot',
                     transformer_series = None
                 )

    steps = 3
    n_validation = 12
    param_grid = {'alpha': [0.01, 0.1, 1]}

    results = grid_search_forecaster_multiseries(
                  forecaster          = forecaster,
                  series              = series,
                  param_grid          = param_grid,
                  steps               = steps,
                  metric              = 'mean_absolute_error',
                  aggregate_metric    = 'weighted_average',
                  initial_train_size  = len(series) - n_validation,
                  fixed_train_size    = False,
                  levels              = None,
                  exog                = None,
                  lags_grid           = None,
                  refit               = False,
                  return_best         = False,
                  verbose             = True
              )
    
    expected_results = pd.DataFrame({
        'levels': [['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2']],
        'lags'  : ['custom predictors', 'custom predictors', 'custom predictors'],
        'lags_label': ['custom predictors', 'custom predictors', 'custom predictors'],
        'params': [{'alpha': 1}, {'alpha': 0.1}, {'alpha': 0.01}],
        'mean_absolute_error__weighted_average':  np.array(
            [0.21077344827205086, 0.21078653113227208, 0.21078779824759553]
        ),                                                               
        'alpha' : np.array([1., 0.1, 0.01])
        },
        index = pd.Index([0, 1, 2], dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_grid_search_forecaster_multiseries_ForecasterAutoregMultiVariate_with_mocked():
    """
    Test output of grid_search_forecaster_multiseries in ForecasterAutoregMultiVariate 
    with mocked (mocked done in Skforecast v0.6.0)
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 3, 
                     transformer_series = None
                 )

    steps = 3
    n_validation = 12
    lags_grid = [2, 4]
    param_grid = {'alpha': [0.01, 0.1, 1]}

    results = grid_search_forecaster_multivariate(
                  forecaster          = forecaster,
                  series              = series,
                  param_grid          = param_grid,
                  steps               = steps,
                  metric              = 'mean_absolute_error',
                  aggregate_metric    = 'weighted_average',
                  initial_train_size  = len(series) - n_validation,
                  fixed_train_size    = False,
                  levels              = None,
                  exog                = None,
                  lags_grid           = lags_grid,
                  refit               = False,
                  return_best         = False,
                  verbose             = True
              )
    
    expected_results = pd.DataFrame({
        'levels': [['l1'], ['l1'], ['l1'], ['l1'], ['l1'], ['l1']],
        'lags': [[1, 2], [1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        'lags_label': [[1, 2], [1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, {'alpha': 1}, {'alpha': 0.1}, {'alpha': 0.01}],
        'mean_absolute_error':  np.array(
            [0.20115194, 0.20183032, 0.20566862, 0.22224269, 0.22625017, 0.22644284]
        ),                                                               
        'alpha': np.array([0.01, 0.1, 1., 1., 0.1, 0.01])
        },
        index = pd.Index([0, 1, 2, 3, 4, 5], dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results)