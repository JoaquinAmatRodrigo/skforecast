# Unit test bayesian_search_forecaster_multiseries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.ForecasterAutoregMultiSeriesCustom import ForecasterAutoregMultiSeriesCustom
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from skforecast.model_selection_multiseries import bayesian_search_forecaster_multiseries
from skforecast.model_selection_multiseries import bayesian_search_forecaster_multivariate

# Fixtures
from .fixtures_model_selection_multiseries import series

def create_predictors(y): # pragma: no cover
    """
    Create first 4 lags of a time series.
    """

    lags = y[-1:-5:-1]

    return lags


def test_ValueError_bayesian_search_forecaster_multiseries_when_return_best_and_len_series_exog_different():
    """
    Test ValueError is raised in bayesian_search_forecaster_multiseries when 
    return_best and length of `series` and `exog` do not match.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot'
                 )
    exog = series[:30].copy()

    def search_space(trial): # pragma: no cover
        search_space  = {'alpha' : trial.suggest_float('alpha', 1e-2, 1.0)}

        return search_space

    err_msg = re.escape(
            (f"`exog` must have same number of samples as `series`. "
             f"length `exog`: ({len(exog)}), length `series`: ({len(series)})")
        )
    with pytest.raises(ValueError, match = err_msg):
        bayesian_search_forecaster_multiseries(
            forecaster         = forecaster,
            series             = series,
            exog               = exog,
            lags_grid          = [2, 4],
            search_space       = search_space,
            steps              = 3,
            metric             = 'mean_absolute_error',
            refit              = True,
            initial_train_size = len(series[:-12]),
            fixed_train_size   = True,
            n_trials           = 10,
            random_state       = 123,
            return_best        = True,
            verbose            = False,
            engine             = 'optuna'
        )


def test_bayesian_search_forecaster_multiseries_ValueError_when_engine_not_optuna():
    """
    Test ValueError in bayesian_search_forecaster_multiseries is raised when engine 
    is not 'optuna'.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot'
                 )

    def search_space(trial): # pragma: no cover
        search_space  = {'alpha' : trial.suggest_float('alpha', 1e-2, 1.0)}

        return search_space

    engine = 'not_optuna'
    
    err_msg = re.escape(f"`engine` only allows 'optuna', got {engine}.")
    with pytest.raises(ValueError, match = err_msg):
        bayesian_search_forecaster_multiseries(
            forecaster         = forecaster,
            series             = series,
            lags_grid          = [2, 4],
            search_space       = search_space,
            steps              = 3,
            metric             = 'mean_absolute_error',
            refit              = True,
            initial_train_size = len(series[:-12]),
            fixed_train_size   = True,
            n_trials           = 10,
            random_state       = 123,
            return_best        = False,
            verbose            = False,
            engine             = engine
        )


def test_results_output_bayesian_search_forecaster_multiseries_optuna_engine_ForecasterAutoregMultiSeries_with_mocked():
    """
    Test output of bayesian_search_forecaster_multiseries in 
    ForecasterAutoregMultiSeries with mocked using 
    optuna engine (mocked done in Skforecast v0.12.0).
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot'
                 )

    steps = 3
    n_validation = 12
    lags_grid = {'lags_1': 2, 'lags_2': 4}

    def search_space(trial):
        search_space  = {'alpha' : trial.suggest_float('alpha', 1e-2, 1.0)}

        return search_space

    results = bayesian_search_forecaster_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  steps              = steps,
                  lags_grid          = lags_grid,
                  search_space       = search_space,
                  metric             = 'mean_absolute_error',
                  refit              = False,
                  initial_train_size = len(series) - n_validation,
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False,
                  engine             = 'optuna'
              )[0]
    
    expected_results = pd.DataFrame({
        'levels': [['l1', 'l2']]*10*2,
        'lags'  : ['lags_2']*10 + ['lags_1']*10,
        'params': [{'alpha': 0.2345829390285611}, {'alpha': 0.29327794160087567},
                   {'alpha': 0.398196343012209}, {'alpha': 0.42887539552321635},
                   {'alpha': 0.48612258246951734}, {'alpha': 0.5558016213920624},
                   {'alpha': 0.6879814411990146}, {'alpha': 0.6995044937418831},
                   {'alpha': 0.7222742800877074}, {'alpha': 0.9809565564007693},
                   {'alpha': 0.2345829390285611}, {'alpha': 0.29327794160087567},
                   {'alpha': 0.398196343012209}, {'alpha': 0.42887539552321635},
                   {'alpha': 0.48612258246951734}, {'alpha': 0.5558016213920624},
                   {'alpha': 0.6879814411990146}, {'alpha': 0.6995044937418831},
                   {'alpha': 0.7222742800877074}, {'alpha': 0.9809565564007693}],
        'mean_absolute_error': np.array([0.20880267, 0.2088056 , 0.20881083, 0.20881236, 0.2088152 ,
                                         0.20881864, 0.20882514, 0.20882571, 0.20882682, 0.20883941,
                                         0.20912689, 0.20912843, 0.2091312 , 0.209132  , 0.2091335 ,
                                         0.20913533, 0.20913878, 0.20913908, 0.20913967, 0.20914639]),
        'alpha': np.array([0.23458294, 0.29327794, 0.39819634, 0.4288754 , 0.48612258,
                           0.55580162, 0.68798144, 0.69950449, 0.72227428, 0.98095656,
                           0.23458294, 0.29327794, 0.39819634, 0.4288754 , 0.48612258,
                           0.55580162, 0.68798144, 0.69950449, 0.72227428, 0.98095656])
        },
        index=pd.Index([12, 11, 19, 15, 18, 13, 17, 10, 14, 16, 2, 1, 9, 5, 8, 3, 7, 0, 4, 6], dtype="int64")
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_results_output_bayesian_search_forecaster_multiseries_optuna_engine_ForecasterAutoregMultiSeriesCustom_with_mocked():
    """
    Test output of bayesian_search_forecaster_multiseries in 
    ForecasterAutoregMultiSeriesCustom with mocked using 
    optuna engine (mocked done in Skforecast v0.12.0).
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = Ridge(random_state=123),
                     fun_predictors = create_predictors,
                     window_size    = 4
                 )

    steps = 3
    n_validation = 12
    lags_grid = None # Ignored

    def search_space(trial):
        search_space  = {'alpha': trial.suggest_float('alpha', 1e-2, 1.0)}
        
        return search_space

    results = bayesian_search_forecaster_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  steps              = steps,
                  lags_grid          = lags_grid,
                  search_space       = search_space,
                  metric             = 'mean_absolute_error',
                  refit              = True,
                  initial_train_size = len(series) - n_validation,
                  fixed_train_size   = True,
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False,
                  engine             = 'optuna'
              )[0]
    
    expected_results = pd.DataFrame({
        'levels': [['l1', 'l2']]*10,
        'lags'  : ['custom predictors']*10,
        'params': [{'alpha': 0.2345829390285611}, {'alpha': 0.29327794160087567},
                   {'alpha': 0.398196343012209}, {'alpha': 0.42887539552321635},
                   {'alpha': 0.48612258246951734}, {'alpha': 0.5558016213920624},
                   {'alpha': 0.6879814411990146}, {'alpha': 0.6995044937418831},
                   {'alpha': 0.7222742800877074}, {'alpha': 0.9809565564007693}],
        'mean_absolute_error': np.array([0.21476183, 0.21476722, 0.21477679, 0.21477957, 0.21478474,
                                         0.21479101, 0.21480278, 0.2148038 , 0.21480582, 0.21482843]),
        'alpha': np.array([0.23458294, 0.29327794, 0.39819634, 0.4288754 , 0.48612258,
                           0.55580162, 0.68798144, 0.69950449, 0.72227428, 0.98095656])
        },
        index=pd.Index([2, 1, 9, 5, 8, 3, 7, 0, 4, 6], dtype="int64")
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_results_output_bayesian_search_forecaster_multivariate_optuna_engine_ForecasterAutoregMultiVariate_with_mocked():
    """
    Test output of bayesian_search_forecaster_multivariate in 
    ForecasterAutoregMultiVariate with mocked using 
    optuna engine (mocked done in Skforecast v0.12.0).
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor = Ridge(random_state=123),
                     level     = 'l1',
                     lags      = 2,
                     steps     = 3
                 )

    steps = 3
    n_validation = 12
    lags_grid = [2, {'l1': 4, 'l2': [2, 3]}]

    def search_space(trial):
        search_space  = {'alpha' : trial.suggest_float('alpha', 1e-2, 1.0)}

        return search_space

    results = bayesian_search_forecaster_multivariate(
                  forecaster         = forecaster,
                  series             = series,
                  steps              = steps,
                  lags_grid          = lags_grid,
                  search_space       = search_space,
                  metric             = 'mean_absolute_error',
                  refit              = False,
                  initial_train_size = len(series) - n_validation,
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False,
                  engine             = 'optuna'
              )[0]
    
    expected_results = pd.DataFrame({
        'levels': [['l1']]*10*2,
        'lags'  : [{'l1': [1, 2, 3, 4], 'l2': [2, 3]}]*10 + [[1, 2]]*10,
        'params': [{'alpha': 0.2345829390285611}, {'alpha': 0.29327794160087567},
                   {'alpha': 0.398196343012209}, {'alpha': 0.42887539552321635},
                   {'alpha': 0.48612258246951734}, {'alpha': 0.5558016213920624},
                   {'alpha': 0.6879814411990146}, {'alpha': 0.6995044937418831},
                   {'alpha': 0.7222742800877074}, {'alpha': 0.9809565564007693},
                   {'alpha': 0.2345829390285611}, {'alpha': 0.29327794160087567},
                   {'alpha': 0.398196343012209}, {'alpha': 0.42887539552321635},
                   {'alpha': 0.48612258246951734}, {'alpha': 0.5558016213920624},
                   {'alpha': 0.6879814411990146}, {'alpha': 0.6995044937418831},
                   {'alpha': 0.7222742800877074}, {'alpha': 0.9809565564007693}],
        'mean_absolute_error': np.array([0.19308029, 0.19311433, 0.19317444, 0.19319184, 0.1932241 ,
                                         0.19326299, 0.19333567, 0.19334194, 0.1933543 , 0.19349184,
                                         0.20117356, 0.20119892, 0.20124394, 0.20125703, 0.20128136,
                                         0.20131081, 0.2013662 , 0.201371  , 0.20138047, 0.20148678]),
        'alpha': np.array([0.23458294, 0.29327794, 0.39819634, 0.4288754 , 0.48612258,
                           0.55580162, 0.68798144, 0.69950449, 0.72227428, 0.98095656,
                           0.23458294, 0.29327794, 0.39819634, 0.4288754 , 0.48612258,
                           0.55580162, 0.68798144, 0.69950449, 0.72227428, 0.98095656])
        },
        index=pd.Index([12, 11, 19, 15, 18, 13, 17, 10, 14, 16, 2, 1, 9, 5, 8, 3, 7, 0, 4, 6], dtype="int64")
    )

    pd.testing.assert_frame_equal(results, expected_results)