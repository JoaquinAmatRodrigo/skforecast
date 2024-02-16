# Unit test bayesian_search_forecaster
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
# from skopt.space import Real
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.model_selection import bayesian_search_forecaster

# Fixtures
from .fixtures_model_selection import y

def create_predictors(y): # pragma: no cover
    """
    Create first 4 lags of a time series, used in ForecasterAutoregCustom.
    """

    lags = y[-1:-5:-1]

    return lags


def test_ValueError_bayesian_search_forecaster_when_return_best_and_len_y_exog_different():
    """
    Test ValueError is raised in bayesian_search_forecaster when return_best 
    and length of `y` and `exog` do not match.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )
    exog = y[:30]

    def search_space(trial): # pragma: no cover
        search_space  = {'alpha' : trial.suggest_float('alpha', 1e-2, 1.0)
                        }
        return search_space

    err_msg = re.escape(
        (f"`exog` must have same number of samples as `y`. "
         f"length `exog`: ({len(exog)}), length `y`: ({len(y)})")
    )
    with pytest.raises(ValueError, match = err_msg):
        bayesian_search_forecaster(
            forecaster         = forecaster,
            y                  = y,
            exog               = exog,
            search_space       = search_space,
            steps              = 3,
            metric             = 'mean_absolute_error',
            refit              = True,
            initial_train_size = len(y[:-12]),
            fixed_train_size   = True,
            n_trials           = 10,
            random_state       = 123,
            return_best        = True,
            verbose            = False,
            engine             = 'optuna'
        )


def test_bayesian_search_forecaster_ValueError_when_engine_not_optuna():
    """
    Test ValueError in bayesian_search_forecaster is raised when engine 
    is not 'optuna'.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    def search_space(trial): # pragma: no cover
        search_space  = {'alpha' : trial.suggest_float('alpha', 1e-2, 1.0)
                        }
        return search_space

    engine = 'not_optuna'
    
    err_msg = re.escape(f"`engine` only allows 'optuna', got {engine}.")
    with pytest.raises(ValueError, match = err_msg):
        bayesian_search_forecaster(
            forecaster         = forecaster,
            y                  = y,
            search_space       = search_space,
            steps              = 3,
            metric             = 'mean_absolute_error',
            refit              = True,
            initial_train_size = len(y[:-12]),
            fixed_train_size   = True,
            n_trials           = 10,
            random_state       = 123,
            return_best        = False,
            verbose            = False,
            engine             = engine
        )


def test_results_output_bayesian_search_forecaster_optuna_engine_ForecasterAutoreg_with_mocked():
    """
    Test output of bayesian_search_forecaster in ForecasterAutoreg with 
    mocked using optuna engine (mocked done in Skforecast v0.4.3).
    """    
    forecaster = ForecasterAutoreg(
                     regressor = Ridge(random_state=123),
                     lags      = 4
                 )

    def search_space(trial): # pragma: no cover
        search_space  = {'alpha' : trial.suggest_float('alpha', 1e-2, 1.0)}
        
        return search_space

    results = bayesian_search_forecaster(
                  forecaster         = forecaster,
                  y                  = y,
                  search_space       = search_space,
                  steps              = 3,
                  metric             = 'mean_absolute_error',
                  refit              = True,
                  initial_train_size = len(y[:-12]),
                  fixed_train_size   = True,
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False,
                  engine             = 'optuna'
              )[0]
    
    expected_results = pd.DataFrame(
        np.array([[np.array([1, 2, 3, 4]), {'alpha': 0.9809565564007693},
            0.21539791166603497, 0.9809565564007693],
        [np.array([1, 2, 3, 4]), {'alpha': 0.7222742800877074},
            0.21576131952657338, 0.7222742800877074],
        [np.array([1, 2, 3, 4]), {'alpha': 0.6995044937418831},
            0.21579460210585208, 0.6995044937418831],
        [np.array([1, 2, 3, 4]), {'alpha': 0.6879814411990146},
            0.21581150916013206, 0.6879814411990146],
        [np.array([1, 2, 3, 4]), {'alpha': 0.5558016213920624},
            0.21600778429729228, 0.5558016213920624],
        [np.array([1, 2, 3, 4]), {'alpha': 0.48612258246951734},
            0.21611205459571634, 0.48612258246951734],
        [np.array([1, 2, 3, 4]), {'alpha': 0.42887539552321635},
            0.2161973389956996, 0.42887539552321635],
        [np.array([1, 2, 3, 4]), {'alpha': 0.398196343012209},
            0.2162426532005299, 0.398196343012209],
        [np.array([1, 2, 3, 4]), {'alpha': 0.29327794160087567},
            0.2163933942116072, 0.29327794160087567],
        [np.array([1, 2, 3, 4]), {'alpha': 0.2345829390285611},
            0.21647289061896782, 0.2345829390285611]], dtype=object),
        columns=['lags', 'params', 'mean_absolute_error', 'alpha'],
        index=pd.Index([6, 4, 0, 7, 3, 8, 5, 9, 1, 2], dtype='int64'),
    ).astype({
        'mean_absolute_error': float, 
        'alpha': float
    })

    pd.testing.assert_frame_equal(results, expected_results)


def test_results_output_bayesian_search_forecaster_optuna_engine_ForecasterAutoregCustom_with_mocked():
    """
    Test output of bayesian_search_forecaster in ForecasterAutoregCustom with 
    mocked using optuna engine (mocked done in Skforecast v0.4.3).
    """    
    forecaster = ForecasterAutoregCustom(
                     regressor      = Ridge(random_state=123),
                     fun_predictors = create_predictors,
                     window_size    = 4
                 )

    def search_space(trial): # pragma: no cover
        search_space  = {'alpha' : trial.suggest_float('alpha', 1e-2, 1.0)}

        return search_space

    results = bayesian_search_forecaster(
                  forecaster         = forecaster,
                  y                  = y,
                  search_space       = search_space,
                  steps              = 3,
                  metric             = 'mean_absolute_error',
                  refit              = True,
                  initial_train_size = len(y[:-12]),
                  fixed_train_size   = True,
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False,
                  engine             = 'optuna'
              )[0]
    
    expected_results = pd.DataFrame(
        np.array([['custom function: create_predictors',
        {'alpha': 0.9809565564007693}, 0.21539791166603497,
        0.9809565564007693],
        ['custom function: create_predictors',
        {'alpha': 0.7222742800877074}, 0.21576131952657338,
        0.7222742800877074],
        ['custom function: create_predictors',
        {'alpha': 0.6995044937418831}, 0.21579460210585208,
        0.6995044937418831],
        ['custom function: create_predictors',
        {'alpha': 0.6879814411990146}, 0.21581150916013206,
        0.6879814411990146],
        ['custom function: create_predictors',
        {'alpha': 0.5558016213920624}, 0.21600778429729228,
        0.5558016213920624],
        ['custom function: create_predictors',
        {'alpha': 0.48612258246951734}, 0.21611205459571634,
        0.48612258246951734],
        ['custom function: create_predictors',
        {'alpha': 0.42887539552321635}, 0.21619733899569962,
        0.42887539552321635],
        ['custom function: create_predictors',
        {'alpha': 0.398196343012209}, 0.2162426532005299,
        0.398196343012209],
        ['custom function: create_predictors',
        {'alpha': 0.29327794160087567}, 0.2163933942116072,
        0.29327794160087567],
        ['custom function: create_predictors',
        {'alpha': 0.2345829390285611}, 0.21647289061896782,
        0.2345829390285611]], dtype=object),
        columns=['lags', 'params', 'mean_absolute_error', 'alpha'],
        index=pd.Index([6, 4, 0, 7, 3, 8, 5, 9, 1, 2], dtype='int64')
    ).astype({
        'mean_absolute_error': float, 
        'alpha': float
    })

    pd.testing.assert_frame_equal(results, expected_results)