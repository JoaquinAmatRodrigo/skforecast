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
            lags_grid          = [2, 4],
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
    
    err_msg = re.escape(f"""`engine` only allows 'optuna', got {engine}.""")
    with pytest.raises(ValueError, match = err_msg):
        bayesian_search_forecaster(
            forecaster         = forecaster,
            y                  = y,
            lags_grid          = [2, 4],
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
        search_space  = {'alpha' : trial.suggest_float('alpha', 1e-2, 1.0)
                        }
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
    
    expected_results = pd.DataFrame({
        'lags'  :['custom predictors', 'custom predictors', 'custom predictors',
                  'custom predictors', 'custom predictors', 'custom predictors',
                  'custom predictors', 'custom predictors', 'custom predictors',
                  'custom predictors'],
        'params':[{'alpha': 0.6995044937418831}, {'alpha': 0.29327794160087567},
                  {'alpha': 0.2345829390285611}, {'alpha': 0.5558016213920624},
                  {'alpha': 0.7222742800877074}, {'alpha': 0.42887539552321635},
                  {'alpha': 0.9809565564007693}, {'alpha': 0.6879814411990146},
                  {'alpha': 0.48612258246951734}, {'alpha': 0.398196343012209}],
        'mean_absolute_error':np.array([0.2157946 , 0.21639339, 0.21647289, 0.21600778, 0.21576132,
                                        0.21619734, 0.21539791, 0.21581151, 0.21611205, 0.216242651]),
        'alpha' :np.array([0.69950449, 0.29327794, 0.23458294, 0.55580162, 0.72227428,
                           0.4288754 , 0.98095656, 0.68798144, 0.48612258, 0.39819634])
        },
        index=pd.RangeIndex(start=0, stop=10, step=1)
    ).sort_values(by='mean_absolute_error', ascending=True)

    pd.testing.assert_frame_equal(results, expected_results)


@pytest.mark.skip(reason="`_bayesian_search_skopt` is deprecated since skforecast 0.7.0")
def test_results_output_bayesian_search_forecaster_skopt_engine_ForecasterAutoregCustom_with_mocked(): # pragma: no cover
    """
    Test output of bayesian_search_forecaster in ForecasterAutoregCustom with mocked
    using skopt engine (mocked done in Skforecast v0.4.3).
    """    
    forecaster = ForecasterAutoregCustom(
                        regressor      = Ridge(random_state=123),
                        fun_predictors = create_predictors,
                        window_size    = 4
                 )

    engine = 'skopt'

    results = bayesian_search_forecaster(
                  forecaster         = forecaster,
                  y                  = y,
                  search_space       = {'alpha': Real(0.01, 1.0, "log-uniform", name='alpha')},
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
              )[0]
    
    expected_results = pd.DataFrame({
            'lags'  :['custom predictors', 'custom predictors', 'custom predictors',
                      'custom predictors', 'custom predictors', 'custom predictors',
                      'custom predictors', 'custom predictors', 'custom predictors',
                      'custom predictors'],
            'params':[{'alpha': 0.26663099972129245}, {'alpha': 0.07193526575307788},
                      {'alpha': 0.24086278856848584}, {'alpha': 0.27434725570656354},
                      {'alpha': 0.0959926247515687}, {'alpha': 0.3631244766604131},
                      {'alpha': 0.06635119445083354}, {'alpha': 0.14434062917737708},
                      {'alpha': 0.019050287104581624}, {'alpha': 0.0633920962590419}],
            'mean_absolute_error':np.array([0.21643005790510492, 0.21665565996188138, 0.2164646190462156,
                               0.21641953058020516, 0.21663365234334242, 0.2162939165190013, 
                               0.21666043214039407, 0.21658325961136823, 0.21669499028423744, 
                               0.21666290650172168]),                                                               
            'alpha' :np.array([0.26663099972129245, 0.07193526575307788, 0.24086278856848584, 
                               0.27434725570656354, 0.0959926247515687, 0.3631244766604131,
                               0.06635119445083354, 0.14434062917737708, 0.019050287104581624, 
                               0.0633920962590419])
                                     },
            index=pd.RangeIndex(start=0, stop=10, step=1)
                                   ).sort_values(by='mean_absolute_error', ascending=True)

    pd.testing.assert_frame_equal(results, expected_results)