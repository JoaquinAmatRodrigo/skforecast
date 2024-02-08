# Unit test _bayesian_search_optuna
# ==============================================================================
import re
import pytest
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection.model_selection import _bayesian_search_optuna
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm
from functools import partialmethod

# Fixtures
from .fixtures_model_selection import y

def create_predictors(y): # pragma: no cover
    """
    Create first 4 lags of a time series, used in ForecasterAutoregCustom.
    """

    lags = y[-1:-5:-1]

    return lags

optuna.logging.set_verbosity(optuna.logging.WARNING)
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True) # hide progress bar


def test_ValueError_bayesian_search_optuna_metric_list_duplicate_names():
    """
    Test ValueError is raised in _bayesian_search_optuna when a `list` of 
    metrics is used with duplicate names.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    def search_space(trial): # pragma: no cover
        search_space  = {
            'alpha': trial.suggest_float('not_alpha', 1e-2, 1.0),
            'lags' : trial.suggest_int('lags', 2, 4)
        }

        return search_space

    err_msg = re.escape("When `metric` is a `list`, each metric name must be unique.")
    with pytest.raises(ValueError, match = err_msg):
        _bayesian_search_optuna(
            forecaster         = forecaster,
            y                  = y,
            search_space       = search_space,
            steps              = steps,
            metric             = ['mean_absolute_error', mean_absolute_error],
            refit              = True,
            initial_train_size = len(y_train),
            fixed_train_size   = True,
            n_trials           = 10,
            random_state       = 123,
            return_best        = False,
            verbose            = False,
        )


def test_ValueError_bayesian_search_optuna_when_search_space_names_do_not_match():
    """
    Test ValueError is raised when search_space key name do not match the trial 
    object name from optuna.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('not_alpha', 1e-2, 1.0),
            'lags' : trial.suggest_int('lags', 2, 4)
        }

        return search_space
    
    err_msg = re.escape(
                """Some of the key values do not match the search_space key names.
                Search Space keys  : ['alpha', 'lags']
                Trial objects keys : ['not_alpha', 'lags'].""")
    with pytest.raises(ValueError, match = err_msg):
        _bayesian_search_optuna(
            forecaster         = forecaster,
            y                  = y,
            search_space       = search_space,
            steps              = steps,
            metric             = 'mean_absolute_error',
            refit              = True,
            initial_train_size = len(y_train),
            fixed_train_size   = True,
            n_trials           = 10,
            random_state       = 123,
            return_best        = False,
            verbose            = False,
        )


# This mark allows to only run test with "slow" label or all except this, "not slow".
# The mark should be included in the pytest.ini file
# pytest -m slow --verbose
# pytest -m "not slow" --verbose
@pytest.mark.slow
@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires python3.8 or higher")
def test_results_output_bayesian_search_optuna_ForecasterAutoreg_with_mocked():
    """
    Test output of _bayesian_search_optuna in ForecasterAutoreg with mocked
    (mocked done in Skforecast v0.4.3).
    """
    forecaster = ForecasterAutoreg(
                    regressor = RandomForestRegressor(random_state=123),
                    lags      = 2
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]

    def search_space(trial):
        search_space  = {
            'n_estimators'    : trial.suggest_int('n_estimators', 10, 20),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 1., log=True),
            'max_features'    : trial.suggest_categorical('max_features', ['log2', 'sqrt']),
            'lags'            : trial.suggest_categorical('lags', [2, 4])
        } 
        
        return search_space

    results = _bayesian_search_optuna(
                  forecaster         = forecaster,
                  y                  = y,
                  search_space       = search_space,
                  steps              = steps,
                  metric             = 'mean_absolute_error',
                  refit              = True,
                  initial_train_size = len(y_train),
                  fixed_train_size   = True,
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False
              )[0]
    
    expected_results = pd.DataFrame(
        np.array([[np.array([1, 2, 3, 4]),
                {'n_estimators': 20, 'min_samples_leaf': 0.4839825891759374, 'max_features': 'log2'},
                0.21252324019730395, 20, 0.4839825891759374, 'log2'],
            [np.array([1, 2]),
                {'n_estimators': 15, 'min_samples_leaf': 0.34027307604369605, 'max_features': 'sqrt'},
                0.21479600790277778, 15, 0.34027307604369605, 'sqrt'],
            [np.array([1, 2]),
                {'n_estimators': 15, 'min_samples_leaf': 0.41010449151752726, 'max_features': 'sqrt'},
                0.21479600790277778, 15, 0.41010449151752726, 'sqrt'],
            [np.array([1, 2]),
                {'n_estimators': 14, 'min_samples_leaf': 0.782328520465639, 'max_features': 'log2'},
                0.21661388810185186, 14, 0.782328520465639, 'log2'],
            [np.array([1, 2]),
                {'n_estimators': 13, 'min_samples_leaf': 0.20142843988664705, 'max_features': 'sqrt'},
                0.22084665733716438, 13, 0.20142843988664705, 'sqrt'],
            [np.array([1, 2, 3, 4]),
                {'n_estimators': 17, 'min_samples_leaf': 0.21035794225904136, 'max_features': 'log2'},
                0.2229839747692736, 17, 0.21035794225904136, 'log2'],
            [np.array([1, 2]),
                {'n_estimators': 17, 'min_samples_leaf': 0.19325882509735576, 'max_features': 'sqrt'},
                0.2235827591941962, 17, 0.19325882509735576, 'sqrt'],
            [np.array([1, 2]),
                {'n_estimators': 11, 'min_samples_leaf': 0.2714570796881701, 'max_features': 'sqrt'},
                0.22384439522399655, 11, 0.2714570796881701, 'sqrt'],
            [np.array([1, 2, 3, 4]),
                {'n_estimators': 13, 'min_samples_leaf': 0.2599119286878713, 'max_features': 'log2'},
                0.22522461935181756, 13, 0.2599119286878713, 'log2'],
            [np.array([1, 2]),
                {'n_estimators': 14, 'min_samples_leaf': 0.1147302385573586, 'max_features': 'sqrt'},
                0.22764885273610677, 14, 0.1147302385573586, 'sqrt']],
            dtype=object),
        columns=['lags', 'params', 'mean_absolute_error', 'n_estimators', 'min_samples_leaf', 'max_features'],
        index=pd.Index([1, 3, 8, 6, 9, 4, 0, 5, 7, 2], dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results, check_dtype=False)
    

def test_results_output_bayesian_search_optuna_ForecasterAutoreg_with_mocked_when_kwargs_create_study():
    """
    Test output of _bayesian_search_optuna in ForecasterAutoreg when 
    kwargs_create_study with mocked (mocked done in Skforecast v0.4.3).
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-2, 1.0),
            'lags' : trial.suggest_categorical('lags', [4, 2])
        }
        
        return search_space

    # kwargs_create_study
    sampler = TPESampler(seed=123, prior_weight=2.0, consider_magic_clip=False)

    results = _bayesian_search_optuna(
                  forecaster          = forecaster,
                  y                   = y,
                  search_space        = search_space,
                  steps               = steps,
                  metric              = 'mean_absolute_error',
                  refit               = True,
                  initial_train_size  = len(y_train),
                  fixed_train_size    = True,
                  n_trials            = 10,
                  random_state        = 123,
                  return_best         = False,
                  verbose             = False,
                  kwargs_create_study = {'sampler':sampler}
              )[0]
    
    expected_results = pd.DataFrame(
        np.array([[np.array([1, 2]), {'alpha': 0.23598059857016607},
            0.21239141697571848, 0.23598059857016607],
        [np.array([1, 2]), {'alpha': 0.398196343012209}, 0.21271021033387602,
            0.398196343012209],
        [np.array([1, 2]), {'alpha': 0.4441865222328282}, 0.2127897499229874,
            0.4441865222328282],
        [np.array([1, 2]), {'alpha': 0.53623586010342}, 0.21293692257888705,
            0.53623586010342],
        [np.array([1, 2]), {'alpha': 0.7252189487445193},
            0.21319693043832985, 0.7252189487445193],
        [np.array([1, 2, 3, 4]), {'alpha': 0.9809565564007693},
            0.21539791166603497, 0.9809565564007693],
        [np.array([1, 2, 3, 4]), {'alpha': 0.8509374761370117},
            0.21557690844753197, 0.8509374761370117],
        [np.array([1, 2, 3, 4]), {'alpha': 0.7406154516747153},
            0.2157346392837304, 0.7406154516747153],
        [np.array([1, 2, 3, 4]), {'alpha': 0.6995044937418831},
            0.21579460210585208, 0.6995044937418831],
        [np.array([1, 2, 3, 4]), {'alpha': 0.5558016213920624},
            0.21600778429729228, 0.5558016213920624]], dtype=object),
        columns=['lags', 'params', 'mean_absolute_error', 'alpha'],
        index=pd.Index([9, 3, 4, 6, 8, 2, 7, 5, 0, 1], dtype='int64')
    )
    expected_results[['mean_absolute_error', 'alpha']] = expected_results[['mean_absolute_error', 'alpha']].astype(float)

    pd.testing.assert_frame_equal(results, expected_results)


def test_results_output_bayesian_search_optuna_ForecasterAutoreg_with_mocked_when_kwargs_study_optimize():
    """
    Test output of _bayesian_search_optuna in ForecasterAutoreg when 
    kwargs_study_optimize with mocked (mocked done in Skforecast v0.4.3).
    """
    forecaster = ForecasterAutoreg(
                    regressor = RandomForestRegressor(random_state=123),
                    lags      = 2 
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]

    def search_space(trial):
        search_space  = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 200),
            'max_depth'   : trial.suggest_int('max_depth', 20, 35, log=True),
            'max_features': trial.suggest_categorical('max_features', ['log2', 'sqrt'])
        } 
        
        return search_space

    # kwargs_study_optimize
    timeout = 2.0

    results = _bayesian_search_optuna(
                  forecaster            = forecaster,
                  y                     = y,
                  lags_grid             = lags_grid,
                  search_space          = search_space,
                  steps                 = steps,
                  metric                = 'mean_absolute_error',
                  refit                 = True,
                  initial_train_size    = len(y_train),
                  fixed_train_size      = True,
                  n_trials              = 10,
                  random_state          = 123,
                  n_jobs                = 1,
                  return_best           = False,
                  verbose               = False,
                  kwargs_study_optimize = {'timeout': timeout}
              )[0].reset_index(drop=True)
    
    expected_results = pd.DataFrame({
        'lags'  : [[1, 2], [1, 2], [1, 2], 
                   [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        'lags_label': [[1, 2], [1, 2], [1, 2], 
                        [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        'params': [{'n_estimators': 170, 'max_depth': 23, 'max_features': 'sqrt'},
                   {'n_estimators': 172, 'max_depth': 25, 'max_features': 'log2'},
                   {'n_estimators': 148, 'max_depth': 25, 'max_features': 'sqrt'}, 
                   {'n_estimators': 170, 'max_depth': 23, 'max_features': 'sqrt'}, 
                   {'n_estimators': 172, 'max_depth': 25, 'max_features': 'log2'}, 
                   {'n_estimators': 148, 'max_depth': 25, 'max_features': 'sqrt'}],
        'mean_absolute_error': np.array([0.21698591071568632, 0.21677310983527143, 0.2207701537612612, 
                                         0.22227287699019596, 0.22139945523255808, 0.22838451457770267]),                                                               
        'n_estimators': np.array([170, 172, 148, 170, 172, 148]),
        'max_depth': np.array([23, 25, 25, 23, 25, 25]),
        'max_features': ['sqrt', 'log2', 'sqrt', 'sqrt', 'log2', 'sqrt']
        },
        index=pd.RangeIndex(start=0, stop=6, step=1)
    ).sort_values(by='mean_absolute_error', ascending=True).reset_index(drop=True)

    pd.testing.assert_frame_equal(results.head(2), expected_results.head(2), check_dtype=False)


def test_results_output_bayesian_search_optuna_ForecasterAutoreg_with_mocked_when_lags_grid_is_None():
    """
    Test output of _bayesian_search_optuna in ForecasterAutoreg when lags_grid 
    is None with mocked (mocked done in Skforecast v0.4.3), should use 
    forecaster.lags as lags_grid.
    """
    forecaster = ForecasterAutoreg(
                     regressor = Ridge(random_state=123),
                     lags      = 4
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = None
    
    def search_space(trial):
        search_space  = {'alpha': trial.suggest_float('alpha', 1e-2, 1.0)}

        return search_space

    results = _bayesian_search_optuna(
                  forecaster         = forecaster,
                  y                  = y,
                  lags_grid          = lags_grid,
                  search_space       = search_space,
                  steps              = steps,
                  metric             = 'mean_absolute_error',
                  refit              = True,
                  initial_train_size = len(y_train),
                  fixed_train_size   = True,
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False
              )[0]
    
    expected_results = pd.DataFrame({
        'lags'  :[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                  [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        'lags_label': [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                        [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
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


def test_results_output_bayesian_search_optuna_ForecasterAutoregCustom_with_mocked():
    """
    Test output of _bayesian_search_optuna in ForecasterAutoregCustom with mocked
    (mocked done in Skforecast v0.4.3).
    """    
    forecaster = ForecasterAutoregCustom(
                     regressor      = Ridge(random_state=123),
                     fun_predictors = create_predictors,
                     window_size    = 4
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    
    def search_space(trial):
        search_space  = {'alpha': trial.suggest_float('alpha', 1e-2, 1.0)}

        return search_space

    results = _bayesian_search_optuna(
                  forecaster         = forecaster,
                  y                  = y,
                  search_space       = search_space,
                  steps              = steps,
                  metric             = 'mean_absolute_error',
                  refit              = True,
                  initial_train_size = len(y_train),
                  fixed_train_size   = True,
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False
              )[0]
    
    expected_results = pd.DataFrame({
        'lags'  :['custom predictors', 'custom predictors', 'custom predictors',
                  'custom predictors', 'custom predictors', 'custom predictors',
                  'custom predictors', 'custom predictors', 'custom predictors',
                  'custom predictors'],
        'lags_label': ['custom predictors', 'custom predictors', 'custom predictors',
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


@pytest.mark.parametrize("lags_grid", 
                         [[2, 4], {'lags_1': 2, 'lags_2': 4}], 
                         ids=lambda lg: f'lags_grid: {lg}')
def test_evaluate_bayesian_search_optuna_when_return_best_ForecasterAutoreg(lags_grid):
    """
    Test forecaster is refitted when return_best=True in _bayesian_search_optuna
    with a ForecasterAutoreg.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    
    def search_space(trial):
        search_space  = {'alpha': trial.suggest_float('alpha', 1e-2, 1.0)}
        
        return search_space

    _bayesian_search_optuna(
        forecaster         = forecaster,
        y                  = y,
        lags_grid          = lags_grid,
        search_space       = search_space,
        steps              = steps,
        metric             = 'mean_absolute_error',
        refit              = True,
        initial_train_size = len(y_train),
        fixed_train_size   = True,
        n_trials           = 10,
        return_best        = True,
        verbose            = False
    )
    
    expected_lags = np.array([1, 2])
    expected_alpha = 0.2345829390285611
    
    np.testing.assert_array_equal(forecaster.lags, expected_lags)
    assert expected_alpha == forecaster.regressor.alpha


def test_evaluate_bayesian_search_optuna_when_return_best_ForecasterAutoregCustom():
    """
    Test forecaster is refitted when return_best=True in _bayesian_search_optuna
    with a ForecasterAutoregCustom.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = Ridge(random_state=123),
                     fun_predictors = create_predictors,
                     window_size    = 4
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = None # Ignored
    
    def search_space(trial):
        search_space  = {'alpha': trial.suggest_float('alpha', 1e-2, 1.0)}
        
        return search_space

    _bayesian_search_optuna(
        forecaster         = forecaster,
        y                  = y,
        lags_grid          = lags_grid,
        search_space       = search_space,
        steps              = steps,
        metric             = 'mean_absolute_error',
        refit              = True,
        initial_train_size = len(y_train),
        fixed_train_size   = True,
        n_trials           = 10,
        random_state       = 123,
        return_best        = True,
        verbose            = False,
        show_progress      = False
    )
    
    expected_alpha = 0.9809565564007693
    
    assert expected_alpha == forecaster.regressor.alpha


def test_results_opt_best_output_bayesian_search_optuna_with_output_study_best_trial_optuna():
    """
    Test results_opt_best output of _bayesian_search_optuna with output 
    study.best_trial optuna.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    metric     = 'mean_absolute_error'
    initial_train_size = len(y_train)
    fixed_train_size   = True
    refit      = True
    verbose    = False
    
    n_trials = 10
    random_state = 123

    def objective(
        trial,
        forecaster         = forecaster,
        y                  = y,
        steps              = steps,
        metric             = metric,
        initial_train_size = initial_train_size,
        fixed_train_size   = fixed_train_size,
        refit              = refit,
        verbose            = verbose,
    ) -> float:
        
        alpha = trial.suggest_float('alpha', 1e-2, 1.0)
        
        forecaster = ForecasterAutoreg(
                        regressor = Ridge(random_state=random_state, 
                                          alpha=alpha),
                        lags      = 2
                     )

        metric, _ = backtesting_forecaster(
                        forecaster = forecaster,
                        y          = y,
                        steps      = steps,
                        metric     = metric,
                        initial_train_size = initial_train_size,
                        fixed_train_size   = fixed_train_size,
                        refit      = refit,
                        verbose    = verbose       
                    )

        return abs(metric)
  
    study = optuna.create_study(direction="minimize", 
                                sampler=TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial

    lags_grid = [4, 2]
    def search_space(trial):
        search_space  = {'alpha' : trial.suggest_float('alpha', 1e-2, 1.0)
                        }
        return search_space
    return_best  = False

    results_opt_best = _bayesian_search_optuna(
                           forecaster         = forecaster,
                           y                  = y,
                           lags_grid          = lags_grid,
                           search_space       = search_space,
                           steps              = steps,
                           metric             = metric,
                           refit              = refit,
                           initial_train_size = initial_train_size,
                           fixed_train_size   = fixed_train_size,
                           n_trials           = n_trials,
                           return_best        = return_best,
                           verbose            = verbose
                       )[1]

    assert best_trial.number == results_opt_best.number
    assert best_trial.values == results_opt_best.values
    assert best_trial.params == results_opt_best.params


def test_bayesian_search_optuna_output_file():
    """ 
    Test output file of _bayesian_search_optuna.
    """

    forecaster = ForecasterAutoreg(
                     regressor = RandomForestRegressor(random_state=123),
                     lags      = 2 
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]

    def search_space(trial):
        search_space  = {
            'n_estimators'    : trial.suggest_int('n_estimators', 10, 20),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 1., log=True),
            'max_features'    : trial.suggest_categorical('max_features', ['log2', 'sqrt'])
        } 
        
        return search_space

    output_file = 'test_bayesian_search_optuna_output_file.txt'
    results = _bayesian_search_optuna(
                  forecaster         = forecaster,
                  y                  = y,
                  lags_grid          = lags_grid,
                  search_space       = search_space,
                  steps              = steps,
                  metric             = 'mean_absolute_error',
                  refit              = True,
                  initial_train_size = len(y_train),
                  fixed_train_size   = True,
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False,
                  output_file        = output_file
              )[0]

    assert os.path.isfile(output_file)
    os.remove(output_file)