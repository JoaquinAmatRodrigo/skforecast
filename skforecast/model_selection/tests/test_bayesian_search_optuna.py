# Unit test _bayesian_search_optuna
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
import sys
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


def test_exception_bayesian_search_optuna_metric_list_duplicate_names():
    """
    Test exception is raised in _bayesian_search_optuna when a `list` of 
    metrics is used with duplicate names.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]
    def search_space(trial): # pragma: no cover
        search_space  = {'alpha' : trial.suggest_float('not_alpha', 1e-2, 1.0)
                        }
        return search_space

    err_msg = re.escape("When `metric` is a `list`, each metric name must be unique.")
    with pytest.raises(ValueError, match = err_msg):
        _bayesian_search_optuna(
            forecaster         = forecaster,
            y                  = y,
            lags_grid          = lags_grid,
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


def test_bayesian_search_optuna_exception_when_search_space_names_do_not_match():
    """
    Test Exception is raised when search_space key name do not match the trial 
    object name from optuna.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]
    def search_space(trial):
        search_space  = {'alpha' : trial.suggest_float('not_alpha', 1e-2, 1.0)
                        }
        return search_space
    
    err_msg = re.escape(
                """Some of the key values do not match the search_space key names.
                Dict keys     : ['alpha']
                Trial objects : ['not_alpha'].""")
    with pytest.raises(ValueError, match = err_msg):
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
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]

    def search_space(trial):
        search_space  = {'n_estimators'     : trial.suggest_int('n_estimators', 10, 20),
                         'min_samples_leaf' : trial.suggest_float('min_samples_leaf', 0.1, 1., log=True),
                         'max_features'     : trial.suggest_categorical('max_features', ['log2', 'sqrt'])
                        } 
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
            'lags'  :[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2], [1, 2], [1, 2],
                      [1, 2], [1, 2], [1, 2, 3, 4], [1, 2], [1, 2, 3, 4], [1, 2], [1, 2],
                      [1, 2], [1, 2, 3, 4], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                      [1, 2, 3, 4]],
            'params':[{'n_estimators': 13, 'min_samples_leaf': 0.42753938073418213, 'max_features': 'sqrt'},
                      {'n_estimators': 14, 'min_samples_leaf': 0.782328520465639, 'max_features': 'log2'},
                      {'n_estimators': 16, 'min_samples_leaf': 0.707020154488976, 'max_features': 'log2'},
                      {'n_estimators': 16, 'min_samples_leaf': 0.707020154488976, 'max_features': 'log2'},
                      {'n_estimators': 14, 'min_samples_leaf': 0.782328520465639, 'max_features': 'log2'},
                      {'n_estimators': 12, 'min_samples_leaf': 0.14977928606210794, 'max_features': 'sqrt'},
                      {'n_estimators': 13, 'min_samples_leaf': 0.42753938073418213, 'max_features': 'sqrt'},
                      {'n_estimators': 14, 'min_samples_leaf': 0.3116628929828935, 'max_features': 'log2'},
                      {'n_estimators': 14, 'min_samples_leaf': 0.3116628929828935, 'max_features': 'log2'},
                      {'n_estimators': 17, 'min_samples_leaf': 0.21035794225904136, 'max_features': 'log2'},
                      {'n_estimators': 17, 'min_samples_leaf': 0.21035794225904136, 'max_features': 'log2'},
                      {'n_estimators': 17, 'min_samples_leaf': 0.19325882509735576, 'max_features': 'sqrt'},
                      {'n_estimators': 15, 'min_samples_leaf': 0.2466706727024324, 'max_features': 'sqrt'},
                      {'n_estimators': 17, 'min_samples_leaf': 0.2649149454284987, 'max_features': 'log2'},
                      {'n_estimators': 15, 'min_samples_leaf': 0.2466706727024324, 'max_features': 'sqrt'},
                      {'n_estimators': 14, 'min_samples_leaf': 0.1147302385573586, 'max_features': 'sqrt'},
                      {'n_estimators': 17, 'min_samples_leaf': 0.2649149454284987, 'max_features': 'log2'},
                      {'n_estimators': 17, 'min_samples_leaf': 0.19325882509735576, 'max_features': 'sqrt'},
                      {'n_estimators': 14, 'min_samples_leaf': 0.1147302385573586, 'max_features': 'sqrt'},
                      {'n_estimators': 12, 'min_samples_leaf': 0.14977928606210794, 'max_features': 'sqrt'}],
            'mean_absolute_error':np.array([0.2077587228, 0.2095295831, 0.2104587292, 0.2152253922,
                                            0.2166138881, 0.2173338079, 0.2173997593, 0.2212038456,
                                            0.2213244194, 0.2228916722, 0.2229839748, 0.2235827592,
                                            0.2247429583, 0.2254556258, 0.2263380591, 0.2276488527,
                                            0.2288371667, 0.2323304396, 0.2373974755, 0.2407737066]),                                                               
            'n_estimators' :np.array([13, 14, 16, 16, 14, 12, 13, 14, 14, 17, 17, 17, 15, 17, 15,
                                      14, 17, 17, 14, 12]),
            'min_samples_leaf' :np.array([0.4275393807, 0.7823285205, 0.7070201545, 0.7070201545,
                                          0.7823285205, 0.1497792861, 0.4275393807, 0.311662893 ,
                                          0.311662893 , 0.2103579423, 0.2103579423, 0.1932588251,
                                          0.2466706727, 0.2649149454, 0.2466706727, 0.1147302386,
                                          0.2649149454, 0.1932588251, 0.1147302386, 0.1497792861]),
            'max_features' :['sqrt', 'log2', 'log2', 'log2', 'log2', 'sqrt', 'sqrt', 'log2',
                             'log2', 'log2', 'log2', 'sqrt', 'sqrt', 'log2', 'sqrt', 'sqrt',
                             'log2', 'sqrt', 'sqrt', 'sqrt']
                                     },
            index=pd.Index([17, 19, 15, 5, 9, 4, 7, 8, 18, 6, 16, 0, 2, 1, 12, 3, 11, 10, 13, 14], dtype="int64")
                                   ).sort_values(by='mean_absolute_error', ascending=True)

    pd.testing.assert_frame_equal(results, expected_results, check_dtype=False)
    

def test_results_output_bayesian_search_optuna_ForecasterAutoreg_with_mocked_when_kwargs_create_study():
    """
    Test output of _bayesian_search_optuna in ForecasterAutoreg when 
    kwargs_create_study with mocked (mocked done in Skforecast v0.4.3).
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [4, 2]

    def search_space(trial):
        search_space  = {'alpha' : trial.suggest_float('alpha', 1e-2, 1.0)
                        }
        return search_space

    # kwargs_create_study
    sampler = TPESampler(seed=123, prior_weight=2.0, consider_magic_clip=False)

    results = _bayesian_search_optuna(
                  forecaster          = forecaster,
                  y                   = y,
                  lags_grid           = lags_grid,
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
    
    expected_results = pd.DataFrame({
        'lags'  :[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                  [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                  [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]],
        'params':[{'alpha': 0.6995044937418831}, {'alpha': 0.29327794160087567},
                  {'alpha': 0.2345829390285611}, {'alpha': 0.5558016213920624},
                  {'alpha': 0.7222742800877074}, {'alpha': 0.42887539552321635},
                  {'alpha': 0.9809565564007693}, {'alpha': 0.6879814411990146},
                  {'alpha': 0.48612258246951734}, {'alpha': 0.398196343012209},
                  {'alpha': 0.6995044937418831}, {'alpha': 0.29327794160087567},
                  {'alpha': 0.2345829390285611}, {'alpha': 0.5558016213920624},
                  {'alpha': 0.7222742800877074}, {'alpha': 0.42887539552321635},
                  {'alpha': 0.9809565564007693}, {'alpha': 0.6879814411990146},
                  {'alpha': 0.48612258246951734}, {'alpha': 0.398196343012209}],
        'mean_absolute_error':np.array([0.2157946 , 0.21639339, 0.21647289, 0.21600778, 0.21576132,
                                        0.21619734, 0.21539791, 0.21581151, 0.21611205, 0.21624265,
                                        0.21316446, 0.21251148, 0.21238838, 0.21296631, 0.21319325,
                                        0.21276374, 0.21347973, 0.21314963, 0.21285869, 0.21271021]),
        'alpha' :np.array([0.69950449, 0.29327794, 0.23458294, 0.55580162, 0.72227428,
                           0.4288754 , 0.98095656, 0.68798144, 0.48612258, 0.39819634,
                           0.69950449, 0.29327794, 0.23458294, 0.55580162, 0.72227428,
                           0.4288754 , 0.98095656, 0.68798144, 0.48612258, 0.39819634])
        },
        index=pd.RangeIndex(start=0, stop=20, step=1)
    ).sort_values(by='mean_absolute_error', ascending=True)

    pd.testing.assert_frame_equal(results, expected_results)


def test_results_output_bayesian_search_optuna_ForecasterAutoreg_with_mocked_when_kwargs_study_optimize():
    """
    Test output of _bayesian_search_optuna in ForecasterAutoreg when 
    kwargs_study_optimize with mocked (mocked done in Skforecast v0.4.3).
    """
    forecaster = ForecasterAutoreg(
                    regressor = RandomForestRegressor(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]

    def search_space(trial):
        search_space  = {'n_estimators' : trial.suggest_int('n_estimators', 100, 200),
                         'max_depth'    : trial.suggest_int('max_depth', 20, 35, log=True),
                         'max_features' : trial.suggest_categorical('max_features', ['log2', 'sqrt'])
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
            'lags'  :[[1, 2], [1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            'params':[{'n_estimators': 170, 'max_depth': 23, 'max_features': 'sqrt'},
                      {'n_estimators': 172, 'max_depth': 25, 'max_features': 'log2'},
                      {'n_estimators': 148, 'max_depth': 25, 'max_features': 'sqrt'}, 
                      {'n_estimators': 170, 'max_depth': 23, 'max_features': 'sqrt'}, 
                      {'n_estimators': 172, 'max_depth': 25, 'max_features': 'log2'}, 
                      {'n_estimators': 148, 'max_depth': 25, 'max_features': 'sqrt'}],
            'mean_absolute_error':np.array([0.21698591071568632, 0.21677310983527143, 0.2207701537612612, 
                               0.22227287699019596, 0.22139945523255808, 0.22838451457770267]),                                                               
            'n_estimators' :np.array([170, 172, 148, 170, 172, 148]),
            'max_depth' :np.array([23, 25, 25, 23, 25, 25]),
            'max_features' :['sqrt', 'log2', 'sqrt', 'sqrt', 'log2', 'sqrt']
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
        search_space  = {'alpha' : trial.suggest_float('alpha', 1e-2, 1.0)
                        }
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
        search_space  = {'alpha' : trial.suggest_float('alpha', 1e-2, 1.0)
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
    

def test_evaluate_bayesian_search_optuna_when_return_best():
    """
    Test forecaster is refitted when return_best=True in _bayesian_search_optuna.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]
    
    def search_space(trial):
        search_space  = {'alpha' : trial.suggest_float('alpha', 1e-2, 1.0)
                        }
        return search_space

    _bayesian_search_optuna(
        forecaster   = forecaster,
        y            = y,
        lags_grid    = lags_grid,
        search_space = search_space,
        steps        = steps,
        metric       = 'mean_absolute_error',
        refit        = True,
        initial_train_size = len(y_train),
        fixed_train_size   = True,
        n_trials     = 10,
        return_best  = True,
        verbose      = False
    )
    
    expected_lags = np.array([1, 2])
    expected_alpha = 0.2345829390285611
    
    np.testing.assert_array_equal(forecaster.lags, expected_lags)
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