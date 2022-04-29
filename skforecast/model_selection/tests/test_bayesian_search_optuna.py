# Unit test _bayesian_search_optuna
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection.model_selection import _bayesian_search_optuna

from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True) # hide progress bar

# Fixtures _backtesting_forecaster_refit Series (skforecast==0.4.2)
# np.random.seed(123)
# y = np.random.rand(50)

y = pd.Series(
    np.array([0.69646919, 0.28613933, 0.22685145, 0.55131477, 0.71946897,
              0.42310646, 0.9807642 , 0.68482974, 0.4809319 , 0.39211752,
              0.34317802, 0.72904971, 0.43857224, 0.0596779 , 0.39804426,
              0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759,
              0.63440096, 0.84943179, 0.72445532, 0.61102351, 0.72244338,
              0.32295891, 0.36178866, 0.22826323, 0.29371405, 0.63097612,
              0.09210494, 0.43370117, 0.43086276, 0.4936851 , 0.42583029,
              0.31226122, 0.42635131, 0.89338916, 0.94416002, 0.50183668,
              0.62395295, 0.1156184 , 0.31728548, 0.41482621, 0.86630916,
              0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453]))


def test_bayesian_search_optuna_exception_when_search_space_names_do_not_match():
    '''
    Test Exception is raised when search_space key name do not match the trial 
    object name from optuna.
    '''
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]
    def search_space(trial):
        search_space  = {'alpha' : trial.suggest_loguniform('not_alpha', 1e-2, 1.0)
                        }
        return search_space
    
    with pytest.raises(Exception):
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
            random_state = 123,
            return_best  = False,
            verbose      = False,
        )


def test_results_output_bayesian_search_optuna_ForecasterAutoreg_with_mocked():
    '''
    Test output of _bayesian_search_optuna in ForecasterAutoreg with mocked
    (mocked done in Skforecast v0.4.3).
    '''
    forecaster = ForecasterAutoreg(
                    regressor = RandomForestRegressor(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]

    def search_space(trial):
        search_space  = {'n_estimators' : trial.suggest_int('n_estimators', 10, 20),
                        'max_depth'    : trial.suggest_float('max_depth', 1, 5, log=True),
                        'max_features' : trial.suggest_categorical('max_features', ['auto', 'sqrt'])
                        } 
        return search_space

    results = _bayesian_search_optuna(
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
                    random_state = 123,
                    return_best  = False,
                    verbose      = False
              )[0]
    
    expected_results = pd.DataFrame({
            'lags'  :[[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2],
                      [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                      [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            'params':[{'n_estimators': 17, 'max_depth': 1.5849034551675116, 'max_features': 'sqrt'}, 
                      {'n_estimators': 17, 'max_depth': 1.9757806338799633, 'max_features': 'auto'}, 
                      {'n_estimators': 15, 'max_depth': 1.8796560348402316, 'max_features': 'sqrt'}, 
                      {'n_estimators': 14, 'max_depth': 1.1008117578626078, 'max_features': 'sqrt'}, 
                      {'n_estimators': 12, 'max_depth': 1.3262808982725627, 'max_features': 'sqrt'}, 
                      {'n_estimators': 16, 'max_depth': 3.923985056308672, 'max_features': 'auto'}, 
                      {'n_estimators': 17, 'max_depth': 1.6816615459597437, 'max_features': 'auto'}, 
                      {'n_estimators': 13, 'max_depth': 2.760790068786577, 'max_features': 'sqrt'}, 
                      {'n_estimators': 14, 'max_depth': 2.2134569252046123, 'max_features': 'auto'}, 
                      {'n_estimators': 14, 'max_depth': 4.211649556832716, 'max_features': 'auto'}, 
                      {'n_estimators': 17, 'max_depth': 1.5849034551675116, 'max_features': 'sqrt'}, 
                      {'n_estimators': 17, 'max_depth': 1.9757806338799633, 'max_features': 'auto'}, 
                      {'n_estimators': 15, 'max_depth': 1.8796560348402316, 'max_features': 'sqrt'}, 
                      {'n_estimators': 14, 'max_depth': 1.1008117578626078, 'max_features': 'sqrt'}, 
                      {'n_estimators': 12, 'max_depth': 1.3262808982725627, 'max_features': 'sqrt'}, 
                      {'n_estimators': 16, 'max_depth': 3.923985056308672, 'max_features': 'auto'}, 
                      {'n_estimators': 17, 'max_depth': 1.6816615459597437, 'max_features': 'auto'}, 
                      {'n_estimators': 13, 'max_depth': 2.760790068786577, 'max_features': 'sqrt'}, 
                      {'n_estimators': 14, 'max_depth': 2.2134569252046123, 'max_features': 'auto'},
                      {'n_estimators': 14, 'max_depth': 4.211649556832716, 'max_features': 'auto'}],
            'metric':np.array([0.21615799463348997, 0.22474421769386224, 0.21783651613263622,
                               0.2179087225230932, 0.21637266491159377, 0.2059816523217619,
                               0.22474421769386224, 0.22570844556862046, 0.227837004285555, 
                               0.19088687600216567, 0.2198566352889403, 0.22528691869878895,
                               0.22451991800112966, 0.22323775188486458, 0.22513141094009306,
                               0.2215061369598327, 0.22528691869878895, 0.21935612354779513,
                               0.23578208465562722, 0.20480074634509826]),                                                               
            'n_estimators' :np.array([17, 17, 15, 14, 12, 16, 17, 13, 14, 14, 17, 17, 15, 14,
                                      12, 16, 17, 13, 14, 14]),
            'max_depth' :np.array([1.5849034551675116, 1.9757806338799633, 1.8796560348402316,
                                   1.1008117578626078, 1.3262808982725627, 3.923985056308672,
                                   1.6816615459597437, 2.760790068786577, 2.2134569252046123,
                                   4.211649556832716, 1.5849034551675116, 1.9757806338799633,
                                   1.8796560348402316, 1.1008117578626078, 1.3262808982725627,
                                   3.923985056308672, 1.6816615459597437, 2.760790068786577,
                                   2.2134569252046123, 4.211649556832716]),
            'max_features' :['sqrt', 'auto', 'sqrt', 'sqrt', 'sqrt', 'auto', 'auto', 'sqrt', 
                             'auto', 'auto', 'sqrt', 'auto', 'sqrt', 'sqrt', 'sqrt', 'auto',
                             'auto', 'sqrt', 'auto', 'auto']
                                     },
            index=list(range(20))
                                   ).sort_values(by='metric', ascending=True)

    pd.testing.assert_frame_equal(results, expected_results)
    

def test_results_output_bayesian_search_optuna_ForecasterAutoreg_with_mocked_when_args_kwargs():
    '''
    Test output of _bayesian_search_optuna in ForecasterAutoreg when args and kwargs with mocked
    (mocked done in Skforecast v0.4.3).
    '''
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [4, 2]

    def search_space(trial):
        search_space  = {'alpha' : trial.suggest_loguniform('alpha', 1e-2, 1.0)
                        }
        return search_space

    # *args and **kwargs for optuna.samplers.TPESampler() argument in optuna.create_study()
    prior_weight        = 2.0
    consider_magic_clip = False

    results = _bayesian_search_optuna(
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
                    random_state = 123,
                    return_best  = False,
                    verbose      = False,
                    prior_weight        = prior_weight,
                    consider_magic_clip = consider_magic_clip
              )[0]
    
    expected_results = pd.DataFrame({
            'lags'  :[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                      [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                      [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]],
            'params':[{'alpha': 0.24713734184878824}, {'alpha': 0.03734897347801035}, 
                      {'alpha': 0.028425159292991616}, {'alpha': 0.12665709946616685},
                      {'alpha': 0.274750150868112}, {'alpha': 0.07017992831138445}, 
                      {'alpha': 0.9152261002780916}, {'alpha': 0.23423914662544418},
                      {'alpha': 0.09159332036121723}, {'alpha': 0.06084642077147053},
                      {'alpha': 0.24713734184878824}, {'alpha': 0.03734897347801035}, 
                      {'alpha': 0.028425159292991616}, {'alpha': 0.12665709946616685},
                      {'alpha': 0.274750150868112}, {'alpha': 0.07017992831138445},
                      {'alpha': 0.9152261002780916}, {'alpha': 0.23423914662544418}, 
                      {'alpha': 0.09159332036121723}, {'alpha': 0.06084642077147053}],
            'metric':np.array([0.216456292811124, 0.21668294660495988, 0.2166890437876308, 
                               0.21660255096339978, 0.2164189788190982, 0.2166571743908303,
                               0.21548741275543748, 0.2164733416179101, 0.2166378443581651, 
                               0.21666500432810773, 0.2124154959419979, 0.21189487956437836, 
                               0.21186904800174858, 0.21213539597840395, 0.21247361506525617,
                               0.2119869822558119, 0.21341333210491734, 0.21238762670131375,
                               0.21204468608736057, 0.21196125599125668]),                                                               
            'alpha' :np.array([0.24713734184878824, 0.03734897347801035, 0.028425159292991616,
                               0.12665709946616685, 0.274750150868112, 0.07017992831138445, 
                               0.9152261002780916, 0.23423914662544418, 0.09159332036121723,
                               0.06084642077147053, 0.24713734184878824, 0.03734897347801035, 
                               0.028425159292991616, 0.12665709946616685, 0.274750150868112, 
                               0.07017992831138445, 0.9152261002780916, 0.23423914662544418, 
                               0.09159332036121723, 0.06084642077147053,])
                                     },
            index=list(range(20))
                                   ).sort_values(by='metric', ascending=True)

    pd.testing.assert_frame_equal(results, expected_results)


def test_results_output_bayesian_search_optuna_ForecasterAutoreg_with_mocked_when_lags_grid_is_None():
    '''
    Test output of _bayesian_search_optuna in ForecasterAutoreg when lags_grid is None with mocked
    (mocked done in Skforecast v0.4.3), should use forecaster.lags as lags_grid.
    '''
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 4
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = None
    
    def search_space(trial):
        search_space  = {'alpha' : trial.suggest_loguniform('alpha', 1e-2, 1.0)
                        }
        return search_space

    results = _bayesian_search_optuna(
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
                    random_state = 123,
                    return_best  = False,
                    verbose      = False
              )[0]
    
    expected_results = pd.DataFrame({
            'lags'  :[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                      [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            'params':[{'alpha': 0.24713734184878824}, {'alpha': 0.03734897347801035}, 
                      {'alpha': 0.028425159292991616}, {'alpha': 0.12665709946616685}, 
                      {'alpha': 0.274750150868112}, {'alpha': 0.07017992831138445},
                      {'alpha': 0.9152261002780916}, {'alpha': 0.23423914662544418}, 
                      {'alpha': 0.09159332036121723}, {'alpha': 0.06084642077147053}],
            'metric':np.array([0.216456292811124, 0.21668294660495988, 0.2166890437876308,
                               0.21660255096339978, 0.2164189788190982, 0.2166571743908303,
                               0.21548741275543748, 0.2164733416179101, 0.2166378443581651,
                               0.21666500432810773]),                                                               
            'alpha' :np.array([0.24713734184878824, 0.03734897347801035, 0.028425159292991616, 
                               0.12665709946616685, 0.274750150868112, 0.07017992831138445, 
                               0.9152261002780916, 0.23423914662544418, 0.09159332036121723,
                               0.06084642077147053])
                                     },
            index=list(range(10))
                                   ).sort_values(by='metric', ascending=True)

    pd.testing.assert_frame_equal(results, expected_results)


def test_results_output_bayesian_search_optuna_ForecasterAutoregCustom_with_mocked():
    '''
    Test output of _bayesian_search_optuna in ForecasterAutoregCustom with mocked
    (mocked done in Skforecast v0.4.3).
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
    
    def search_space(trial):
        search_space  = {'alpha' : trial.suggest_loguniform('alpha', 1e-2, 1.0)
                        }
        return search_space

    results = _bayesian_search_optuna(
                    forecaster   = forecaster,
                    y            = y,
                    search_space = search_space,
                    steps        = steps,
                    metric       = 'mean_absolute_error',
                    refit        = True,
                    initial_train_size = len(y_train),
                    fixed_train_size   = True,
                    n_trials     = 10,
                    random_state = 123,
                    return_best  = False,
                    verbose      = False
              )[0]
    
    expected_results = pd.DataFrame({
            'lags'  :['custom predictors', 'custom predictors', 'custom predictors',
                      'custom predictors', 'custom predictors', 'custom predictors',
                      'custom predictors', 'custom predictors', 'custom predictors',
                      'custom predictors'],
            'params':[{'alpha': 0.24713734184878824}, {'alpha': 0.03734897347801035},
                      {'alpha': 0.028425159292991616}, {'alpha': 0.12665709946616685}, 
                      {'alpha': 0.274750150868112}, {'alpha': 0.07017992831138445}, 
                      {'alpha': 0.9152261002780916}, {'alpha': 0.23423914662544418},
                      {'alpha': 0.09159332036121723}, {'alpha': 0.06084642077147053}],
            'metric':np.array([0.216456292811124, 0.21668294660495988, 0.2166890437876308, 
                               0.21660255096339978, 0.21641897881909822, 0.2166571743908303, 
                               0.21548741275543748, 0.21647334161791013, 0.2166378443581651,
                               0.21666500432810773]),                                                               
            'alpha' :np.array([0.24713734184878824, 0.03734897347801035, 0.028425159292991616,
                               0.12665709946616685, 0.274750150868112, 0.07017992831138445, 
                               0.9152261002780916, 0.23423914662544418, 0.09159332036121723,
                               0.06084642077147053])
                                     },
            index=list(range(10))
                                   ).sort_values(by='metric', ascending=True)

    pd.testing.assert_frame_equal(results, expected_results)
    

def test_evaluate_bayesian_search_optuna_when_return_best():
    '''
    Test forecaster is refited when return_best=True in _bayesian_search_optuna.
    '''
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]
    
    def search_space(trial):
        search_space  = {'alpha' : trial.suggest_loguniform('alpha', 1e-2, 1.0)
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
        random_state = 123,
        return_best  = True,
        verbose      = False
    )
    
    expected_lags = np.array([1, 2])
    expected_alpha = 0.028425159292991616
    
    assert (expected_lags == forecaster.lags).all()
    assert expected_alpha == forecaster.regressor.alpha


def test_results_opt_best_output_bayesian_search_optuna_with_output_study_best_trial_optuna():
    '''
    Test results_opt_best output of _bayesian_search_optuna with output study.best_trial optuna.
    '''
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
            forecaster = forecaster,
            y          = y,
            steps      = steps,
            metric     = metric,
            initial_train_size = initial_train_size,
            fixed_train_size   = fixed_train_size,
            refit      = refit,
            verbose    = verbose,
        ) -> float:
        
        alpha = trial.suggest_loguniform('alpha', 1e-2, 1.0)
        
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
        search_space  = {'alpha' : trial.suggest_loguniform('alpha', 1e-2, 1.0)
                        }
        return search_space
    return_best  = False

    results_opt_best = _bayesian_search_optuna(
                            forecaster   = forecaster,
                            y            = y,
                            lags_grid    = lags_grid,
                            search_space = search_space,
                            steps        = steps,
                            metric       = metric,
                            refit        = refit,
                            initial_train_size = initial_train_size,
                            fixed_train_size   = fixed_train_size,
                            n_trials     = n_trials,
                            random_state = random_state,
                            return_best  = return_best,
                            verbose      = verbose
                       )[1]

    assert best_trial.number == results_opt_best.number
    assert best_trial.values == results_opt_best.values
    assert best_trial.params == results_opt_best.params