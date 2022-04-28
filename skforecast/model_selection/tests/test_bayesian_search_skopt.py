# Unit test _bayesian_search_skopt
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from skopt.space import Categorical, Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection.model_selection import _bayesian_search_skopt

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


def test_bayesian_search_skopt_exception_when_search_space_names_do_not_match():
    '''
    Test Exception is raised when search_space key name do not match the Space object name from skopt.
    '''
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]
    search_space = {'not_alpha': Real(0.01, 1.0, "log-uniform", name='alpha')}
    
    with pytest.raises(Exception):
        _bayesian_search_skopt(
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


def test_results_output_bayesian_search_skopt_ForecasterAutoreg_with_mocked():
    '''
    Test output of _bayesian_search_skopt in ForecasterAutoreg with mocked
    (mocked done in Skforecast v0.4.3)
    '''
    forecaster = ForecasterAutoreg(
                    regressor = RandomForestRegressor(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]
    search_space = {'n_estimators': Integer(10, 20, "uniform", name='n_estimators'),
                    'max_depth': Real(1, 5, "log-uniform", name='max_depth'),
                    'max_features': Categorical(['auto', 'sqrt'], name='max_features')
                   }

    results = _bayesian_search_skopt(
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
            'params':[{'n_estimators': 17, 'max_depth': 1.9929129312200498, 'max_features': 'sqrt'}, 
                      {'n_estimators': 17, 'max_depth': 2.2043340187845697, 'max_features': 'sqrt'}, 
                      {'n_estimators': 14, 'max_depth': 2.5420783112854197, 'max_features': 'auto'}, 
                      {'n_estimators': 14, 'max_depth': 2.7445792319355813, 'max_features': 'auto'}, 
                      {'n_estimators': 12, 'max_depth': 3.059236337842803, 'max_features': 'sqrt'}, 
                      {'n_estimators': 16, 'max_depth': 2.0310778111301357, 'max_features': 'auto'}, 
                      {'n_estimators': 17, 'max_depth': 1.9909655528496835, 'max_features': 'auto'}, 
                      {'n_estimators': 15, 'max_depth': 3.29188739864399, 'max_features': 'auto'}, 
                      {'n_estimators': 14, 'max_depth': 2.8683395097937403, 'max_features': 'auto'},
                      {'n_estimators': 12, 'max_depth': 4.904323050812992, 'max_features': 'sqrt'},
                      {'n_estimators': 17, 'max_depth': 1.9929129312200498, 'max_features': 'sqrt'}, 
                      {'n_estimators': 17, 'max_depth': 2.2043340187845697, 'max_features': 'sqrt'}, 
                      {'n_estimators': 14, 'max_depth': 2.5420783112854197, 'max_features': 'auto'}, 
                      {'n_estimators': 14, 'max_depth': 2.7445792319355813, 'max_features': 'auto'},
                      {'n_estimators': 12, 'max_depth': 3.059236337842803, 'max_features': 'sqrt'}, 
                      {'n_estimators': 16, 'max_depth': 2.0310778111301357, 'max_features': 'auto'}, 
                      {'n_estimators': 17, 'max_depth': 1.9909655528496835, 'max_features': 'auto'}, 
                      {'n_estimators': 15, 'max_depth': 3.29188739864399, 'max_features': 'auto'}, 
                      {'n_estimators': 14, 'max_depth': 2.8683395097937403, 'max_features': 'auto'}, 
                      {'n_estimators': 12, 'max_depth': 4.904323050812992, 'max_features': 'sqrt'}],
            'metric':np.array([0.21615799463348997, 0.21704325818847112, 0.227837004285555,
                               0.227837004285555, 0.22228329404011593, 0.22331462080401032, 
                               0.22474421769386224, 0.21041138481130603, 0.227837004285555, 
                               0.20541235571650687, 0.2198566352889403, 0.21368735246085513, 
                               0.23578208465562722, 0.23578208465562722, 0.21856857925536957,
                               0.23649308193173593, 0.22528691869878895, 0.22001004752280182,
                               0.23578208465562722, 0.22606118006271245]),                                                               
            'n_estimators' :np.array([17, 17, 14, 14, 12, 16, 17, 15, 14, 12, 17,
                                      17, 14, 14, 12, 16, 17, 15, 14, 12]),
            'max_depth' :np.array([1.9929129312200498, 2.2043340187845697, 2.5420783112854197, 
                                   2.7445792319355813, 3.059236337842803, 2.0310778111301357, 
                                   1.9909655528496835, 3.29188739864399, 2.8683395097937403,
                                   4.904323050812992, 1.9929129312200498, 2.2043340187845697, 
                                   2.5420783112854197, 2.7445792319355813, 3.059236337842803,
                                   2.0310778111301357, 1.9909655528496835, 3.29188739864399,
                                   2.8683395097937403, 4.904323050812992]),
            'max_features' :['sqrt', 'sqrt', 'auto', 'auto', 'sqrt', 'auto', 'auto', 
                             'auto', 'auto', 'sqrt', 'sqrt', 'sqrt', 'auto', 'auto', 
                             'sqrt', 'auto', 'auto', 'auto', 'auto', 'sqrt']
                                     },
            index=list(range(20))
                                   ).sort_values(by='metric', ascending=True)

    pd.testing.assert_frame_equal(results, expected_results)
    

def test_results_output_bayesian_search_skopt_ForecasterAutoreg_with_mocked_when_args_kwargs():
    '''
    Test output of _bayesian_search_skopt in ForecasterAutoreg when args and kwargs with mocked
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
    search_space = {'alpha': Real(0.01, 1.0, "log-uniform", name='alpha')}

    # *args and **kwargs for gp_minimize()
    initial_point_generator = 'lhs'
    kappa = 1.8

    results = _bayesian_search_skopt(
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
                    initial_point_generator = initial_point_generator,
                    kappa = kappa
              )[0]
    
    expected_results = pd.DataFrame({
            'lags'  :[[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2],
                      [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                      [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            'params':[{'alpha': 0.016838723959617538}, {'alpha': 0.10033990027966379}, 
                      {'alpha': 0.30984231371002086}, {'alpha': 0.02523894961617201},
                      {'alpha': 0.06431449919265146}, {'alpha': 0.04428828255962529},
                      {'alpha': 0.7862467218336935}, {'alpha': 0.21382904045131165},
                      {'alpha': 0.4646709105348175}, {'alpha': 0.01124059864722814}, 
                      {'alpha': 0.016838723959617538}, {'alpha': 0.10033990027966379},
                      {'alpha': 0.30984231371002086}, {'alpha': 0.02523894961617201},
                      {'alpha': 0.06431449919265146}, {'alpha': 0.04428828255962529},
                      {'alpha': 0.7862467218336935}, {'alpha': 0.21382904045131165},
                      {'alpha': 0.4646709105348175}, {'alpha': 0.01124059864722814}],
            'metric':np.array([0.21183497939493612, 0.2120677429498087, 0.2125445833833647,
                               0.21185973952472195, 0.21197085675244506, 0.21191472647731882, 
                               0.2132707683116569, 0.21234254975249803, 0.21282383637032143, 
                               0.21181829991953996, 0.21669632191054566, 0.21662944006267573,
                               0.21637019858109752, 0.2166911187311533, 0.2166621393072383,
                               0.21667792427267493, 0.21566880163156743, 0.21649975575675726, 
                               0.21614409053015884, 0.21669956732317974]),                                                               
            'alpha' :np.array([0.016838723959617538, 0.10033990027966379, 0.30984231371002086,
                               0.02523894961617201, 0.06431449919265146, 0.04428828255962529,
                               0.7862467218336935, 0.21382904045131165, 0.4646709105348175,
                               0.01124059864722814, 0.016838723959617538, 0.10033990027966379,
                               0.30984231371002086, 0.02523894961617201, 0.06431449919265146,
                               0.04428828255962529, 0.7862467218336935, 0.21382904045131165,
                               0.4646709105348175, 0.01124059864722814])
                                     },
            index=list(range(20))
                                   ).sort_values(by='metric', ascending=True)

    pd.testing.assert_frame_equal(results, expected_results)


def test_results_output_bayesian_search_skopt_ForecasterAutoreg_with_mocked_when_lags_grid_is_None():
    '''
    Test output of _bayesian_search_skopt in ForecasterAutoreg when lags_grid is None with mocked
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
    search_space = {'alpha': Real(0.01, 1.0, "log-uniform", name='alpha')}

    results = _bayesian_search_skopt(
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
            'params':[{'alpha': 0.26663099972129245}, {'alpha': 0.07193526575307788},
                      {'alpha': 0.24086278856848584}, {'alpha': 0.27434725570656354},
                      {'alpha': 0.0959926247515687}, {'alpha': 0.3631244766604131},
                      {'alpha': 0.06635119445083354}, {'alpha': 0.14434062917737708},
                      {'alpha': 0.019050287104581624}, {'alpha': 0.0633920962590419}],
            'metric':np.array([0.21643005790510492, 0.21665565996188138, 0.2164646190462156,
                               0.21641953058020516, 0.21663365234334242, 0.2162939165190013, 
                               0.21666043214039407, 0.21658325961136823, 0.21669499028423744, 
                               0.21666290650172168]),                                                               
            'alpha' :np.array([0.26663099972129245, 0.07193526575307788, 0.24086278856848584, 
                               0.27434725570656354, 0.0959926247515687, 0.3631244766604131,
                               0.06635119445083354, 0.14434062917737708, 0.019050287104581624, 
                               0.0633920962590419])
                                     },
            index=list(range(10))
                                   ).sort_values(by='metric', ascending=True)

    pd.testing.assert_frame_equal(results, expected_results)


def test_results_output_bayesian_search_skopt_ForecasterAutoregCustom_with_mocked():
    '''
    Test output of _bayesian_search_skopt in ForecasterAutoregCustom with mocked
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
    search_space = {'alpha': Real(0.01, 1.0, "log-uniform", name='alpha')}

    results = _bayesian_search_skopt(
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
            'params':[{'alpha': 0.26663099972129245}, {'alpha': 0.07193526575307788},
                      {'alpha': 0.24086278856848584}, {'alpha': 0.27434725570656354},
                      {'alpha': 0.0959926247515687}, {'alpha': 0.3631244766604131},
                      {'alpha': 0.06635119445083354}, {'alpha': 0.14434062917737708},
                      {'alpha': 0.019050287104581624}, {'alpha': 0.0633920962590419}],
            'metric':np.array([0.21643005790510492, 0.21665565996188138, 0.2164646190462156,
                               0.21641953058020516, 0.21663365234334242, 0.2162939165190013, 
                               0.21666043214039407, 0.21658325961136823, 0.21669499028423744, 
                               0.21666290650172168]),                                                               
            'alpha' :np.array([0.26663099972129245, 0.07193526575307788, 0.24086278856848584, 
                               0.27434725570656354, 0.0959926247515687, 0.3631244766604131,
                               0.06635119445083354, 0.14434062917737708, 0.019050287104581624, 
                               0.0633920962590419])
                                     },
            index=list(range(10))
                                   ).sort_values(by='metric', ascending=True)

    pd.testing.assert_frame_equal(results, expected_results)
    

def test_evaluate_bayesian_search_skopt_when_return_best():
    '''
    Test forecaster is refited when return_best=True in _bayesian_search_skopt
    '''
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]
    search_space = {'alpha': Real(0.01, 1.0, "log-uniform", name='alpha')}

    _bayesian_search_skopt(
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
    expected_alpha = 0.019050287104581624
    
    assert (expected_lags == forecaster.lags).all()
    assert expected_alpha == forecaster.regressor.alpha


def test_results_opt_best_output_bayesian_search_skopt_with_output_gp_minimize_skopt():
    '''
    Test results_opt_best output of _bayesian_search_skopt with output gp_minimize() skopt
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

    search_space = [Real(0.01, 1.0, "log-uniform", name='alpha')]
    n_trials = 10
    random_state = 123

    @use_named_args(search_space)
    def objective(
        forecaster = forecaster,
        y          = y,
        steps      = steps,
        metric     = metric,
        initial_train_size = initial_train_size,
        fixed_train_size   = fixed_train_size,
        refit      = refit,
        verbose    = verbose,
        **params
    ) -> float:

        forecaster.set_params(**params)

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

    results_opt = gp_minimize(
                        func         = objective,
                        dimensions   = search_space,
                        n_calls      = n_trials,
                        random_state = random_state
                  )

    lags_grid = [4, 2]
    search_space = {'alpha': Real(0.01, 1.0, "log-uniform", name='alpha')}
    return_best  = False

    results_opt_best = _bayesian_search_skopt(
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

    assert results_opt.x == results_opt_best.x
    assert results_opt.fun == results_opt_best.fun