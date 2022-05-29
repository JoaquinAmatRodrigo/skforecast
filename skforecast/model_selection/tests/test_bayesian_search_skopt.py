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
    Test Exception is raised when search_space key name do not match the Space 
    object name from skopt.
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
        )


def test_results_output_bayesian_search_skopt_ForecasterAutoreg_with_mocked():
    '''
    Test output of _bayesian_search_skopt in ForecasterAutoreg with mocked
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
    search_space = {'n_estimators'    : Integer(10, 20, "uniform", name='n_estimators'),
                    'min_samples_leaf': Real(1., 3.7, "log-uniform", name='min_samples_leaf'),
                    'max_features'    : Categorical(['log2', 'sqrt'], name='max_features')
                   }

    results = _bayesian_search_skopt(
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
            'lags'  :[[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2],
                      [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                      [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            'params':[{'n_estimators': 17, 'min_samples_leaf': 1.7516926954624357, 'max_features': 'sqrt'},
                      {'n_estimators': 17, 'min_samples_leaf': 1.901317408816101, 'max_features': 'sqrt'}, 
                      {'n_estimators': 14, 'min_samples_leaf': 2.134928323868993, 'max_features': 'log2'}, 
                      {'n_estimators': 14, 'min_samples_leaf': 2.272179337814922, 'max_features': 'log2'}, 
                      {'n_estimators': 12, 'min_samples_leaf': 2.481767480132848, 'max_features': 'sqrt'}, 
                      {'n_estimators': 16, 'min_samples_leaf': 1.778913732091379, 'max_features': 'log2'}, 
                      {'n_estimators': 17, 'min_samples_leaf': 1.7503011299565185, 'max_features': 'log2'}, 
                      {'n_estimators': 15, 'min_samples_leaf': 2.634132917484797, 'max_features': 'log2'}, 
                      {'n_estimators': 14, 'min_samples_leaf': 2.3551239865216638, 'max_features': 'log2'}, 
                      {'n_estimators': 12, 'min_samples_leaf': 3.6423411901787532, 'max_features': 'sqrt'}, 
                      {'n_estimators': 17, 'min_samples_leaf': 1.7516926954624357, 'max_features': 'sqrt'}, 
                      {'n_estimators': 17, 'min_samples_leaf': 1.901317408816101, 'max_features': 'sqrt'}, 
                      {'n_estimators': 14, 'min_samples_leaf': 2.134928323868993, 'max_features': 'log2'},
                      {'n_estimators': 14, 'min_samples_leaf': 2.272179337814922, 'max_features': 'log2'},
                      {'n_estimators': 12, 'min_samples_leaf': 2.481767480132848, 'max_features': 'sqrt'},
                      {'n_estimators': 16, 'min_samples_leaf': 1.778913732091379, 'max_features': 'log2'}, 
                      {'n_estimators': 17, 'min_samples_leaf': 1.7503011299565185, 'max_features': 'log2'},
                      {'n_estimators': 15, 'min_samples_leaf': 2.634132917484797, 'max_features': 'log2'}, 
                      {'n_estimators': 14, 'min_samples_leaf': 2.3551239865216638, 'max_features': 'log2'},
                      {'n_estimators': 12, 'min_samples_leaf': 3.6423411901787532, 'max_features': 'sqrt'}],
            'metric':np.array([0.21433589521377985, 0.21433589521377985, 0.21661388810185186, 
                               0.21661388810185186, 0.21685103020640437, 0.2152253922034143, 
                               0.21433589521377985, 0.21479600790277778, 0.21661388810185186,
                               0.21685103020640437, 0.20997867726211075, 0.20997867726211075,
                               0.20952958306022407, 0.20952958306022407, 0.20980140118464055,
                               0.21045872919117656, 0.20997867726211075, 0.20840381228758173, 
                               0.20952958306022407, 0.20980140118464055]),                                                               
            'n_estimators' :np.array([17, 17, 14, 14, 12, 16, 17, 15, 14, 12, 
                                      17, 17, 14, 14, 12, 16, 17, 15, 14, 12]),
            'min_samples_leaf' :np.array([1.7516926954624357, 1.901317408816101, 2.134928323868993, 
                                   2.272179337814922, 2.481767480132848, 1.778913732091379,
                                   1.7503011299565185, 2.634132917484797, 2.3551239865216638,
                                   3.6423411901787532, 1.7516926954624357, 1.901317408816101,
                                   2.134928323868993, 2.272179337814922, 2.481767480132848, 
                                   1.778913732091379, 1.7503011299565185, 2.634132917484797, 
                                   2.3551239865216638, 3.6423411901787532]),
            'max_features' :['sqrt', 'sqrt', 'log2', 'log2', 'sqrt', 'log2', 'log2', 'log2', 'log2',
                             'sqrt', 'sqrt', 'sqrt', 'log2', 'log2', 'sqrt', 'log2', 'log2', 'log2', 
                             'log2', 'sqrt']
                                     },
            index=list(range(20))
                                   ).sort_values(by='metric', ascending=True)

    pd.testing.assert_frame_equal(results, expected_results, check_dtype=False)
    

def test_results_output_bayesian_search_skopt_ForecasterAutoreg_with_mocked_when_kwargs_gp_minimize():
    '''
    Test output of _bayesian_search_skopt in ForecasterAutoreg when kwargs_gp_minimize with mocked
    (mocked done in Skforecast v0.4.3).
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

    # kwargs_gp_minimize
    initial_point_generator = 'lhs'
    kappa = 1.8

    results = _bayesian_search_skopt(
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
                    kwargs_gp_minimize = {'initial_point_generator': initial_point_generator,
                                          'kappa': kappa }
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
    search_space = {'alpha': Real(0.01, 1.0, "log-uniform", name='alpha')}

    results = _bayesian_search_skopt(
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
    Test forecaster is refited when return_best=True in _bayesian_search_skopt.
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
        verbose            = False
    )
    
    expected_lags = np.array([1, 2])
    expected_alpha = 0.019050287104581624
    
    assert (expected_lags == forecaster.lags).all()
    assert expected_alpha == forecaster.regressor.alpha


def test_results_opt_best_output_bayesian_search_skopt_with_output_gp_minimize_skopt():
    '''
    Test results_opt_best output of _bayesian_search_skopt with output gp_minimize() skopt.
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
                            random_state       = random_state,
                            return_best        = return_best,
                            verbose            = verbose,
                            kwargs_gp_minimize = {}
                       )[1]

    assert results_opt.x == results_opt_best.x
    assert results_opt.fun == results_opt_best.fun