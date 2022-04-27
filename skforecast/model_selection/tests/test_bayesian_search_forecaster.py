# Unit test bayesian_search_forecaster
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from skopt.space import Real
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.model_selection import bayesian_search_forecaster

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


def test_bayesian_search_forecaster_exception_when_search_space_names_do_not_match():
    '''
    Test Exception in bayesian_search_forecaster is raised when engine is not 'optuna' 
    or 'skopt'.
    '''
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    engine = 'not_optuna_or_skopt'
    
    with pytest.raises(Exception):
        bayesian_search_forecaster(
            forecaster   = forecaster,
            y            = y,
            lags_grid    = [2, 4],
            search_space = {'not_alpha': Real(0.01, 1.0, "log-uniform", name='alpha')},
            steps        = 3,
            metric       = 'mean_absolute_error',
            refit        = True,
            initial_train_size = len(y[:-12]),
            fixed_train_size   = True,
            n_trials     = 10,
            random_state = 123,
            return_best  = False,
            verbose      = False,
            engine       = engine
        )


@pytest.mark.skip(reason="not developed yet")
def test_results_output_bayesian_search_forecaster_engine_optuna_ForecasterAutoregCustom_with_mocked():
    '''
    Test output of bayesian_search_forecaster in ForecasterAutoregCustom with mocked
    using optuna engine (mocked done in Skforecast v0.4.3)
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

    engine = 'optuna'

    results = bayesian_search_forecaster(
                    forecaster   = forecaster,
                    y            = y,
                    search_space = {'alpha': Real(0.01, 1.0, "log-uniform", name='alpha')},
                    steps        = 3,
                    metric       = 'mean_absolute_error',
                    refit        = True,
                    initial_train_size = len(y[:-12]),
                    fixed_train_size   = True,
                    n_trials     = 10,
                    random_state = 123,
                    return_best  = False,
                    verbose      = False,
                    engine       = engine
              )[0]
    
    ###################
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


def test_results_output_bayesian_search_forecaster_engine_skopt_ForecasterAutoregCustom_with_mocked():
    '''
    Test output of bayesian_search_forecaster in ForecasterAutoregCustom with mocked
    using skopt engine (mocked done in Skforecast v0.4.3)
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

    engine = 'skopt'

    results = bayesian_search_forecaster(
                    forecaster   = forecaster,
                    y            = y,
                    search_space = {'alpha': Real(0.01, 1.0, "log-uniform", name='alpha')},
                    steps        = 3,
                    metric       = 'mean_absolute_error',
                    refit        = True,
                    initial_train_size = len(y[:-12]),
                    fixed_train_size   = True,
                    n_trials     = 10,
                    random_state = 123,
                    return_best  = False,
                    verbose      = False,
                    engine       = engine
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