# Unit test _evaluate_grid_hyperparameters
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.model_selection.model_selection import _evaluate_grid_hyperparameters

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


def test_exception_evaluate_grid_hyperparameters_when_return_best_and_len_y_exog_different():
    """
    Test exception is raised in _evaluate_grid_hyperparameters when return_best 
    and length of `y` and `exog` do not match.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )
    exog = y[:30]

    err_msg = re.escape(
            f'`exog` must have same number of samples as `y`. '
            f'length `exog`: ({len(exog)}), length `y`: ({len(y)})'
        )
    with pytest.raises(ValueError, match = err_msg):
        _evaluate_grid_hyperparameters(
            forecaster  = forecaster,
            y           = y,
            exog        = exog,
            lags_grid   = [2, 4],
            param_grid  = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            steps       = 3,
            refit       = False,
            metric      = 'mean_absolute_error',
            initial_train_size = len(y[:-12]),
            fixed_train_size   = False,
            return_best = True,
            verbose     = False
        )


def test_exception_evaluate_grid_hyperparameters_metric_list_duplicate_names():
    """
    Test exception is raised in _evaluate_grid_hyperparameters when a `list` of 
    metrics is used with duplicate names.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    err_msg = re.escape('When `metric` is a `list`, each metric name must be unique.')
    with pytest.raises(ValueError, match = err_msg):
        _evaluate_grid_hyperparameters(
            forecaster  = forecaster,
            y           = y,
            lags_grid   = [2, 4],
            param_grid  = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            steps       = 3,
            refit       = False,
            metric      = ['mean_absolute_error', mean_absolute_error],
            initial_train_size = len(y[:-12]),
            fixed_train_size   = False,
            return_best = False,
            verbose     = False
        )


def test_output_evaluate_grid_hyperparameters_ForecasterAutoreg_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterAutoreg with mocked
    (mocked done in Skforecast v0.4.3).
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]
    idx = len(lags_grid)*len(param_grid)

    results = _evaluate_grid_hyperparameters(
                            forecaster  = forecaster,
                            y           = y,
                            lags_grid   = lags_grid,
                            param_grid  = param_grid,
                            steps       = steps,
                            refit       = False,
                            metric      = 'mean_squared_error',
                            initial_train_size = len(y_train),
                            fixed_train_size   = False,
                            return_best = False,
                            verbose     = False
              )
    
    expected_results = pd.DataFrame({
            'lags'  :[[1, 2], [1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            'params':[{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, {'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            'mean_squared_error':np.array([0.06464646, 0.06502362, 0.06745534, 0.06779272, 0.06802481, 0.06948609]),                                                               
            'alpha' :np.array([0.01, 0.1 , 1.  , 0.01, 0.1 , 1.  ])
                                     },
            index=pd.RangeIndex(start=0, stop=idx, step=1)
                                   )

    pd.testing.assert_frame_equal(results, expected_results)
    

def test_output_evaluate_grid_hyperparameters_ForecasterAutoreg_lags_grid_is_None_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterAutoreg when lags_grid is None with mocked
    (mocked done in Skforecast v0.4.3), should use forecaster.lags as lags_grid.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = None
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters(
                            forecaster  = forecaster,
                            y           = y,
                            lags_grid   = lags_grid,
                            param_grid  = param_grid,
                            steps       = steps,
                            refit       = False,
                            metric      = 'mean_squared_error',
                            initial_train_size = len(y_train),
                            fixed_train_size   = False,
                            return_best = False,
                            verbose     = False
              )
    
    expected_results = pd.DataFrame({
            'lags'  :[[1, 2], [1, 2], [1, 2]],
            'params':[{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            'mean_squared_error':np.array([0.06464646, 0.06502362, 0.06745534]),                                                               
            'alpha' :np.array([0.01, 0.1 , 1.])
                                     },
            index=pd.RangeIndex(start=0, stop=3, step=1)
                                   )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_ForecasterAutoreg_metric_list_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterAutoreg with mocked
    and multiple metrics (mocked done in Skforecast v0.4.3).
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]
    idx = len(lags_grid)*len(param_grid)

    results = _evaluate_grid_hyperparameters(
                            forecaster  = forecaster,
                            y           = y,
                            lags_grid   = lags_grid,
                            param_grid  = param_grid,
                            steps       = steps,
                            refit       = False,
                            metric      = ['mean_squared_error', mean_absolute_error],
                            initial_train_size = len(y_train),
                            fixed_train_size   = False,
                            return_best = False,
                            verbose     = False
              )
    
    expected_results = pd.DataFrame({
                            'lags'  : [[1, 2], [1, 2], [1, 2], [1, 2, 3, 4], 
                                       [1, 2, 3, 4], [1, 2, 3, 4]],
                            'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, 
                                       {'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
                            'mean_squared_error': np.array([0.06464646, 0.06502362, 0.06745534, 
                                                            0.06779272, 0.06802481, 0.06948609]),   
                            'mean_absolute_error': np.array([0.20278812, 0.20314819, 0.20519952, 
                                                             0.20601567, 0.206323, 0.20747017]),                                                          
                            'alpha': np.array([0.01, 0.1 , 1.  , 0.01, 0.1 , 1.  ])},
                            index = pd.RangeIndex(start=0, stop=idx, step=1)
                       )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_ForecasterAutoregCustom_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterAutoregCustom with mocked
    (mocked done in Skforecast v0.4.3).
    """
    def create_predictors(y):
        """
        Create first 4 lags of a time series, used in ForecasterAutoregCustom.
        """

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
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]
    idx = len(param_grid)

    results = _evaluate_grid_hyperparameters(
                            forecaster  = forecaster,
                            y           = y,
                            param_grid  = param_grid,
                            steps       = steps,
                            refit       = False,
                            metric      = 'mean_squared_error',
                            initial_train_size = len(y_train),
                            fixed_train_size   = False,
                            return_best = False,
                            verbose     = False
              )
    
    expected_results = pd.DataFrame({
            'lags'  :['custom predictors', 'custom predictors', 'custom predictors'],
            'params':[{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            'mean_squared_error':np.array([0.06779272, 0.06802481, 0.06948609]),                                                               
            'alpha' :np.array([0.01, 0.1 , 1.])
                                     },
            index=pd.RangeIndex(start=0, stop=idx, step=1)
                                   )
    
    pd.testing.assert_frame_equal(results, expected_results)
    

def test_evaluate_grid_hyperparameters_when_return_best():
    """
    Test forecaster is refitted when return_best=True in _evaluate_grid_hyperparameters.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    _evaluate_grid_hyperparameters(
            forecaster  = forecaster,
            y           = y,
            lags_grid   = lags_grid,
            param_grid  = param_grid,
            steps       = steps,
            refit       = False,
            metric      = 'mean_squared_error',
            initial_train_size = len(y_train),
            fixed_train_size   = False,
            return_best = True,
            verbose     = False
    )
    
    expected_lags = np.array([1, 2])
    expected_alpha = 0.01
    
    assert (expected_lags == forecaster.lags).all()
    assert expected_alpha == forecaster.regressor.alpha


def test_evaluate_grid_hyperparameters_when_return_best_and_list_metrics():
    """
    Test forecaster is refitted when return_best=True in _evaluate_grid_hyperparameters
    and multiple metrics.
    """
    forecaster = ForecasterAutoreg(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    _evaluate_grid_hyperparameters(
            forecaster  = forecaster,
            y           = y,
            lags_grid   = lags_grid,
            param_grid  = param_grid,
            steps       = steps,
            refit       = False,
            metric      = [mean_absolute_percentage_error, 'mean_squared_error'],
            initial_train_size = len(y_train),
            fixed_train_size   = False,
            return_best = True,
            verbose     = False
    )
    
    expected_lags = np.array([1, 2])
    expected_alpha = 1.
    
    assert (expected_lags == forecaster.lags).all()
    assert expected_alpha == forecaster.regressor.alpha