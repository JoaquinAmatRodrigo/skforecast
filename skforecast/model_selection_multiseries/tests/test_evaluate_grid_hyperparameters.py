# Unit test _evaluate_grid_hyperparameters_multiseries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from skforecast.model_selection_multiseries.model_selection_multiseries import _evaluate_grid_hyperparameters_multiseries

from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True) # hide progress bar

# Fixtures
# np.random.seed(123)
# series_1 = np.random.rand(50)
# series_2 = np.random.rand(50)
series = pd.DataFrame({'l1': pd.Series(np.array(
                                [0.69646919, 0.28613933, 0.22685145, 0.55131477, 0.71946897,
                                 0.42310646, 0.9807642 , 0.68482974, 0.4809319 , 0.39211752,
                                 0.34317802, 0.72904971, 0.43857224, 0.0596779 , 0.39804426,
                                 0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759,
                                 0.63440096, 0.84943179, 0.72445532, 0.61102351, 0.72244338,
                                 0.32295891, 0.36178866, 0.22826323, 0.29371405, 0.63097612,
                                 0.09210494, 0.43370117, 0.43086276, 0.4936851 , 0.42583029,
                                 0.31226122, 0.42635131, 0.89338916, 0.94416002, 0.50183668,
                                 0.62395295, 0.1156184 , 0.31728548, 0.41482621, 0.86630916,
                                 0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453]
                                      )
                            ), 
                       'l2': pd.Series(np.array(
                                [0.12062867, 0.8263408 , 0.60306013, 0.54506801, 0.34276383,
                                 0.30412079, 0.41702221, 0.68130077, 0.87545684, 0.51042234,
                                 0.66931378, 0.58593655, 0.6249035 , 0.67468905, 0.84234244,
                                 0.08319499, 0.76368284, 0.24366637, 0.19422296, 0.57245696,
                                 0.09571252, 0.88532683, 0.62724897, 0.72341636, 0.01612921,
                                 0.59443188, 0.55678519, 0.15895964, 0.15307052, 0.69552953,
                                 0.31876643, 0.6919703 , 0.55438325, 0.38895057, 0.92513249,
                                 0.84167   , 0.35739757, 0.04359146, 0.30476807, 0.39818568,
                                 0.70495883, 0.99535848, 0.35591487, 0.76254781, 0.59317692,
                                 0.6917018 , 0.15112745, 0.39887629, 0.2408559 , 0.34345601]
                                      )
                            )
                      }
         )


def test_exception_evaluate_grid_hyperparameters_multiseries_when_return_best_and_len_series_exog_different():
    """
    Test exception is raised in _evaluate_grid_hyperparameters_multiseries when 
    `return_best = True` and length of `series` and `exog` do not match.
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 3
                 )
    exog = series.iloc[:30, 0]

    err_msg = re.escape(
            f'`exog` must have same number of samples as `series`. '
            f'length `exog`: ({len(exog)}), length `series`: ({len(series)})'
        )
    with pytest.raises(ValueError, match = err_msg):
        _evaluate_grid_hyperparameters_multiseries(
            forecaster          = forecaster,
            series              = series,
            exog                = exog,
            param_grid          = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            steps               = 4,
            metric              = 'mean_absolute_error',
            initial_train_size  = 12,
            fixed_train_size    = False,
            levels              = None,
            lags_grid           = [2, 4],
            refit               = False,
            return_best         = True,
            verbose             = False
        )


def test_evaluate_grid_hyperparameters_multiseries_exception_when_levels_not_list_str_None():
    """
    Test Exception is raised in _evaluate_grid_hyperparameters_multiseries when 
    `levels` is not a `list`, `str` or `None`.
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 3
                 )

    levels = 1
    
    err_msg = re.escape(
                (f'`levels` must be a `list` of column names, a `str` '
                 f'of a column name or `None`.')
              )
    with pytest.raises(TypeError, match = err_msg):
        _evaluate_grid_hyperparameters_multiseries(
            forecaster          = forecaster,
            series              = series,
            param_grid          = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            steps               = 4,
            metric              = 'mean_absolute_error',
            initial_train_size  = 12,
            fixed_train_size    = False,
            levels              = levels,
            exog                = None,
            lags_grid           = [2, 4],
            refit               = False,
            return_best         = False,
            verbose             = False
        )


def test_evaluate_grid_hyperparameters_multiseries_multivariate_warning_when_levels():
    """
    Test Warning is raised when levels is not forecaster.level or None.
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor = Ridge(random_state=123),
                     level     = 'l1',
                     lags      = 2,
                     steps     = 3
                 )
    
    warn_msg = re.escape(
                (f"`levels` argument have no use when the forecaster is of type ForecasterAutoregMultiVariate. "
                 f"The level of this forecaster is {forecaster.level}, to predict another level, change the `level` "
                 f"argument when initializing the forecaster. \n")
            )
    with pytest.warns(UserWarning, match = warn_msg):
        _evaluate_grid_hyperparameters_multiseries(
            forecaster         = forecaster,
            series             = series,
            lags_grid          = [2, 4],
            param_grid         = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            steps              = 3,
            levels             = 'not_forecaster.level',
            metric             = 'mean_absolute_error',
            initial_train_size = 12,
            refit              = True,
            fixed_train_size   = False,
            exog               = None,
            return_best        = False,
            verbose            = False
        )


def test_evaluate_grid_hyperparameters_multiseries_exception_when_metric_list_duplicate_names():
    """
    Test exception is raised in _evaluate_grid_hyperparameters when a `list` of 
    metrics is used with duplicate names.
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 3
                 )
    
    err_msg = re.escape('When `metric` is a `list`, each metric name must be unique.')
    with pytest.raises(ValueError, match = err_msg):
        _evaluate_grid_hyperparameters_multiseries(
            forecaster          = forecaster,
            series              = series,
            param_grid          = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            steps               = 4,
            metric              = ['mean_absolute_error', mean_absolute_error],
            initial_train_size  = 12,
            fixed_train_size    = False,
            levels              = ['l1'],
            exog                = None,
            lags_grid           = [2, 4],
            refit               = False,
            return_best         = False,
            verbose             = False
        )

# ForecasterAutoregMultiSeries
# ======================================================================================================================
def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterAutoregMultiSeries_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in ForecasterAutoregMultiSeries 
    with mocked (mocked done in Skforecast v0.5.0).
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster          = forecaster,
                  series              = series,
                  param_grid          = param_grid,
                  steps               = steps,
                  metric              = 'mean_absolute_error',
                  initial_train_size  = len(series) - n_validation,
                  fixed_train_size    = False,
                  levels              = None,
                  exog                = None,
                  lags_grid           = lags_grid,
                  refit               = False,
                  return_best         = False,
                  verbose             = False
              )
    
    expected_results = pd.DataFrame({
            'levels':[['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2']],
            'lags'  :[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2], [1, 2], [1, 2]],
            'params':[{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, {'alpha': 1}, {'alpha': 0.1}, {'alpha': 0.01}],
            'mean_absolute_error':np.array([0.20968100463227382, 0.20969259779858337, 0.20977945312386406, 
                                            0.21077344827205086, 0.21078653113227208, 0.21078779824759553]),                                                               
            'alpha' :np.array([0.01, 0.1, 1., 1., 0.1, 0.01])
                                     },
            index=[3, 4, 5, 2, 1, 0]
                                   )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_ForecasterAutoregMultiSeries_lags_grid_is_None_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterAutoregMultiSeries 
    when `lags_grid` is `None` with mocked (mocked done in Skforecast v0.5.0), 
    should use forecaster.lags as lags_grid.
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )

    lags_grid = None
    steps = 3
    n_validation = 12
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster          = forecaster,
                  series              = series,
                  param_grid          = param_grid,
                  steps               = steps,
                  metric              = mean_absolute_error,
                  initial_train_size  = len(series) - n_validation,
                  fixed_train_size    = False,
                  levels              = ['l1', 'l2'],
                  exog                = None,
                  lags_grid           = lags_grid,
                  refit               = False,
                  return_best         = False,
                  verbose             = False
              )
    
    expected_results = pd.DataFrame({
            'levels':[['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2']],
            'lags'  :[[1, 2], [1, 2], [1, 2]],
            'params':[{'alpha': 1}, {'alpha': 0.1}, {'alpha': 0.01}],
            'mean_absolute_error':np.array([0.21077344827205086, 0.21078653113227208, 0.21078779824759553]),                                                               
            'alpha' :np.array([1., 0.1 , 0.01])
                                     },
            index=[2, 1, 0]
                                   )

    pd.testing.assert_frame_equal(results, expected_results)


@pytest.mark.parametrize("levels", 
                         ['l1', ['l1']], 
                         ids = lambda value : f'levels: {value}' )
def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterAutoregMultiSeries_levels_str_list_with_mocked(levels):
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in ForecasterAutoregMultiSeries 
    with mocked when `levels` is a `str` or a `list` (mocked done in Skforecast v0.5.0).
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster          = forecaster,
                  series              = series,
                  param_grid          = param_grid,
                  steps               = steps,
                  metric              = 'mean_absolute_error',
                  initial_train_size  = len(series) - n_validation,
                  fixed_train_size    = False,
                  levels              = levels,
                  exog                = None,
                  lags_grid           = lags_grid,
                  refit               = False,
                  return_best         = False,
                  verbose             = False
              )
    
    expected_results = pd.DataFrame({
            'levels':[['l1'], ['l1'], ['l1'], ['l1'], ['l1'], ['l1']],
            'lags'  :[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2], [1, 2], [1, 2]],
            'params':[{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, {'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            'mean_absolute_error':np.array([0.20669393332187616, 0.20671040715338015, 0.20684013292264494, 
                                            0.2073988652614679, 0.20741562577568792, 0.2075484707375347]),                                                               
            'alpha' :np.array([0.01, 0.1, 1., 0.01, 0.1, 1.])
                                     },
            index=[3, 4, 5, 0, 1, 2]
                                   )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterAutoregMultiSeries_multiple_metrics_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in ForecasterAutoregMultiSeries 
    with mocked when multiple metrics (mocked done in Skforecast v0.6.0).
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster          = forecaster,
                  series              = series,
                  param_grid          = param_grid,
                  steps               = steps,
                  metric              = [mean_squared_error, 'mean_absolute_error'],
                  initial_train_size  = len(series) - n_validation,
                  fixed_train_size    = False,
                  levels              = None,
                  exog                = None,
                  lags_grid           = lags_grid,
                  refit               = False,
                  return_best         = False,
                  verbose             = False
              )
    
    expected_results = pd.DataFrame({
            'levels':[['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2']],
            'lags'  :[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2], [1, 2], [1, 2]],
            'params':[{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, {'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            'mean_squared_error':np.array([0.06365397633008085, 0.06367614582294409, 0.06385378127252679, 
                                           0.06389613553855186, 0.06391570591810977, 0.06407787633532819]),
            'mean_absolute_error':np.array([0.20968100463227382, 0.20969259779858337, 0.20977945312386406, 
                                            0.21078779824759553, 0.21078653113227208, 0.21077344827205086]),
            'alpha' :np.array([0.01, 0.1, 1., 0.01, 0.1, 1.])
                                     },
            index=[3, 4, 5, 0, 1, 2]
                                   )

    pd.testing.assert_frame_equal(results, expected_results)


def test_evaluate_grid_hyperparameters_multiseries_when_return_best_ForecasterAutoregMultiSeries():
    """
    Test forecaster is refitted when `return_best = True` in 
    _evaluate_grid_hyperparameters_multiseries.
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    steps = 3
    n_validation = 12
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    _evaluate_grid_hyperparameters_multiseries(
        forecaster          = forecaster,
        series              = series,
        param_grid          = param_grid,
        steps               = steps,
        metric              = 'mean_absolute_error',
        initial_train_size  = len(series) - n_validation,
        fixed_train_size    = False,
        levels              = None,
        exog                = None,
        lags_grid           = lags_grid,
        refit               = False,
        return_best         = True,
        verbose             = False
    )

    expected_lags = np.array([1, 2, 3, 4])
    expected_alpha = 0.01
    expected_series_col_names = ['l1', 'l2']
    
    assert (expected_lags == forecaster.lags).all()
    assert expected_alpha == forecaster.regressor.alpha
    assert expected_series_col_names ==  forecaster.series_col_names


# ForecasterAutoregMultiVariate
# ======================================================================================================================
def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterAutoregMultiVariate_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in ForecasterAutoregMultiVariate 
    with mocked (mocked done in Skforecast v0.6.0).
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor = Ridge(random_state=123),
                     level     = 'l1',
                     lags      = 2,
                     steps     = 3
                 )

    steps = 3
    n_validation = 12
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster          = forecaster,
                  series              = series,
                  param_grid          = param_grid,
                  steps               = steps,
                  metric              = 'mean_absolute_error',
                  initial_train_size  = len(series) - n_validation,
                  fixed_train_size    = False,
                  levels              = None,
                  exog                = None,
                  lags_grid           = lags_grid,
                  refit               = False,
                  return_best         = False,
                  verbose             = True
              )
    
    expected_results = pd.DataFrame({
            'levels': [['l1'], ['l1'], ['l1'], ['l1'], ['l1'], ['l1']],
            'lags'  : [[1, 2], [1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, {'alpha': 1}, {'alpha': 0.1}, {'alpha': 0.01}],
            'mean_absolute_error': np.array([0.20115194, 0.20183032, 0.20566862, 0.22224269, 0.22625017, 0.22644284]),                                                               
            'alpha' : np.array([0.01, 0.1, 1., 1., 0.1, 0.01])
                                     },
            index=[0, 1, 2, 5, 4, 3]
                                   )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_ForecasterAutoregMultiVariate_lags_grid_is_None_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterAutoregMultiVariate 
    when `lags_grid` is `None` with mocked (mocked done in Skforecast v0.6.0), 
    should use forecaster.lags as lags_grid.
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor = Ridge(random_state=123),
                     level     = 'l1',
                     lags      = 2,
                     steps     = 3
                 )

    lags_grid = None
    steps = 3
    n_validation = 12
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster          = forecaster,
                  series              = series,
                  param_grid          = param_grid,
                  steps               = steps,
                  metric              = mean_absolute_error,
                  initial_train_size  = len(series) - n_validation,
                  fixed_train_size    = False,
                  levels              = None,
                  exog                = None,
                  lags_grid           = lags_grid,
                  refit               = False,
                  return_best         = False,
                  verbose             = False
              )
    
    expected_results = pd.DataFrame({
            'levels': [['l1'], ['l1'], ['l1']],
            'lags'  : [[1, 2], [1, 2], [1, 2]],
            'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1.}],
            'mean_absolute_error': np.array([0.20115194, 0.20183032, 0.20566862]),                                                               
            'alpha' : np.array([0.01, 0.1 , 1.])
                                     },
            index=[0, 1, 2]
                                   )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_ForecasterAutoregMultiVariate_lags_grid_is_dict():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterAutoregMultiVariate 
    when `lags_grid` is a dict with mocked (mocked done in Skforecast v0.6.0).
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor = Ridge(random_state=123),
                     level     = 'l1',
                     lags      = 2,
                     steps     = 3
                 )

    lags_grid = [{'l1': 2, 'l2': 3}, {'l1': [1, 3], 'l2': 3}, {'l1': 2, 'l2': [1, 4]}]
    steps = 3
    n_validation = 12
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster          = forecaster,
                  series              = series,
                  param_grid          = param_grid,
                  steps               = steps,
                  metric              = mean_absolute_error,
                  initial_train_size  = len(series) - n_validation,
                  fixed_train_size    = False,
                  levels              = None,
                  exog                = None,
                  lags_grid           = lags_grid,
                  refit               = False,
                  return_best         = False,
                  verbose             = False
              )
    
    expected_results = pd.DataFrame({
            'levels': [['l1'], ['l1'], ['l1'], ['l1'], ['l1'], ['l1'], ['l1'], ['l1'], ['l1']],
            'lags'  : [{'l1': np.array([1, 2]), 'l2': np.array([1, 2, 3])},
                       {'l1': np.array([1, 2]), 'l2': np.array([1, 2, 3])},
                       {'l1': np.array([1, 2]), 'l2': np.array([1, 2, 3])},
                       {'l1': np.array([1, 3]), 'l2': np.array([1, 2, 3])},
                       {'l1': np.array([1, 3]), 'l2': np.array([1, 2, 3])},
                       {'l1': np.array([1, 3]), 'l2': np.array([1, 2, 3])},
                       {'l1': np.array([1, 2]), 'l2': np.array([1, 4])},
                       {'l1': np.array([1, 2]), 'l2': np.array([1, 4])},
                       {'l1': np.array([1, 2]), 'l2': np.array([1, 4])}],
            'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1.}, 
                       {'alpha': 1.}  , {'alpha': 0.1}, {'alpha': 0.01}, 
                       {'alpha': 1.}  , {'alpha': 0.1}, {'alpha': 0.01}],
            'mean_absolute_error': np.array([0.2053202 , 0.20555199, 0.20677802, 0.21443621, 0.21801147,
                                             0.21863968, 0.22401526, 0.22830217, 0.22878132]),                                                               
            'alpha' : np.array([0.01, 0.1 , 1.  , 1.  , 0.1 , 0.01, 1.  , 0.1 , 0.01])
                                     },
            index=[0, 1, 2, 5, 4, 3, 8, 7, 6]
                                   )

    # Skip `lags` column because is not easy to compare, but checked when realizing the test
    pd.testing.assert_frame_equal(results.loc[:, results.columns != 'lags'], expected_results.loc[:, expected_results.columns != 'lags'])


def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterAutoregMultiVariate_multiple_metrics_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in ForecasterAutoregMultiVariate 
    with mocked when multiple metrics (mocked done in Skforecast v0.6.0).
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor = Ridge(random_state=123),
                     level     = 'l1',
                     lags      = 2,
                     steps     = 3
                 )

    steps = 3
    n_validation = 12
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster          = forecaster,
                  series              = series,
                  param_grid          = param_grid,
                  steps               = steps,
                  metric              = [mean_squared_error, 'mean_absolute_error'],
                  initial_train_size  = len(series) - n_validation,
                  fixed_train_size    = False,
                  levels              = None,
                  exog                = None,
                  lags_grid           = lags_grid,
                  refit               = False,
                  return_best         = False,
                  verbose             = False
              )
    
    expected_results = pd.DataFrame({
            'levels': [['l1'], ['l1'], ['l1'], ['l1'], ['l1'], ['l1']],
            'lags'  : [[1, 2], [1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, {'alpha': 1.}, {'alpha': 0.1}, {'alpha': 0.01}],
            'mean_squared_error': np.array([0.06260985, 0.06309219, 0.06627699, 0.08032378, 0.08400047, 0.08448937]),
            'mean_absolute_error': np.array([0.20115194, 0.20183032, 0.20566862, 0.22224269, 0.22625017, 0.22644284]),
            'alpha' : np.array([0.01, 0.1, 1., 1., 0.1, 0.01])
                                     },
            index=[0, 1, 2, 5, 4, 3]
                                   )

    pd.testing.assert_frame_equal(results, expected_results)


def test_evaluate_grid_hyperparameters_multiseries_when_return_best_ForecasterAutoregMultiVariate():
    """
    Test forecaster is refitted when `return_best = True` in 
    _evaluate_grid_hyperparameters_multiseries.
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor = Ridge(random_state=123),
                     level     = 'l1',
                     lags      = 2,
                     steps     = 3
                 )

    steps = 3
    n_validation = 12
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    _evaluate_grid_hyperparameters_multiseries(
        forecaster          = forecaster,
        series              = series,
        param_grid          = param_grid,
        steps               = steps,
        metric              = 'mean_absolute_error',
        initial_train_size  = len(series) - n_validation,
        fixed_train_size    = False,
        levels              = None,
        exog                = None,
        lags_grid           = lags_grid,
        refit               = False,
        return_best         = True,
        verbose             = False
    )

    expected_lags = np.array([1, 2])
    expected_alpha = 0.01
    expected_series_col_names = ['l1', 'l2']
    
    assert (expected_lags == forecaster.lags).all()
    for i in range(forecaster.steps):
        assert expected_alpha == forecaster.regressors_[i].alpha
    assert expected_series_col_names ==  forecaster.series_col_names