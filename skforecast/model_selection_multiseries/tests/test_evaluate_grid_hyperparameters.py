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
from skforecast.model_selection_multiseries.model_selection_multiseries import _evaluate_grid_hyperparameters_multiseries

from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True) # hide progress bar

# Fixtures
# np.random.seed(123)
# series_1 = np.random.rand(50)
# series_2 = np.random.rand(50)
series = pd.DataFrame({'1': pd.Series(np.array(
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
                       '2': pd.Series(np.array(
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


def test_evaluate_grid_hyperparameters_multiseries_exception_when_levels_list_not_list_str_None():
    """
    Test Exception is raised in _evaluate_grid_hyperparameters_multiseries when 
    levels_list is not a list or str or None.
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 3
                 )

    levels_list = 1
    
    err_msg = re.escape(
                (f'`levels_list` must be a `list` of column names, a `str` '
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
            levels_list         = levels_list,
            levels_weights      = None,
            exog                = None,
            lags_grid           = [2, 4],
            refit               = False,
            return_best         = False,
            verbose             = False
        )


def test_evaluate_grid_hyperparameters_multiseries_exception_when_levels_weights_not_dict():
    """
    Test Exception is raised in _evaluate_grid_hyperparameters_multiseries when 
    levels_weights is not a dict.
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 3
                 )

    levels_weights = 'not_a_dict'
    
    err_msg = re.escape(f'`levels_weights` must be a `dict` or `None`.')
    with pytest.raises(TypeError, match = err_msg):
        _evaluate_grid_hyperparameters_multiseries(
            forecaster          = forecaster,
            series              = series,
            param_grid          = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            steps               = 4,
            metric              = 'mean_absolute_error',
            initial_train_size  = 12,
            fixed_train_size    = False,
            levels_list         = None,
            levels_weights      = levels_weights,
            exog                = None,
            lags_grid           = [2, 4],
            refit               = False,
            return_best         = False,
            verbose             = False
        )


def test_evaluate_grid_hyperparameters_multiseries_exception_when_metric_is_a_list():
    """
    Test Exception is raised in _evaluate_grid_hyperparameters_multiseries when 
    metric is a list (Not implemented yet).
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 3
                 )
    
    err_msg = re.escape(
                (f'The calculation of a list of metrics is not yet implemented '
                 f'in the multi-series forecast hyperparameter search.')
              )
    with pytest.raises(TypeError, match = err_msg):
        _evaluate_grid_hyperparameters_multiseries(
            forecaster          = forecaster,
            series              = series,
            param_grid          = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            steps               = 4,
            metric              = ['mean_absolute_error', mean_squared_error],
            initial_train_size  = 12,
            fixed_train_size    = False,
            levels_list         = '1',
            levels_weights      = None,
            exog                = None,
            lags_grid           = [2, 4],
            refit               = False,
            return_best         = False,
            verbose             = False
        )


def test_evaluate_grid_hyperparameters_multiseries_exception_when_levels_list_and_levels_weights_not_match():
    """
    Test Exception is raised in _evaluate_grid_hyperparameters_multiseries when 
    levels_weights are different from column names of series, `levels_list`.
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 3
                 )

    levels_list = ['1', '2']
    levels_weights = {'1': 0.5, 'not_2': 0.5}
    
    err_msg = re.escape(
                (f'`levels_weights` keys must be the same as the column '
                 f'names of series, `levels_list`.')
              )
    with pytest.raises(ValueError, match = err_msg):
        _evaluate_grid_hyperparameters_multiseries(
            forecaster          = forecaster,
            series              = series,
            param_grid          = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            steps               = 4,
            metric              = 'mean_absolute_error',
            initial_train_size  = 12,
            fixed_train_size    = False,
            levels_list         = levels_list,
            levels_weights      = levels_weights,
            exog                = None,
            lags_grid           = [2, 4],
            refit               = False,
            return_best         = False,
            verbose             = False
        )


def test_evaluate_grid_hyperparameters_multiseries_exception_when_levels_weights_not_sum_1():
    """
    Test Exception is raised in _evaluate_grid_hyperparameters_multiseries when 
    levels_weights do not add up to 1.0.
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 3
                 )

    levels_weights = {'1': 0.5, '2': 0.6}
    
    err_msg = re.escape(f'Weights in `levels_weights` must add up to 1.0.')
    with pytest.raises(ValueError,  match = err_msg):
        _evaluate_grid_hyperparameters_multiseries(
            forecaster          = forecaster,
            series              = series,
            param_grid          = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            steps               = 4,
            metric              = 'mean_absolute_error',
            initial_train_size  = 12,
            fixed_train_size    = False,
            levels_list         = None,
            levels_weights      = levels_weights,
            exog                = None,
            lags_grid           = [2, 4],
            refit               = False,
            return_best         = False,
            verbose             = False
        )


def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterAutoregMultiSeries_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in ForecasterAutoregMultiSeries with mocked
    (mocked done in Skforecast v0.5.0).
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
                    levels_list         = None,
                    levels_weights      = None,
                    exog                = None,
                    lags_grid           = lags_grid,
                    refit               = False,
                    return_best         = False,
                    verbose             = False
              )
    
    expected_results = pd.DataFrame({
            'levels':[['1', '2'], ['1', '2'], ['1', '2'], ['1', '2'], ['1', '2'], ['1', '2']],
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
    when lags_grid is None with mocked (mocked done in Skforecast v0.5.0), 
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
                    levels_list         = None,
                    levels_weights      = None,
                    exog                = None,
                    lags_grid           = lags_grid,
                    refit               = False,
                    return_best         = False,
                    verbose             = False
              )
    
    expected_results = pd.DataFrame({
            'levels':[['1', '2'], ['1', '2'], ['1', '2']],
            'lags'  :[[1, 2], [1, 2], [1, 2]],
            'params':[{'alpha': 1}, {'alpha': 0.1}, {'alpha': 0.01}],
            'mean_absolute_error':np.array([0.21077344827205086, 0.21078653113227208, 0.21078779824759553]),                                                               
            'alpha' :np.array([1., 0.1 , 0.01])
                                     },
            index=[2, 1, 0]
                                   )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterAutoregMultiSeries_levels_list_str_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in ForecasterAutoregMultiSeries with mocked
    when levels_list str (mocked done in Skforecast v0.5.0).
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    levels_list = '1'
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
                    levels_list         = levels_list,
                    levels_weights      = None,
                    exog                = None,
                    lags_grid           = lags_grid,
                    refit               = False,
                    return_best         = False,
                    verbose             = False
              )
    
    expected_results = pd.DataFrame({
            'levels':[['1'], ['1'], ['1'], ['1'], ['1'], ['1']],
            'lags'  :[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2], [1, 2], [1, 2]],
            'params':[{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, {'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            'mean_absolute_error':np.array([0.20669393332187616, 0.20671040715338015, 0.20684013292264494, 
                                            0.2073988652614679, 0.20741562577568792, 0.2075484707375347]),                                                               
            'alpha' :np.array([0.01, 0.1, 1., 0.01, 0.1, 1.])
                                     },
            index=[3, 4, 5, 0, 1, 2]
                                   )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterAutoregMultiSeries_levels_list_list_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in ForecasterAutoregMultiSeries with mocked
    when levels_list list (mocked done in Skforecast v0.5.0).
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    levels_list = ['1']
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
                    levels_list         = levels_list,
                    levels_weights      = None,
                    exog                = None,
                    lags_grid           = lags_grid,
                    refit               = False,
                    return_best         = False,
                    verbose             = False
              )
    
    expected_results = pd.DataFrame({
            'levels':[['1'], ['1'], ['1'], ['1'], ['1'], ['1']],
            'lags'  :[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2], [1, 2], [1, 2]],
            'params':[{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, {'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            'mean_absolute_error':np.array([0.20669393332187616, 0.20671040715338015, 0.20684013292264494, 
                                            0.2073988652614679, 0.20741562577568792, 0.2075484707375347]),                                                               
            'alpha' :np.array([0.01, 0.1, 1., 0.01, 0.1, 1.])
                                     },
            index=[3, 4, 5, 0, 1, 2]
                                   )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterAutoregMultiSeries_levels_weights_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in ForecasterAutoregMultiSeries with mocked
    when levels_weights (mocked done in Skforecast v0.5.0).
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2 # Placeholder, the value will be overwritten
                 )

    levels_weights = {'1': 0.6, '2': 0.4} 
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
                    levels_list         = None,
                    levels_weights      = levels_weights,
                    exog                = None,
                    lags_grid           = lags_grid,
                    refit               = False,
                    return_best         = False,
                    verbose             = False
              )
    
    expected_results = pd.DataFrame({
            'levels':[['1', '2'], ['1', '2'], ['1', '2'], ['1', '2'], ['1', '2'], ['1', '2']],
            'lags'  :[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2], [1, 2], [1, 2]],
            'params':[{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, {'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            'mean_absolute_error':np.array([0.20908359037019428, 0.20909615966954273, 0.20919158908362023,
                                            0.21011001165037002, 0.21011235006095524, 0.21012845276514763]),                                                               
            'alpha' :np.array([0.01, 0.1, 1., 0.01, 0.1, 1.])
                                     },
            index=[3, 4, 5, 0, 1, 2]
                                   )

    pd.testing.assert_frame_equal(results, expected_results)


def test_evaluate_grid_hyperparameters_multiseries_when_return_best():
    """
    Test forecaster is refited when return_best=True in _evaluate_grid_hyperparameters_multiseries.
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
        levels_list         = None,
        levels_weights      = None,
        exog                = None,
        lags_grid           = lags_grid,
        refit               = False,
        return_best         = True,
        verbose             = False
    )

    expected_lags = np.array([1, 2, 3, 4])
    expected_alpha = 0.01
    expected_series_levels = ['1', '2']
    
    assert (expected_lags == forecaster.lags).all()
    assert expected_alpha == forecaster.regressor.alpha
    assert expected_series_levels ==  forecaster.series_levels