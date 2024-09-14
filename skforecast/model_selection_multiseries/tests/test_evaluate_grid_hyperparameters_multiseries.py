# Unit test _evaluate_grid_hyperparameters_multiseries
# ==============================================================================
import re
import os
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from skforecast.metrics import mean_absolute_scaled_error
from skforecast.metrics import root_mean_squared_scaled_error
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.ForecasterAutoregMultiSeriesCustom import ForecasterAutoregMultiSeriesCustom
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from skforecast.model_selection_multiseries.model_selection_multiseries import _evaluate_grid_hyperparameters_multiseries

from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # hide progress bar

# Fixtures
from .fixtures_model_selection_multiseries import series
from .fixtures_model_selection_multiseries import exog
series.index = pd.date_range(start='2024-01-01', periods=len(series), freq='D')
exog.index = pd.date_range(start='2024-01-01', periods=len(exog), freq='D')


def create_predictors(y):  # pragma: no cover
    """
    Create first 4 lags of a time series.
    """
    lags = y[-1:-5:-1]

    return lags


def test_ValueError_evaluate_grid_hyperparameters_multiseries_when_return_best_and_len_series_exog_different():
    """
    Test ValueError is raised in _evaluate_grid_hyperparameters_multiseries when 
    `return_best = True` and length of `series` and `exog` do not match.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 3,
                     encoding  = 'onehot'
                 )
    exog = series.iloc[:30, 0]

    err_msg = re.escape(
        (f"`exog` must have same number of samples as `series`. "
         f"length `exog`: ({len(exog)}), length `series`: ({len(series)})")
    )
    with pytest.raises(ValueError, match = err_msg):
        _evaluate_grid_hyperparameters_multiseries(
            forecaster         = forecaster,
            series             = series,
            exog               = exog,
            param_grid         = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            steps              = 4,
            metric             = 'mean_absolute_error',
            initial_train_size = 12,
            fixed_train_size   = False,
            levels             = None,
            lags_grid          = [2, 4],
            refit              = False,
            return_best        = True,
            verbose            = False
        )


def test_ValueError_evaluate_grid_hyperparameters_multiseries_when_not_allowed_aggregate_metric():
    """
    Test ValueError is raised in _evaluate_grid_hyperparameters_multiseries when 
    `aggregate_metric` has not a valid value.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 3,
                     encoding  = 'onehot'
                 )

    err_msg = re.escape(
        ("Allowed `aggregate_metric` are: ['average', 'weighted_average', 'pooling']. "
         "Got: ['not_valid'].")
    )
    with pytest.raises(ValueError, match = err_msg):
        _evaluate_grid_hyperparameters_multiseries(
            forecaster         = forecaster,
            series             = series,
            param_grid         = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            steps              = 4,
            metric             = 'mean_absolute_error',
            aggregate_metric   = 'not_valid',
            initial_train_size = 12,
            fixed_train_size   = False,
            levels             = None,
            lags_grid          = [2, 4],
            refit              = False,
            return_best        = True,
            verbose            = False
        )


def test_evaluate_grid_hyperparameters_multiseries_exception_when_metric_list_duplicate_names():
    """
    Test exception is raised in _evaluate_grid_hyperparameters when a `list` of 
    metrics is used with duplicate names.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 3,
                     encoding  = 'onehot'
                 )
    
    err_msg = re.escape("When `metric` is a `list`, each metric name must be unique.")
    with pytest.raises(ValueError, match = err_msg):
        _evaluate_grid_hyperparameters_multiseries(
            forecaster         = forecaster,
            series             = series,
            param_grid         = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            steps              = 4,
            metric             = ['mean_absolute_error', mean_absolute_error],
            initial_train_size = 12,
            fixed_train_size   = False,
            levels             = ['l1'],
            exog               = None,
            lags_grid          = [2, 4],
            refit              = False,
            return_best        = False,
            verbose            = False
        )


# ForecasterAutoregMultiSeries
# ======================================================================================================================
def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterAutoregMultiSeries_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in ForecasterAutoregMultiSeries 
    with mocked (mocked done in Skforecast v0.5.0).
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = Ridge(random_state=123),
                     lags               = 2, 
                     encoding           = 'onehot',
                     transformer_series = None
                 )

    steps = 3
    n_validation = 12
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  param_grid         = param_grid,
                  steps              = steps,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  initial_train_size = len(series) - n_validation,
                  fixed_train_size   = False,
                  levels             = None,
                  exog               = None,
                  lags_grid          = lags_grid,
                  refit              = False,
                  return_best        = False,
                  verbose            = False
              )

    expected_results = pd.DataFrame(
        {
            "levels": [["l1", "l2"]] * 6,
            "lags": [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2], [1, 2], [1, 2]],
            "lags_label": [
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2],
                [1, 2],
                [1, 2],
            ],
            "params": [
                {"alpha": 0.01},
                {"alpha": 0.1},
                {"alpha": 1},
                {"alpha": 1},
                {"alpha": 0.1},
                {"alpha": 0.01},
            ],
            "mean_absolute_error__weighted_average": np.array(
                [
                    0.20968100463227382,
                    0.20969259779858337,
                    0.20977945312386406,
                    0.21077344827205086,
                    0.21078653113227208,
                    0.21078779824759553,
                ]
            ),
            "alpha": np.array([0.01, 0.1, 1.0, 1.0, 0.1, 0.01]),
        },
        index=pd.Index([0, 1, 2, 3, 4, 5], dtype="int64"),
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_ForecasterAutoregMultiSeries_lags_grid_dict_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterAutoregMultiSeries 
    when `lags_grid` is a dict with mocked (mocked done in Skforecast v0.5.0).
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = Ridge(random_state=123),
                     lags               = 2,
                     encoding           = 'onehot', 
                     transformer_series = None
                 )

    steps = 3
    n_validation = 12
    lags_grid = {'lags_1': 2, 'lags_2': 4}
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  param_grid         = param_grid,
                  steps              = steps,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  initial_train_size = len(series) - n_validation,
                  fixed_train_size   = False,
                  levels             = None,
                  exog               = None,
                  lags_grid          = lags_grid,
                  refit              = False,
                  return_best        = False,
                  verbose            = False
              )

    expected_results = pd.DataFrame(
        {
            "levels": [["l1", "l2"]] * 6,
            "lags": [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2], [1, 2], [1, 2]],
            "lags_label": ["lags_2", "lags_2", "lags_2", "lags_1", "lags_1", "lags_1"],
            "params": [
                {"alpha": 0.01},
                {"alpha": 0.1},
                {"alpha": 1},
                {"alpha": 1},
                {"alpha": 0.1},
                {"alpha": 0.01},
            ],
            "mean_absolute_error__weighted_average": np.array(
                [
                    0.20968100463227382,
                    0.20969259779858337,
                    0.20977945312386406,
                    0.21077344827205086,
                    0.21078653113227208,
                    0.21078779824759553,
                ]
            ),
            "alpha": np.array([0.01, 0.1, 1.0, 1.0, 0.1, 0.01]),
        },
        index=pd.Index([0, 1, 2, 3, 4, 5], dtype="int64"),
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_ForecasterAutoregMultiSeries_lags_grid_is_None_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterAutoregMultiSeries 
    when `lags_grid` is `None` with mocked (mocked done in Skforecast v0.5.0), 
    should use forecaster.lags as lags_grid.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = Ridge(random_state=123),
                     lags               = 2, 
                     encoding           = 'onehot',
                     transformer_series = None
                 )

    lags_grid = None
    steps = 3
    n_validation = 12
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  param_grid         = param_grid,
                  steps              = steps,
                  metric             = mean_absolute_error,
                  aggregate_metric   = 'weighted_average',
                  initial_train_size = len(series) - n_validation,
                  fixed_train_size   = False,
                  levels             = ['l1', 'l2'],
                  exog               = None,
                  lags_grid          = lags_grid,
                  refit              = False,
                  return_best        = False,
                  verbose            = False
              )

    expected_results = pd.DataFrame(
        {
            "levels": [["l1", "l2"], ["l1", "l2"], ["l1", "l2"]],
            "lags": [[1, 2], [1, 2], [1, 2]],
            "lags_label": [[1, 2], [1, 2], [1, 2]],
            "params": [{"alpha": 1}, {"alpha": 0.1}, {"alpha": 0.01}],
            "mean_absolute_error__weighted_average": np.array(
                [0.21077344827205086, 0.21078653113227208, 0.21078779824759553]
            ),
            "alpha": np.array([1.0, 0.1, 0.01]),
        },
        index=pd.Index([0, 1, 2], dtype="int64"),
    )

    pd.testing.assert_frame_equal(results, expected_results)


@pytest.mark.parametrize("levels", 
                         ['l1', ['l1']], 
                         ids = lambda value: f'levels: {value}')
def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterAutoregMultiSeries_levels_str_list_with_mocked(levels):
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in ForecasterAutoregMultiSeries 
    with mocked when `levels` is a `str` or a `list` (mocked done in Skforecast v0.5.0).
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = Ridge(random_state=123),
                     lags               = 2, 
                     encoding           = 'onehot',
                     transformer_series = None
                 )

    steps = 3
    n_validation = 12
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  param_grid         = param_grid,
                  steps              = steps,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  initial_train_size = len(series) - n_validation,
                  fixed_train_size   = False,
                  levels             = levels,
                  exog               = None,
                  lags_grid          = lags_grid,
                  refit              = False,
                  return_best        = False,
                  verbose            = False
              )

    expected_results = pd.DataFrame(
        {
            "levels": [["l1"]] * 6,
            "lags": [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2], [1, 2], [1, 2]],
            "lags_label": [
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2],
                [1, 2],
                [1, 2],
            ],
            "params": [
                {"alpha": 0.01},
                {"alpha": 0.1},
                {"alpha": 1},
                {"alpha": 0.01},
                {"alpha": 0.1},
                {"alpha": 1},
            ],
            "mean_absolute_error": np.array(
                [
                    0.20669393332187616,
                    0.20671040715338015,
                    0.20684013292264494,
                    0.2073988652614679,
                    0.20741562577568792,
                    0.2075484707375347,
                ]
            ),
            "alpha": np.array([0.01, 0.1, 1.0, 0.01, 0.1, 1.0]),
        },
        index=pd.Index([0, 1, 2, 3, 4, 5], dtype="int64"),
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterAutoregMultiSeries_multiple_metrics_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in ForecasterAutoregMultiSeries 
    with mocked when multiple metrics (mocked done in Skforecast v0.6.0).
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = Ridge(random_state=123),
                     lags               = 2, 
                     encoding           = 'onehot',
                     transformer_series = None
                 )

    steps = 3
    n_validation = 12
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  param_grid         = param_grid,
                  steps              = steps,
                  metric             = [mean_squared_error, 'mean_absolute_error'],
                  aggregate_metric   = 'weighted_average',
                  initial_train_size = len(series) - n_validation,
                  fixed_train_size   = False,
                  levels             = None,
                  exog               = None,
                  lags_grid          = lags_grid,
                  refit              = False,
                  return_best        = False,
                  verbose            = False
              )
    
    expected_results = pd.DataFrame({
        'levels': [['l1', 'l2']] * 6,
        'lags': [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], 
                   [1, 2], [1, 2], [1, 2]],
        'lags_label': [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], 
                        [1, 2], [1, 2], [1, 2]],
        'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, 
                   {'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
        'mean_squared_error__weighted_average': np.array([
            0.06365397633008085, 0.06367614582294409, 0.06385378127252679, 
            0.06389613553855186, 0.06391570591810977, 0.06407787633532819]
        ),
        'mean_absolute_error__weighted_average': np.array(
            [0.20968100463227382, 0.20969259779858337, 0.20977945312386406, 
             0.21078779824759553, 0.21078653113227208, 0.21077344827205086]
        ),
        'alpha': np.array([0.01, 0.1, 1., 0.01, 0.1, 1.])
        },
        index = pd.Index([0, 1, 2, 3, 4, 5], dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results)


@pytest.mark.parametrize("lags_grid", 
                         [[2, 4], {'lags_1': 2, 'lags_2': 4}], 
                         ids=lambda lg: f'lags_grid: {lg}')
def test_evaluate_grid_hyperparameters_multiseries_when_return_best_ForecasterAutoregMultiSeries(lags_grid):
    """
    Test forecaster is refitted when `return_best = True` in 
    _evaluate_grid_hyperparameters_multiseries.
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = Ridge(random_state=123),
                     lags               = 2, 
                     encoding           = 'onehot',
                     transformer_series = None
                 )

    steps = 3
    n_validation = 12
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    _evaluate_grid_hyperparameters_multiseries(
        forecaster         = forecaster,
        series             = series,
        param_grid         = param_grid,
        steps              = steps,
        metric             = 'mean_absolute_error',
        aggregate_metric   = 'weighted_average',
                  initial_train_size = len(series) - n_validation,
        fixed_train_size   = False,
        levels             = None,
        exog               = None,
        lags_grid          = lags_grid,
        refit              = False,
        return_best        = True,
        verbose            = False
    )

    expected_lags = np.array([1, 2, 3, 4])
    expected_alpha = 0.01
    expected_series_col_names = ['l1', 'l2']
    
    assert (expected_lags == forecaster.lags).all()
    assert expected_alpha == forecaster.regressor.alpha
    assert expected_series_col_names ==  forecaster.series_col_names


def test_evaluate_grid_hyperparameters_multiseries_ForecasterAutoregMultiSeries_output_file_single_level():
    """
    Test output file is created when output_file is passed to
    _evaluate_grid_hyperparameters_multiseries and single level.
    """
    forecaster = ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), lags=2)

    steps = 3
    n_validation = 12
    lags_grid = {"lags_1": 2, "lags_2": 4}
    param_grid = [{"alpha": 0.01}, {"alpha": 0.1}, {"alpha": 1}]
    output_file = "test_evaluate_grid_hyperparameters_multiseries_output_file.txt"

    results = _evaluate_grid_hyperparameters_multiseries(
        forecaster=forecaster,
        series=series,
        param_grid=param_grid,
        steps=steps,
        metric="mean_absolute_error",
        aggregate_metric="weighted_average",
        initial_train_size=len(series) - n_validation,
        fixed_train_size=False,
        levels="l1",
        exog=None,
        lags_grid=lags_grid,
        refit=False,
        return_best=False,
        verbose=False,
        show_progress=False,
        output_file=output_file,
    )
    results = results.astype({"levels": str, "lags": str, "params": str})

    assert os.path.isfile(output_file)
    output_file_content = pd.read_csv(output_file, sep="\t", low_memory=False)
    output_file_content = output_file_content.sort_values(
        by="mean_absolute_error"
    ).reset_index(drop=True)
    output_file_content = output_file_content.astype(
        {"levels": str, "lags": str, "params": str}
    )
    pd.testing.assert_frame_equal(results, output_file_content)
    os.remove(output_file)


def test_evaluate_grid_hyperparameters_multiseries_ForecasterAutoregMultiSeries_output_file_multiple_metrics():
    """
    Test output file is created when output_file is passed to
    _evaluate_grid_hyperparameters_multiseries and list of metrics.
    """
    forecaster = ForecasterAutoregMultiSeries(regressor=Ridge(random_state=123), lags=2)

    steps = 3
    n_validation = 12
    lags_grid = [2, 4]
    param_grid = [{"alpha": 0.01}, {"alpha": 0.1}, {"alpha": 1}]
    output_file = "test_evaluate_grid_hyperparameters_multiseries_output_file.txt"

    results = _evaluate_grid_hyperparameters_multiseries(
        forecaster=forecaster,
        series=series,
        param_grid=param_grid,
        steps=steps,
        metric=[mean_squared_error, "mean_absolute_error"],
        aggregate_metric="weighted_average",
        initial_train_size=len(series) - n_validation,
        fixed_train_size=False,
        levels=None,
        exog=None,
        lags_grid=lags_grid,
        refit=False,
        return_best=False,
        verbose=False,
        show_progress=False,
        output_file=output_file,
    )
    results = results.astype(
        {"levels": str, "lags": str, "lags_label": str, "params": str}
    )

    assert os.path.isfile(output_file)
    output_file_content = pd.read_csv(output_file, sep="\t", low_memory=False)
    output_file_content = output_file_content.sort_values(
        by="mean_squared_error__weighted_average"
    ).reset_index(drop=True)
    output_file_content = output_file_content.astype(
        {"levels": str, "lags": str, "lags_label": str, "params": str}
    )
    pd.testing.assert_frame_equal(results, output_file_content)
    os.remove(output_file)


# ForecasterAutoregMultiSeriesCustom
# ======================================================================================================================
def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterAutoregMultiSeriesCustom_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in ForecasterAutoregMultiSeriesCustom 
    with mocked (mocked done in Skforecast v0.5.0).
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = Ridge(random_state=123),
                     fun_predictors     = create_predictors,
                     window_size        = 4,
                     encoding           = 'onehot',
                     transformer_series = None
                 )

    steps = 3
    n_validation = 12
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  param_grid         = param_grid,
                  steps              = steps,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  initial_train_size = len(series) - n_validation,
                  fixed_train_size   = False,
                  levels             = None,
                  exog               = None,
                  lags_grid          = None,
                  refit              = False,
                  return_best        = False,
                  verbose            = False
              )
    
    expected_results = pd.DataFrame({
        'levels': [['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2']],
        'lags': ['custom predictors', 'custom predictors', 'custom predictors'],
        'lags_label': ['custom predictors', 'custom predictors', 'custom predictors'],
        'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
        'mean_absolute_error__weighted_average': np.array(
            [0.20968100463227382, 0.20969259779858337, 0.20977945312386406]
        ),                                                               
        'alpha': np.array([0.01, 0.1, 1.])
        },
        index=pd.Index([0, 1, 2], dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results)


@pytest.mark.parametrize("levels", 
                         ['l1', ['l1']], 
                         ids = lambda value: f'levels: {value}')
def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterAutoregMultiSeriesCustom_levels_str_list_with_mocked(levels):
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in ForecasterAutoregMultiSeriesCustom 
    with mocked when `levels` is a `str` or a `list` (mocked done in Skforecast v0.5.0).
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = Ridge(random_state=123),
                     fun_predictors     = create_predictors,
                     window_size        = 4,
                     encoding           = 'onehot',
                     transformer_series = None
                 )

    steps = 3
    n_validation = 12
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  param_grid         = param_grid,
                  steps              = steps,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  initial_train_size = len(series) - n_validation,
                  fixed_train_size   = False,
                  levels             = levels,
                  exog               = None,
                  lags_grid          = None,
                  refit              = False,
                  return_best        = False,
                  verbose            = False
              )
    
    expected_results = pd.DataFrame({
        'levels': [['l1'], ['l1'], ['l1']],
        'lags': ['custom predictors', 'custom predictors', 'custom predictors'],
        'lags_label': ['custom predictors', 'custom predictors', 'custom predictors'],
        'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
        'mean_absolute_error': np.array(
            [0.20669393332187616, 0.20671040715338015, 0.20684013292264494]
        ),                                                               
        'alpha': np.array([0.01, 0.1, 1.])
        },
        index=pd.Index([0, 1, 2], dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterAutoregMultiSeriesCustom_multiple_metrics_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in ForecasterAutoregMultiSeriesCustom 
    with mocked when multiple metrics (mocked done in Skforecast v0.6.0).
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor          = Ridge(random_state=123),
                     fun_predictors     = create_predictors,
                     window_size        = 4,
                     encoding           = 'onehot',
                     transformer_series = None
                 )

    steps = 3
    n_validation = 12
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  param_grid         = param_grid,
                  steps              = steps,
                  metric             = [mean_squared_error, 'mean_absolute_error'],
                  aggregate_metric   = 'weighted_average',
                  initial_train_size = len(series) - n_validation,
                  fixed_train_size   = False,
                  levels             = None,
                  exog               = None,
                  lags_grid          = None,
                  refit              = False,
                  return_best        = False,
                  verbose            = False
              )
    
    expected_results = pd.DataFrame({
        'levels': [['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2']],
        'lags': ['custom predictors', 'custom predictors', 'custom predictors'],
        'lags_label': ['custom predictors', 'custom predictors', 'custom predictors'],
        'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
        'mean_squared_error__weighted_average': np.array(
            [0.06365397633008085, 0.06367614582294409, 0.06385378127252679]
        ),
        'mean_absolute_error__weighted_average': np.array(
            [0.20968100463227382, 0.20969259779858337, 0.20977945312386406]
        ),
        'alpha': np.array([0.01, 0.1, 1.])
        },
        index=pd.Index([0, 1, 2], dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_evaluate_grid_hyperparameters_multiseries_when_return_best_ForecasterAutoregMultiSeriesCustom():
    """
    Test forecaster is refitted when `return_best = True` in 
    _evaluate_grid_hyperparameters_multiseries.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = Ridge(random_state=123),
                     fun_predictors = create_predictors,
                     window_size    = 4,
                     encoding       = 'onehot'
                 )

    steps = 3
    n_validation = 12
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    _evaluate_grid_hyperparameters_multiseries(
        forecaster         = forecaster,
        series             = series,
        param_grid         = param_grid,
        steps              = steps,
        metric             = 'mean_absolute_error',
        aggregate_metric   = 'weighted_average',
                  initial_train_size = len(series) - n_validation,
        fixed_train_size   = False,
        levels             = None,
        exog               = None,
        lags_grid          = None,
        refit              = False,
        return_best        = True,
        verbose            = False
    )

    expected_alpha = 0.01
    expected_series_col_names = ['l1', 'l2']
    
    assert expected_alpha == forecaster.regressor.alpha
    assert expected_series_col_names ==  forecaster.series_col_names


def test_evaluate_grid_hyperparameters_multiseries_ForecasterAutoregMultiSeriesCustom_output_file_single_level():
    """
    Test output file is created when output_file is passed to 
    _evaluate_grid_hyperparameters_multiseries and single level.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = Ridge(random_state=123),
                     fun_predictors = create_predictors,
                     window_size    = 4,
                     encoding       = 'onehot'
                 )

    steps = 3
    n_validation = 12
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]
    output_file = 'test_evaluate_grid_hyperparameters_multiseries_output_file.txt'

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  param_grid         = param_grid,
                  steps              = steps,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  initial_train_size = len(series) - n_validation,
                  fixed_train_size   = False,
                  levels             = 'l1',
                  exog               = None,
                  lags_grid          = None,
                  refit              = False,
                  return_best        = False,
                  verbose            = False,
                  show_progress      = False,
                  output_file        = output_file
              )
    results  = results.astype({'levels': str, 'lags': str, 'params': str})

    assert os.path.isfile(output_file)
    output_file_content = pd.read_csv(output_file, sep='\t', low_memory=False)
    output_file_content = output_file_content.sort_values(by='mean_absolute_error').reset_index(drop=True)
    output_file_content = output_file_content.astype({'levels': str, 'lags': str, 'params': str})
    pd.testing.assert_frame_equal(results, output_file_content)
    os.remove(output_file)


def test_evaluate_grid_hyperparameters_multiseries_ForecasterAutoregMultiSeriesCustom_output_file_multiple_metrics():
    """
    Test output file is created when output_file is passed to 
    _evaluate_grid_hyperparameters_multiseries and list of metrics.
    """
    forecaster = ForecasterAutoregMultiSeriesCustom(
                     regressor      = Ridge(random_state=123),
                     fun_predictors = create_predictors,
                     window_size    = 4,
                     encoding       = 'onehot'
                 )

    steps = 3
    n_validation = 12
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]
    output_file = 'test_evaluate_grid_hyperparameters_multiseries_output_file.txt'

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  param_grid         = param_grid,
                  steps              = steps,
                  metric             = [mean_squared_error, 'mean_absolute_error'],
                  aggregate_metric   = 'weighted_average',
                  initial_train_size = len(series) - n_validation,
                  fixed_train_size   = False,
                  levels             = None,
                  exog               = None,
                  lags_grid          = None,
                  refit              = False,
                  return_best        = False,
                  verbose            = False,
                  show_progress      = False,
                  output_file        = output_file
              )
    results  = results.astype({'levels': str, 'lags': str, 'params': str})

    assert os.path.isfile(output_file)
    output_file_content = pd.read_csv(output_file, sep='\t', low_memory=False)
    output_file_content = output_file_content.sort_values(by='mean_squared_error__weighted_average').reset_index(drop=True)
    output_file_content = output_file_content.astype({'levels': str, 'lags': str, 'params': str})
    pd.testing.assert_frame_equal(results, output_file_content)
    os.remove(output_file)


def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterAutoregMultiSeries_multiple_metrics_aggregated_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in ForecasterAutoregMultiSeries 
    with mocked when multiple metrics (mocked done in Skforecast v0.6.0).
    """
    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = Ridge(random_state=123),
                     lags               = 2, 
                     encoding           = 'onehot',
                     transformer_series = None
                 )

    steps = 3
    n_validation = 12
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  param_grid         = param_grid,
                  steps              = steps,
                  metric             = ['mean_absolute_error', 'mean_absolute_scaled_error'],
                  aggregate_metric   = ['weighted_average', 'average', 'pooling'],
                  initial_train_size = len(series) - n_validation,
                  fixed_train_size   = False,
                  levels             = None,
                  exog               = None,
                  lags_grid          = lags_grid,
                  refit              = False,
                  return_best        = False,
                  verbose            = False,
                  show_progress     = False,
              )
    
    expected_results = pd.DataFrame({
        "levels": {
            0: ["l1", "l2"],
            1: ["l1", "l2"],
            2: ["l1", "l2"],
            3: ["l1", "l2"],
            4: ["l1", "l2"],
            5: ["l1", "l2"],
        },
        "lags": {
            0: np.array([1, 2, 3, 4]),
            1: np.array([1, 2, 3, 4]),
            2: np.array([1, 2, 3, 4]),
            3: np.array([1, 2]),
            4: np.array([1, 2]),
            5: np.array([1, 2]),
        },
        "lags_label": {
            0: np.array([1, 2, 3, 4]),
            1: np.array([1, 2, 3, 4]),
            2: np.array([1, 2, 3, 4]),
            3: np.array([1, 2]),
            4: np.array([1, 2]),
            5: np.array([1, 2]),
        },
        "params": {
            0: {"alpha": 0.01},
            1: {"alpha": 0.1},
            2: {"alpha": 1},
            3: {"alpha": 1},
            4: {"alpha": 0.1},
            5: {"alpha": 0.01},
        },
        "mean_absolute_error__weighted_average": {
            0: 0.20968100547390048,
            1: 0.20969259864077977,
            2: 0.20977945397058564,
            3: 0.21077344921320568,
            4: 0.21078653208835063,
            5: 0.21078779920557153,
        },
        "mean_absolute_error__average": {
            0: 0.20968100547390048,
            1: 0.20969259864077974,
            2: 0.20977945397058564,
            3: 0.21077344921320565,
            4: 0.21078653208835063,
            5: 0.21078779920557153,
        },
        "mean_absolute_error__pooling": {
            0: 0.20968100547390045,
            1: 0.2096925986407798,
            2: 0.20977945397058564,
            3: 0.21077344921320565,
            4: 0.21078653208835063,
            5: 0.21078779920557153,
        },
        "mean_absolute_scaled_error__weighted_average": {
            0: 0.7969369551529275,
            1: 0.7969838748911608,
            2: 0.7973389652448446,
            3: 0.8009631048212882,
            4: 0.8009302953795885,
            5: 0.8009249124659391,
        },
        "mean_absolute_scaled_error__average": {
            0: 0.7969369551529275,
            1: 0.7969838748911608,
            2: 0.7973389652448445,
            3: 0.8009631048212883,
            4: 0.8009302953795885,
            5: 0.8009249124659391,
        },
        "mean_absolute_scaled_error__pooling": {
            0: 0.7809734688246502,
            1: 0.7810166484905049,
            2: 0.7813401480275807,
            3: 0.7850423618302551,
            4: 0.785091090032226,
            5: 0.7850958095104122,
        },
        "alpha": {0: 0.01, 1: 0.1, 2: 1.0, 3: 1.0, 4: 0.1, 5: 0.01},
    })

    pd.testing.assert_frame_equal(results, expected_results)


# ForecasterAutoregMultiVariate
# ======================================================================================================================
def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterAutoregMultiVariate_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in ForecasterAutoregMultiVariate 
    with mocked (mocked done in Skforecast v0.6.0).
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 3,
                     transformer_series = None
                 )

    steps = 3
    n_validation = 12
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  param_grid         = param_grid,
                  steps              = steps,
                  metric             = 'mean_absolute_error',
                  initial_train_size = len(series) - n_validation,
                  fixed_train_size   = False,
                  levels             = None,
                  exog               = None,
                  lags_grid          = lags_grid,
                  refit              = False,
                  return_best        = False,
                  verbose            = True
              )
    
    expected_results = pd.DataFrame({
        'levels': [['l1']] * 6,
        'lags': [[1, 2], [1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        'lags_label': [[1, 2], [1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, 
                   {'alpha': 1}, {'alpha': 0.1}, {'alpha': 0.01}],
        'mean_absolute_error': np.array(
            [0.20115194, 0.20183032, 0.20566862,
             0.22224269, 0.22625017, 0.22644284]
        ),                                                               
        'alpha': np.array([0.01, 0.1, 1., 1., 0.1, 0.01])
        },
        index = pd.Index([0, 1, 2, 3, 4, 5], dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_ForecasterAutoregMultiVariate_lags_grid_dict_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterAutoregMultiVariate 
    when `lags_grid` is a dict with mocked (mocked done in Skforecast v0.6.0)
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 3,
                     transformer_series = None
                 )

    steps = 3
    n_validation = 12
    lags_grid = {'lags_1': 2, 'lags_2': 4}
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  param_grid         = param_grid,
                  steps              = steps,
                  metric             = 'mean_absolute_error',
                  initial_train_size = len(series) - n_validation,
                  fixed_train_size   = False,
                  levels             = None,
                  exog               = None,
                  lags_grid          = lags_grid,
                  refit              = False,
                  return_best        = False,
                  verbose            = True
              )
    
    expected_results = pd.DataFrame({
        'levels': [['l1']] * 6,
        'lags': [[1, 2], [1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        'lags_label': ['lags_1', 'lags_1', 'lags_1', 'lags_2', 'lags_2', 'lags_2'],
        'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, 
                   {'alpha': 1}, {'alpha': 0.1}, {'alpha': 0.01}],
        'mean_absolute_error': np.array(
            [0.20115194, 0.20183032, 0.20566862,
             0.22224269, 0.22625017, 0.22644284]
            ),                                                               
        'alpha': np.array([0.01, 0.1, 1., 1., 0.1, 0.01])
        },
        index = pd.Index([0, 1, 2, 3, 4, 5], dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_ForecasterAutoregMultiVariate_lags_grid_is_None_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterAutoregMultiVariate 
    when `lags_grid` is `None` with mocked (mocked done in Skforecast v0.6.0), 
    should use forecaster.lags as lags_grid.
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 3,
                     transformer_series = None
                 )

    lags_grid = None
    steps = 3
    n_validation = 12
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  param_grid         = param_grid,
                  steps              = steps,
                  metric             = mean_absolute_error,
                  initial_train_size = len(series) - n_validation,
                  fixed_train_size   = False,
                  levels             = None,
                  exog               = None,
                  lags_grid          = lags_grid,
                  refit              = False,
                  return_best        = False,
                  verbose            = False
              )
    
    expected_results = pd.DataFrame({
        'levels': [['l1']] * 3,
        'lags': [[1, 2], [1, 2], [1, 2]],
        'lags_label': [[1, 2], [1, 2], [1, 2]],
        'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1.}],
        'mean_absolute_error': np.array([0.20115194, 0.20183032, 0.20566862]),                                                               
        'alpha': np.array([0.01, 0.1, 1.])
    },
        index = pd.Index([0, 1, 2], dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_ForecasterAutoregMultiVariate_lags_grid_is_list_of_dicts_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterAutoregMultiVariate 
    when `lags_grid` is a list of dicts with mocked (mocked done in Skforecast v0.6.0).
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 3,
                     transformer_series = None
                 )

    lags_grid = [{'l1': 2, 'l2': 3}, {'l1': [1, 3], 'l2': 3}, {'l1': 2, 'l2': [1, 4]}]
    steps = 3
    n_validation = 12
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  param_grid         = param_grid,
                  steps              = steps,
                  metric             = mean_absolute_error,
                  initial_train_size = len(series) - n_validation,
                  fixed_train_size   = False,
                  levels             = None,
                  exog               = None,
                  lags_grid          = lags_grid,
                  refit              = False,
                  return_best        = False,
                  verbose            = False
              )
    
    expected_results = pd.DataFrame({
        'levels': [['l1']] * 9,
        'lags': [{'l1': np.array([1, 2]), 'l2': np.array([1, 2, 3])},
                   {'l1': np.array([1, 2]), 'l2': np.array([1, 2, 3])},
                   {'l1': np.array([1, 2]), 'l2': np.array([1, 2, 3])},
                   {'l1': np.array([1, 3]), 'l2': np.array([1, 2, 3])},
                   {'l1': np.array([1, 3]), 'l2': np.array([1, 2, 3])},
                   {'l1': np.array([1, 3]), 'l2': np.array([1, 2, 3])},
                   {'l1': np.array([1, 2]), 'l2': np.array([1, 4])},
                   {'l1': np.array([1, 2]), 'l2': np.array([1, 4])},
                   {'l1': np.array([1, 2]), 'l2': np.array([1, 4])}],
        'lags_label': [{'l1': np.array([1, 2]), 'l2': np.array([1, 2, 3])},
                        {'l1': np.array([1, 2]), 'l2': np.array([1, 2, 3])},
                        {'l1': np.array([1, 2]), 'l2': np.array([1, 2, 3])},
                        {'l1': np.array([1, 3]), 'l2': np.array([1, 2, 3])},
                        {'l1': np.array([1, 3]), 'l2': np.array([1, 2, 3])},
                        {'l1': np.array([1, 3]), 'l2': np.array([1, 2, 3])},
                        {'l1': np.array([1, 2]), 'l2': np.array([1, 4])},
                        {'l1': np.array([1, 2]), 'l2': np.array([1, 4])},
                        {'l1': np.array([1, 2]), 'l2': np.array([1, 4])}],
        'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1.}, 
                   {'alpha': 1.}, {'alpha': 0.1}, {'alpha': 0.01}, 
                   {'alpha': 1.}, {'alpha': 0.1}, {'alpha': 0.01}],
        'mean_absolute_error': np.array(
            [0.2053202, 0.20555199, 0.20677802, 
             0.21443621, 0.21801147, 0.21863968, 
             0.22401526, 0.22830217, 0.22878132]
            ),                                                               
        'alpha': np.array([0.01, 0.1, 1., 1., 0.1, 0.01, 1., 0.1, 0.01])
        },
        index = pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_ForecasterAutoregMultiVariate_lags_grid_is_dict_of_dicts_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterAutoregMultiVariate 
    when `lags_grid` is a dict of dicts with mocked (mocked done in Skforecast v0.6.0).
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 3,
                     transformer_series = None
                 )

    lags_grid = {
        'lags_1': {'l1': 2, 'l2': 3},
        'lags_2': {'l1': [1, 3], 'l2': 3},
        'lags_3': {'l1': 2, 'l2': [1, 4]},
        'lags_4': {'l1': 2, 'l2': None},
        'lags_5': {'l1': None, 'l2': 2},
        'lags_6': 3
    }
    steps = 3
    n_validation = 12
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  param_grid         = param_grid,
                  steps              = steps,
                  metric             = mean_absolute_error,
                  initial_train_size = len(series) - n_validation,
                  fixed_train_size   = False,
                  levels             = None,
                  exog               = None,
                  lags_grid          = lags_grid,
                  refit              = False,
                  return_best        = False,
                  verbose            = False
              )
    
    expected_results = pd.DataFrame({
        'levels': [['l1']] * 18,
        'lags': [{'l1': np.array([1, 2]), 'l2': None},
                   {'l1': np.array([1, 2]), 'l2': None},
                   {'l1': np.array([1, 2]), 'l2': None},
                   {'l1': np.array([1, 2]), 'l2': np.array([1, 2, 3])},
                   {'l1': np.array([1, 2]), 'l2': np.array([1, 2, 3])},
                   {'l1': np.array([1, 2]), 'l2': np.array([1, 2, 3])},
                   {'l1': None, 'l2': np.array([1, 2])},
                   {'l1': None, 'l2': np.array([1, 2])},
                   {'l1': None, 'l2': np.array([1, 2])},
                   [1, 2, 3],
                   {'l1': np.array([1, 3]), 'l2': np.array([1, 2, 3])},
                   [1, 2, 3],
                   [1, 2, 3],
                   {'l1': np.array([1, 3]), 'l2': np.array([1, 2, 3])},
                   {'l1': np.array([1, 3]), 'l2': np.array([1, 2, 3])},
                   {'l1': np.array([1, 2]), 'l2': np.array([1, 4])},
                   {'l1': np.array([1, 2]), 'l2': np.array([1, 4])},
                   {'l1': np.array([1, 2]), 'l2': np.array([1, 4])}],
        'lags_label': ['lags_4', 'lags_4', 'lags_4', 
                       'lags_1', 'lags_1', 'lags_1', 
                       'lags_5', 'lags_5', 'lags_5',
                       'lags_6', 'lags_2', 'lags_6', 
                       'lags_6', 'lags_2', 'lags_2', 
                       'lags_3', 'lags_3', 'lags_3'],
        'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, 
                   {'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, 
                   {'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, 
                   {'alpha': 1}, {'alpha': 1}, {'alpha': 0.1}, 
                   {'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 0.01}, 
                   {'alpha': 1}, {'alpha': 0.1}, {'alpha': 0.01}],
        'mean_absolute_error': np.array(
            [0.20155258, 0.20208154, 0.20516149, 0.2053202, 0.20555199,
             0.20677802, 0.21005165, 0.21007475, 0.21071924, 0.21353688,
             0.21443621, 0.21622784, 0.2166998, 0.21801147, 0.21863968,
             0.22401526, 0.22830217, 0.22878132]),
        'alpha': np.array([0.01, 0.1, 1., 0.01, 0.1, 1., 0.01, 0.1, 1., 1., 1.,
       0.1, 0.01, 0.1, 0.01, 1., 0.1, 0.01])
        },
        index = pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterAutoregMultiVariate_multiple_metrics_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in ForecasterAutoregMultiVariate 
    with mocked when multiple metrics (mocked done in Skforecast v0.6.0).
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 3,
                     transformer_series = None
                 )

    steps = 3
    n_validation = 12
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  param_grid         = param_grid,
                  steps              = steps,
                  metric             = [mean_squared_error, 'mean_absolute_error'],
                  initial_train_size = len(series) - n_validation,
                  fixed_train_size   = False,
                  levels             = None,
                  exog               = None,
                  lags_grid          = lags_grid,
                  refit              = False,
                  return_best        = False,
                  verbose            = False
              )
    
    expected_results = pd.DataFrame({
        'levels': [['l1']] * 6,
        'lags': [[1, 2], [1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        'lags_label': [[1, 2], [1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, 
                   {'alpha': 1.}, {'alpha': 0.1}, {'alpha': 0.01}],
        'mean_squared_error': np.array([0.06260985, 0.06309219, 0.06627699, 
                                        0.08032378, 0.08400047, 0.08448937]),
        'mean_absolute_error': np.array(
            [0.20115194, 0.20183032, 0.20566862, 0.22224269, 0.22625017, 0.22644284]),
        'alpha': np.array([0.01, 0.1, 1., 1., 0.1, 0.01])
        },
        index = pd.Index([0, 1, 2, 3, 4, 5], dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results)


@pytest.mark.parametrize("lags_grid", 
                         [[2, 4], {'lags_1': 2, 'lags_2': 4}], 
                         ids=lambda lg: f'lags_grid: {lg}')
def test_evaluate_grid_hyperparameters_multiseries_when_return_best_ForecasterAutoregMultiVariate(lags_grid):
    """
    Test forecaster is refitted when `return_best = True` in 
    _evaluate_grid_hyperparameters_multiseries.
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 3,
                     transformer_series = None
                 )

    steps = 3
    n_validation = 12
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    _evaluate_grid_hyperparameters_multiseries(
        forecaster         = forecaster,
        series             = series,
        param_grid         = param_grid,
        steps              = steps,
        metric             = 'mean_absolute_error',
        aggregate_metric   = 'weighted_average',
                  initial_train_size = len(series) - n_validation,
        fixed_train_size   = False,
        levels             = None,
        exog               = None,
        lags_grid          = lags_grid,
        refit              = False,
        return_best        = True,
        verbose            = False,
        show_progress      = False
    )

    expected_lags = np.array([1, 2])
    expected_alpha = 0.01
    expected_series_col_names = ['l1', 'l2']
    
    assert (expected_lags == forecaster.lags).all()
    for i in range(1, forecaster.steps + 1):
        assert expected_alpha == forecaster.regressors_[i].alpha
    assert expected_series_col_names ==  forecaster.series_col_names


def test_evaluate_grid_hyperparameters_multiseries_ForecasterAutoregMultiVariate_output_file_single_level():
    """
    Test output file is created when output_file is passed to 
    _evaluate_grid_hyperparameters_multiseries and single level.
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor = Ridge(random_state=123),
                     level     = 'l1',
                     lags      = 2,
                     steps     = 3
                 )

    steps = 3
    n_validation = 12
    lags_grid = {'lags_1': 2, 'lags_2': 4}
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]
    output_file = 'test_evaluate_grid_hyperparameters_multiseries_output_file.txt'

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  param_grid         = param_grid,
                  steps              = steps,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  initial_train_size = len(series) - n_validation,
                  fixed_train_size   = False,
                  levels             = 'l1',
                  exog               = None,
                  lags_grid          = lags_grid,
                  refit              = False,
                  return_best        = False,
                  verbose            = False,
                  show_progress      = False,
                  output_file        = output_file
              )
    results  = results.astype({'levels': str, 'lags': str, 'lags_label': str, 'params': str})

    assert os.path.isfile(output_file)
    output_file_content = pd.read_csv(output_file, sep='\t', low_memory=False)
    output_file_content = output_file_content.sort_values(by='mean_absolute_error').reset_index(drop=True)
    output_file_content = output_file_content.astype({'levels': str, 'lags': str, 'lags_label': str, 'params': str})
    pd.testing.assert_frame_equal(results, output_file_content)
    os.remove(output_file)


def test_evaluate_grid_hyperparameters_multiseries_ForecasterAutoregMultiVariate_output_file_multiple_metrics():
    """
    Test output file is created when output_file is passed to 
    _evaluate_grid_hyperparameters_multiseries and list of metrics.
    """
    forecaster = ForecasterAutoregMultiVariate(
                     regressor = Ridge(random_state=123),
                     level     = 'l2',
                     lags      = 2,
                     steps     = 3
                 )

    steps = 3
    n_validation = 12
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]
    output_file = 'test_evaluate_grid_hyperparameters_multiseries_output_file.txt'

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  param_grid         = param_grid,
                  steps              = steps,
                  metric             = [mean_squared_error, 'mean_absolute_error'],
                  initial_train_size = len(series) - n_validation,
                  fixed_train_size   = False,
                  levels             = None,
                  exog               = None,
                  lags_grid          = lags_grid,
                  refit              = False,
                  return_best        = False,
                  verbose            = False,
                  show_progress      = False,
                  output_file        = output_file
              )
    results  = results.astype({'levels': str, 'lags': str, 'lags_label': str, 'params': str})

    assert os.path.isfile(output_file)
    output_file_content = pd.read_csv(output_file, sep='\t', low_memory=False)
    output_file_content = output_file_content.sort_values(
        by="mean_squared_error"
    ).reset_index(drop=True)
    output_file_content = output_file_content.astype({'levels': str, 'lags': str, 'lags_label': str, 'params': str})
    pd.testing.assert_frame_equal(results, output_file_content)
    os.remove(output_file)


forecasters = [
    ForecasterAutoregMultiSeries(regressor=Ridge(random_state=678), lags=3),
    ForecasterAutoregMultiSeries(
        regressor=Ridge(random_state=678),
        lags=3,
        transformer_series=None,
    ),
    ForecasterAutoregMultiSeries(
        regressor=Ridge(random_state=678),
        lags=3,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler()
    ),
    ForecasterAutoregMultiVariate(
        regressor=Ridge(random_state=678),
        level='l1',
        lags=3,
        steps=1,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler()
    )
]
@pytest.mark.parametrize("forecaster", forecasters)
def test_evaluate_grid_hyperparameters_equivalent_outputs_backtesting_one_step_ahead(
    forecaster,
):

    metrics = [
        "mean_absolute_error",
        "mean_squared_error",
        mean_absolute_percentage_error,
        mean_absolute_scaled_error,
        root_mean_squared_scaled_error,
    ]
    steps = 1
    initial_train_size = 20
    param_grid = {
        "alpha": np.logspace(-1, 1, 3),
    }
    lags_grid = [3, 7]
    param_grid = list(ParameterGrid(param_grid))
    results_backtesting = _evaluate_grid_hyperparameters_multiseries(
        forecaster         = forecaster,
        series             = series,
        exog               = exog,
        param_grid         = param_grid,
        lags_grid          = lags_grid,
        steps              = steps,
        refit              = False,
        metric             = metrics,
        initial_train_size = initial_train_size,
        method             = 'backtesting',
        fixed_train_size   = False,
        return_best        = False,
        n_jobs             = 'auto',
        verbose            = False,
        show_progress      = False
    )
    results_one_step_ahead = _evaluate_grid_hyperparameters_multiseries(
        forecaster         = forecaster,
        series             = series,
        exog               = exog,
        param_grid         = param_grid,
        lags_grid          = lags_grid,
        metric             = metrics,
        initial_train_size = initial_train_size,
        method             = 'one_step_ahead',
        return_best        = False,
        verbose            = False,
        show_progress      = False
    )

    pd.testing.assert_frame_equal(results_backtesting, results_one_step_ahead)
