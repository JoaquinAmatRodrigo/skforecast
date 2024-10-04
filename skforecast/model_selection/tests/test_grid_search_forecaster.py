# Unit test grid_search_forecaster
# ==============================================================================
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_percentage_error
from skforecast.metrics import mean_absolute_scaled_error
from skforecast.metrics import root_mean_squared_scaled_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection._search import grid_search_forecaster

from tqdm import tqdm
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # hide progress bar

# Fixtures
from .fixtures_model_selection import y


def test_output_grid_search_forecaster_ForecasterAutoreg_with_mocked():
    """
    Test output of grid_search_forecaster in ForecasterAutoreg with mocked
    (mocked done in Skforecast v0.4.3)
    """
    forecaster = ForecasterAutoreg(
        regressor=Ridge(random_state=123),
        lags=2,  # Placeholder, the value will be overwritten
    )

    steps = 3
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]
    param_grid = {"alpha": [0.01, 0.1, 1]}
    idx = len(lags_grid) * len(param_grid["alpha"])

    results = grid_search_forecaster(
        forecaster=forecaster,
        y=y,
        lags_grid=lags_grid,
        param_grid=param_grid,
        steps=steps,
        refit=False,
        metric="mean_squared_error",
        initial_train_size=len(y_train),
        fixed_train_size=False,
        return_best=False,
        verbose=False,
    )

    expected_results = pd.DataFrame(
        {
            "lags": [[1, 2], [1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            "lags_label": [
                [1, 2],
                [1, 2],
                [1, 2],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
            ],
            "params": [
                {"alpha": 0.01},
                {"alpha": 0.1},
                {"alpha": 1},
                {"alpha": 0.01},
                {"alpha": 0.1},
                {"alpha": 1},
            ],
            "mean_squared_error": np.array(
                [0.06464646, 0.06502362, 0.06745534, 0.06779272, 0.06802481, 0.06948609]
            ),
            "alpha": np.array([0.01, 0.1, 1.0, 0.01, 0.1, 1.0]),
        },
        index=pd.RangeIndex(start=0, stop=idx, step=1),
    )

    pd.testing.assert_frame_equal(results, expected_results)


forecasters = [
    ForecasterAutoreg(regressor=Ridge(random_state=678), lags=3),
    ForecasterAutoregDirect(regressor=Ridge(random_state=678), lags=3, steps=1),
    ForecasterAutoreg(
        regressor=Ridge(random_state=678),
        lags=3,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
    ),
    ForecasterAutoregDirect(
        regressor=Ridge(random_state=678),
        lags=3,
        steps=1,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
    ),
]
@pytest.mark.parametrize("forecaster", forecasters)
def test_output_grid_search_forecaster_equivalence_backtesting_one_step_ahead(
    forecaster,
):
    """
    Test that output of grid_search_forecaster is equivalent with backtesting and
    one step ahead one step=1, fixed_train_size=True retran=False
    """

    metrics = [
        "mean_absolute_error",
        "mean_squared_error",
        mean_absolute_percentage_error,
        mean_absolute_scaled_error,
        root_mean_squared_scaled_error,
    ]
    steps = 1
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]
    param_grid = {"alpha": [0.01, 0.1, 1]}

    results_backtesting = grid_search_forecaster(
        forecaster=forecaster,
        y=y,
        lags_grid=lags_grid,
        param_grid=param_grid,
        steps=steps,
        refit=False,
        metric=metrics,
        method="backtesting",
        initial_train_size=len(y_train),
        fixed_train_size=False,
        return_best=False,
        verbose=False,
    )

    results_one_step_ahead = grid_search_forecaster(
        forecaster=forecaster,
        y=y,
        lags_grid=lags_grid,
        param_grid=param_grid,
        steps=steps,
        metric=metrics,
        method="one_step_ahead",
        initial_train_size=len(y_train),
        return_best=False,
        verbose=False,
    )

    pd.testing.assert_frame_equal(results_backtesting, results_one_step_ahead)
