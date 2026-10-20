"""
PERF-03: cost of the sklearn metrics used for the metrics passed as strings in
backtesting and hyperparameter search.

Prototype in memory (no change in the package): replaces the sklearn MAE and
MSE used by `_get_metric` with plain numpy versions, and compares wall time and
results in multi-series backtesting and one-step-ahead grid search, with 100
and 1000 series. Expected: same metrics (max diff 0) and 2.6x faster grid
search with `OneStepAheadFold`, 1.2x faster backtesting.
"""
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import skforecast.metrics.metrics as skm
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import (
    TimeSeriesFold, OneStepAheadFold, backtesting_forecaster_multiseries,
    grid_search_forecaster_multiseries
)

warnings.simplefilter("ignore")
rng = np.random.default_rng(0)
n_obs = 400
idx = pd.date_range("2020-01-01", periods=n_obs, freq="D")
t = np.arange(n_obs)

SK_MAE, SK_MSE = skm.mean_absolute_error, skm.mean_squared_error


def mean_absolute_error(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mean_squared_error(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def use(fast):
    skm.mean_absolute_error = mean_absolute_error if fast else SK_MAE
    skm.mean_squared_error = mean_squared_error if fast else SK_MSE


def timed(f, n=3):
    times, out = [], None
    for _ in range(n):
        s = time.perf_counter()
        out = f()
        times.append(time.perf_counter() - s)
    return np.median(times), out


metrics = ["mean_absolute_error", "mean_squared_error"]
for n_series in (100, 1000):
    series = pd.DataFrame(
        10 + 3 * np.sin(2 * np.pi * t / 7)[:, None] + rng.normal(0, 1, (n_obs, n_series)),
        index=idx, columns=[f"s{i}" for i in range(n_series)]
    )
    cases = {
        "backtesting Ridge": lambda: backtesting_forecaster_multiseries(
            ForecasterRecursiveMultiSeries(Ridge(), lags=7, encoding="ordinal"), series=series,
            cv=TimeSeriesFold(steps=7, initial_train_size=300, refit=False),
            metric=metrics, show_progress=False, n_jobs=1)[0],
        "grid OneStepAhead Ridge (8 candidates)": lambda: grid_search_forecaster_multiseries(
            ForecasterRecursiveMultiSeries(Ridge(), lags=7, encoding="ordinal"), series=series,
            cv=OneStepAheadFold(initial_train_size=300), param_grid={"alpha": [0.01, 0.1, 1, 10]},
            lags_grid=[7, 14], metric=metrics, return_best=False, show_progress=False,
            verbose=False),
    }
    for label, f in cases.items():
        use(False)
        t_sk, r_sk = timed(f)
        use(True)
        t_np, r_np = timed(f)
        num = r_sk.select_dtypes("number").columns
        diff = np.nanmax(np.abs(r_sk[num].to_numpy(float) - r_np[num].to_numpy(float)))
        print(f"{n_series:5d} series | {label:40s} | sklearn {t_sk:6.2f} s | numpy {t_np:6.2f} s "
              f"| {t_sk / t_np:4.2f}x | max diff {diff:.1e}")
use(False)
