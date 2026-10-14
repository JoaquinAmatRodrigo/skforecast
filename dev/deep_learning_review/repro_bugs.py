"""Minimal reproductions of the ForecasterRnn bugs listed in the handoff."""
import os
os.environ["KERAS_BACKEND"] = "torch"
import contextlib
import io
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from skforecast.deep_learning import ForecasterRnn, create_and_compile_model
from skforecast.model_selection import TimeSeriesFold, grid_search_forecaster_multiseries

rng = np.random.default_rng(0)
idx = pd.date_range("2020-01-01", periods=300, freq="h")
series = pd.DataFrame(
    {"a": np.sin(np.arange(300) / 5) + rng.normal(0, 0.1, 300), "b": rng.normal(size=300)},
    index=idx,
)
exog = pd.DataFrame({"e": rng.normal(size=300)}, index=idx)


def build(**kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return create_and_compile_model(
            series=series, lags=10, steps=3, levels=["a"],
            compile_kwargs={"optimizer": "adam", "loss": "mse"}, **kwargs
        )


def fit(forecaster, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        forecaster.fit(*args, **kwargs)


print("== 1. fit() does not reset the weights (warm start)")
f = ForecasterRnn(build(), levels="a", lags=10, fit_kwargs={"epochs": 3, "verbose": 0})
fit(f, series)
loss_1 = np.round(f.history_["loss"], 4)
fit(f, series)
loss_2 = np.round(f.history_["loss"], 4)
print(f"   loss 1st fit: {loss_1}   loss 2nd fit: {loss_2}  (2nd starts where 1st ended)")

print("== 2. exog without the first window_size rows (accepted by the length check)")
f = ForecasterRnn(build(exog=exog), levels="a", lags=10, fit_kwargs={"epochs": 1, "verbose": 0})
try:
    fit(f, series, exog=exog.iloc[10:])
    print("   fit OK")
except Exception as e:
    print(f"   fit FAILS: {type(e).__name__}: {str(e).splitlines()[0]}")

print("== 3. fit() picks the torch device itself, ignoring KERAS_TORCH_DEVICE")
os.environ["KERAS_TORCH_DEVICE"] = "cpu"  # read by keras at import; set here only to show intent
print(f"   keras device: {f.estimator.weights[0].value.device}")
f = ForecasterRnn(build(), levels="a", lags=10, fit_kwargs={"epochs": 1, "verbose": 0})
cuda_available = torch.cuda.is_available
torch.cuda.is_available = lambda: True  # simulate a machine with a GPU
try:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        f.fit(series)
    print(f"   fit OK, printed: {buffer.getvalue().strip()!r}")
except Exception as e:
    print(f"   skforecast moved the data to 'cuda' on its own: {type(e).__name__}")
finally:
    torch.cuda.is_available = cuda_available

print("== 4. Hyperparameter search never works: set_params calls sklearn clone on a Keras model")
f = ForecasterRnn(build(), levels="a", lags=10, fit_kwargs={"epochs": 1, "verbose": 0})
cv = TimeSeriesFold(steps=3, initial_train_size=200, refit=False)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    with contextlib.redirect_stdout(io.StringIO()):
        results = grid_search_forecaster_multiseries(
            forecaster=f, series=series, cv=cv, param_grid={"loss": ["mse", "mae"]},
            metric="mean_absolute_error", return_best=False, show_progress=False,
        )
print(f"   results shape: {results.shape}")
for w in caught:
    if "Parameters skipped" in str(w.message) or "No valid parameter" in str(w.message):
        print(f"   warning: {str(w.message)[:110]}...")
try:
    f.set_params({"loss": "mae"})
except Exception as e:
    print(f"   set_params FAILS: {type(e).__name__}")
