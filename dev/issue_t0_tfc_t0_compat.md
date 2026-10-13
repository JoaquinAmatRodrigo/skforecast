# Issue draft: T0Adapter is incompatible with tfc-t0 >= 0.4

**Title:** `T0Adapter` fails with `tfc-t0>=0.4`: `predict()` got an unexpected keyword argument `context`

**Labels:** bug, foundation

---

## Description

`T0Adapter` in skforecast 0.26.0 does not work with the current releases of
the backend library `tfc-t0`. Every call to `predict`, `predict_interval` or
`backtesting_foundation` with `model_id="theforecastingcompany/t0-alpha"`
fails when `tfc-t0` 0.4.0 or later is installed (0.6.0 is the latest on PyPI,
released on 2026-10-13).

Users following the install instructions (`pip install tfc-t0`) get the latest
version, so T0 is broken out of the box.

There are two separate breakages:

1. Since 0.4.0, `T0Forecaster.predict` no longer accepts the arguments the
   adapter passes (`context`, and `quantiles` since 0.5.0).
2. Since 0.6.0, `pip install tfc-t0` no longer installs torch: the PyTorch
   runtime is an extra (`tfc-t0[torch]`), so the model cannot even be imported.

## Reproduce

```python
# pip install skforecast==0.26.0 tfc-t0==0.5.0
from skforecast.datasets import fetch_dataset
from skforecast.foundation import FoundationModel, ForecasterFoundation

data = fetch_dataset(name="vic_electricity").resample("h").mean(numeric_only=True)
forecaster = ForecasterFoundation(
    estimator=FoundationModel(model_id="theforecastingcompany/t0-alpha", context_length=500)
)
forecaster.fit(series=data["Demand"])
forecaster.predict(steps=24)
```

```
File .../skforecast/foundation/_adapters.py:3724, in T0Adapter.predict(...)
-> 3724 forecast = self._model.predict(
   3725     context           = context_batch,
   3726     horizon           = steps,
   3727     quantiles         = query_levels,
   3728     future_covariates = future_covariates,
   3729 )

TypeError: T0Forecaster.predict() got an unexpected keyword argument 'context'
```

## Cause

`T0Adapter.predict` calls `T0Forecaster.predict(context=..., quantiles=...)`.
The `tfc-t0` API changed:

| tfc-t0 | Released | 1st argument of `predict` | Quantiles argument | Works with skforecast 0.26.0 |
|---|---|---|---|---|
| 0.3.1 and older | up to 2026-09-01 | `context` | `quantiles` | Yes |
| 0.3.2 | 2026-09-09 | `model_input` (TimeSeries) or `context` (array) overloads | `quantiles` | Yes (last working version) |
| 0.4.0 | 2026-09-15 | `model_input` | `quantiles` | No |
| 0.5.0 | 2026-09-17 | `model_input` | `quantile_levels` | No |
| 0.6.0 | 2026-10-13 | `model_input` | `quantile_levels` | No (and torch is no longer installed by default) |

Unchanged in 0.5.0 and 0.6.0: `horizon`, `future_covariates` (keyword), and the
returned `Forecast.quantiles` attribute used by the adapter.

### torch is an extra since 0.6.0

With `tfc-t0` 0.6.0, `pip install tfc-t0` (the value of
`T0Adapter.backend_package`) does not install torch. `t0.T0Forecaster` is
resolved lazily from `t0.torch`, which fails on import:

```
ImportError: t0.torch is the PyTorch runtime of t0. Install through pip install
'tfc-t0[torch]' or use the MLX runtime with `from t0.mlx import T0Forecaster`
```

Installing the extra (`pip install "tfc-t0[torch]"`) fixes the import, and then
the first breakage shows up:

```
TypeError: got an unexpected keyword argument 'context'
```

Signature of `T0Forecaster.predict` in 0.6.0:
`predict(model_input, horizon, quantile_levels=(0.1, 0.5, 0.9), *,
context_length=None, future_covariates=None, mask=None, group_ids=None, ...)`.

Both were found with the backends compatibility check added in #1394
(`python tools/foundation_compat/run.py --adapter t0 --variant latest`).

## Workaround (0.26.0)

```bash
pip install "tfc-t0<0.4"
```

Verified: the user guide `foundation-forecasting-models.ipynb` runs end to end
with `tfc-t0==0.3.2`, `torch==2.9.1` (CPU) and skforecast 0.26.0 (backtesting
MAE 145.53).

## Proposed fix (next release)

1. Adapt `T0Adapter.predict` to the new API: pass the context positionally
   (`model_input`) and use `quantile_levels`. Require the torch extra and a
   minimum version (`backend_package = "tfc-t0[torch]>=0.6"`, the extra does
   not exist before 0.6.0, to be confirmed) and raise a clear `ImportError` when an
   older version is installed, as `TimesFM3Adapter._load_model` already does
   for `timesfm<3`.
2. Check that `future_covariates` keeps the same shape contract
   (`[batch, n_future_covariates, context + horizon]`) in 0.6.0, and that the
   left NaN padding of shorter series is still read as missing.
3. Update the fake `T0Forecaster` in
   `skforecast/foundation/tests/tests_foundation_models/fixtures_adapters.py`
   to the new signature, and the expected `backend_package` in
   `test_get_model_info.py`. Update also the call of `T0Forecaster.predict` in
   `tools/foundation_compat/tests/contracts.py` and the `t0` entry of
   `tools/foundation_compat/adapters.toml` (`backend_package` and `pinned`).
4. Update the install command in the user guide table, in
   `skills/foundation-forecasting/SKILL.md` and, if needed, in
   `tools/ai/llms-base.txt`; then regenerate the AI context files.
5. Remove the `tfc-t0<0.4` pin added as a workaround in #1384: the warning in
   the T0 section of `docs/user_guides/foundation-forecasting-models.ipynb`, the
   install command in its backends table and in
   `skills/foundation-forecasting/SKILL.md`. Re-execute the T0 cells of the
   user guide with the new backend version.
6. Release note in `docs/releases/releases.md`.

Alternative if the backend API keeps changing: pin `backend_package` to a
tested range (`tfc-t0[torch]>=0.6,<0.7`) and widen it after each check.
