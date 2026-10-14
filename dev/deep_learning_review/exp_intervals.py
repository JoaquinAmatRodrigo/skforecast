"""Coverage of ForecasterRnn conformal intervals: in-sample residuals vs
out-of-sample (validation) residuals. End-to-end with the public API."""
import os
os.environ["KERAS_BACKEND"] = "torch"
import contextlib
import io
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import keras
from skforecast.deep_learning import ForecasterRnn, create_and_compile_model
from skforecast.model_selection import backtesting_forecaster_multiseries, TimeSeriesFold
from common import DATA

L, H = 48, 24
b = pd.read_pickle(f"{DATA}/bike_sharing_extended_features.pkl")
cols = ["temp", "holiday", "hour_day_sin", "hour_day_cos", "week_day_sin", "week_day_cos"]
end_train, end_val = "2012-04-30 23:00", "2012-08-31 23:00"
series, exog = b[["users"]], b[cols]
keras.utils.set_random_seed(0)
with contextlib.redirect_stdout(io.StringIO()):
    model = create_and_compile_model(series=series, levels="users", lags=L, steps=H, exog=exog,
                                     compile_kwargs={"optimizer": "adam", "loss": "mse"})
f = ForecasterRnn(model, levels="users", lags=L, fit_kwargs={
    "epochs": 20, "batch_size": 256, "verbose": 0,
    "series_val": series.loc[:end_val].iloc[-(len(series.loc["2012-05-01":end_val]) + L):],
    "exog_val": exog.loc[:end_val].iloc[-(len(series.loc["2012-05-01":end_val]) + L):],
    "callbacks": [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]})
with contextlib.redirect_stdout(io.StringIO()):
    f.fit(series.loc[:end_train], exog=exog.loc[:end_train], store_in_sample_residuals=True)

# out-of-sample residuals: backtest over the validation period
cv_val = TimeSeriesFold(steps=H, initial_train_size=len(series.loc[:end_train]), refit=False)
with contextlib.redirect_stdout(io.StringIO()):
    _, pv = backtesting_forecaster_multiseries(f, series=series.loc[:end_val], exog=exog.loc[:end_val],
                                               cv=cv_val, metric="mean_absolute_error", show_progress=False)
y_true_val = series.loc[pv.index, "users"].to_numpy()
f.set_out_sample_residuals(y_true={"users": y_true_val}, y_pred={"users": pv["pred"].to_numpy()})

n_te = (len(series) - len(series.loc[:end_val])) // H * H
data, ex = series.iloc[:len(series.loc[:end_val]) + n_te], exog.iloc[:len(series.loc[:end_val]) + n_te]
cv = TimeSeriesFold(steps=H, initial_train_size=len(series.loc[:end_val]), refit=False)
for use_in in [True, False]:
    with contextlib.redirect_stdout(io.StringIO()):
        m, p = backtesting_forecaster_multiseries(f, series=data, exog=ex, cv=cv, interval=[0.1, 0.9], interval_method="conformal",
                                                  use_in_sample_residuals=use_in,
                                                  metric="mean_absolute_error", show_progress=False)
    yt = data.loc[p.index, "users"].to_numpy()
    cov = np.mean((yt >= p["lower_bound"]) & (yt <= p["upper_bound"]))
    width = np.mean(p["upper_bound"] - p["lower_bound"])
    print(f"residuals={'in-sample ' if use_in else 'validation'}  MAE={m['mean_absolute_error'].iloc[0]:.1f}  "
          f"coverage(80% nominal)={cov:.1%}  mean width={width:.1f}")
